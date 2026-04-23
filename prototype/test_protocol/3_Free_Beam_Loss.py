"""
SNR Signal Test Protocol
========================

Measures signal strength at a fixed list of angles and saves per-angle gain data
for later plotting of SNR or level-vs-angle graphs.

Combines:
- Beamformer: MVDR with adaptive smoothing and coherence-based sidelobe suppression
- Pipeline: Same processing chain as test_pipeline.py (Beamformer + Filters + AGC)
- Methodology: Manual angle-by-angle captures using a fixed measurement list
- Output: CSV format with input/output RMS, peak levels, gain, angles, timestamps
- Audio: 4-channel array input for beamforming at each measurement point
"""

from datetime import datetime
from pathlib import Path
import argparse
import time
import math
import logging
import sys

try:
    import msvcrt
except ImportError:
    msvcrt = None

import numpy as np
import pandas as pd
import sounddevice as sd

# Import processing classes from the shared codebase
sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.Microphone import Microphone
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Array_RealTime import Array_RealTime, apply_realtime_processing_chain
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, PedalboardAGC, AGCChain
from classes.Codec import G711Codec


def _safe_dbfs(value: float, floor: float = 1e-10) -> float:
    return 20.0 * np.log10(max(float(value), floor))


def _wait_for_enter_or_backspace(prompt: str) -> str:
    """Wait for Enter or Backspace on Linux and Windows.

    Returns:
        "enter" if Enter was pressed.
        "backspace" if Backspace was pressed.
    """
    print(prompt, end="", flush=True)

    if msvcrt is not None:
        kbhit = getattr(msvcrt, 'kbhit', None)
        getwch = getattr(msvcrt, 'getwch', None)
        if callable(kbhit) and callable(getwch):
            while True:
                if not kbhit():
                    time.sleep(0.01)
                    continue
                key = getwch()
                if key in ('\r', '\n'):
                    print()
                    return 'enter'
                if key in ('\x08', '\x7f'):
                    print()
                    return 'backspace'
        # Fall through to standard input if the Windows console helpers are unavailable.

    try:
        import termios
        import tty
        import select
    except ImportError:
        response = input()
        return 'backspace' if response == '\b' else 'enter'

    fd = sys.stdin.fileno()
    tcgetattr = getattr(termios, 'tcgetattr', None)
    tcsetattr = getattr(termios, 'tcsetattr', None)
    tcsadrain = getattr(termios, 'TCSADRAIN', None)
    setraw = getattr(tty, 'setraw', None)
    if not (callable(tcgetattr) and callable(tcsetattr) and tcsadrain is not None and callable(setraw)):
        response = input()
        return 'backspace' if response == '\b' else 'enter'

    old_settings = tcgetattr(fd)
    try:
        setraw(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in ('\r', '\n'):
                print()
                return 'enter'
            if key in ('\x7f', '\b'):
                print()
                return 'backspace'
    finally:
        tcsetattr(fd, tcsadrain, old_settings)


DEFAULT_MEASUREMENT_ANGLES = [0, 15, 30, 45, 90, 180, 270, 315, 330, 345]


def _parse_angle_list(raw_value: str) -> list[float]:
    angles: list[float] = []
    for part in raw_value.split(','):
        text = part.strip()
        if not text:
            continue
        angles.append(float(text))
    if not angles:
        raise ValueError('Angle list cannot be empty')
    return angles


def test_di_signal(
    num_passes=1,
    angles=None,
    device_index=None,
    sample_duration=2.0,
    settle_seconds=1.0,
    output_dir='data/test_protocol/3_free_beam_loss',
    pattern=1,
    reference_angle=0,
    use_pipeline=True,
    wait_between_passes=False,
    enable_agc=False,
    enable_spectral_filter=True,
    save_on_interrupt=False,
    process_block_ms=20.0,
    doa_min_confidence_db: float | None = 1.0,
    reset_doa_each_capture: bool = False,
):
    """
    Measure signal strength at a fixed set of angles with automatic DOA-based beamforming.
    
    Args:
        num_passes: Number of repeated captures to take at each angle.
        angles: List of angles to measure. Defaults to the fixed base list requested.
        device_index: Audio device index for 4-channel array (None = use default input)
        sample_duration: Duration to record audio at each point in seconds.
                Default 2.0s to allow DOA estimator time to converge to accurate steering
                (at update_rate=3.0 Hz, this gives ~6 updates for convergence).
        settle_seconds: Live settle time before each measurement window.
        output_dir: Directory to save measurement data
        reference_angle: Reference offset applied to the fixed angle list.
        use_pipeline: Whether to apply full processing pipeline with beamformer (default: True)
        wait_between_passes: If True, pause between repeated passes.
        enable_agc: If True, enable both AGC stages (AdaptiveAmplifier + PedalboardAGC).
                Default False for linear directivity measurements.
        enable_spectral_filter: If True, enable SpectralSubtractionFilter (default True).
        save_on_interrupt: If True, save partial data when interrupted by Ctrl+C.
                  Default False to avoid saving incomplete measurements.
    
    Returns:
        DataFrame with averaged measurements
    """
    measurement_angles = list(angles) if angles is not None else list(DEFAULT_MEASUREMENT_ANGLES)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get device info
    if device_index is None:
        device_info = sd.query_devices(kind='input')
        device_name = device_info['name']
        device_index = device_info['index']
    else:
        device_info = sd.query_devices(device_index)
        device_name = device_info['name']
    
    sample_rate = int(device_info['default_samplerate'])
    
    # Use 4 channels for array processing; keep single-channel raw mode.
    num_channels = 4 if use_pipeline else 1

    # Process captured audio using short, fixed-size blocks so overlap-add filters behave like realtime.
    process_block_ms = float(process_block_ms)
    process_block_samples = int(max(32, round((process_block_ms / 1000.0) * sample_rate)))
    capture_samples = int(max(1, round(sample_duration * sample_rate)))
    process_block_samples = int(min(process_block_samples, capture_samples))
    
    print(f"\n{'='*70}")
    print(f"SNR Signal Test Protocol Configuration:")
    print(f"{'='*70}")
    print(f"  Audio device: {device_name} (index {device_index})")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {num_channels} (array-based beamforming)")
    print(f"  Number of passes: {num_passes}")
    print(f"  Fixed measurement angles: {measurement_angles}")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  Live settle time: {settle_seconds} seconds")
    if use_pipeline:
        doa_updates_est = int(sample_duration * 3.0)  # 3.0 Hz update rate
        print(f"    → Estimated DOA updates per capture: ~{doa_updates_est} (sufficient for convergence)")
    if use_pipeline:
        print(f"  Processing chunk: {process_block_ms:.1f} ms ({process_block_samples} samples)")
    print(f"  Microphone reference angle: {reference_angle}°")
    print(f"  Total measurements: {num_passes * len(measurement_angles)}")
    print(f"  Output directory: {output_path}")
    if use_pipeline:
        print(f"  Processing: Full pipeline")
        print(f"    - Beamformer: MVDR (free with DOA tracking)")
        print(f"    - DOA Estimator: ON (updates at 3.0 Hz)")
        print(f"    - BandPass filter: ON")
        print(f"    - Spectral subtraction: {'ON' if enable_spectral_filter else 'OFF'}")
        print(f"    - AGC chain (both stages): {'ON' if enable_agc else 'OFF'}")
    else:
        print(f"  Processing: Raw audio only")
    print(f"  Save on interrupt: {'ON' if save_on_interrupt else 'OFF (default)'}")
    print(f"{'='*70}\n")
    
    # Setup logging
    logger = logging.getLogger("DISignalTest")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Initialize processing pipeline if requested
    if use_pipeline:
        print("Initializing audio processing pipeline...")
        
        # Microphone array geometry (same source as realtime pipeline)
        mic_channel_numbers = [0, 1, 2, 3]
        
        if pattern == 1:
            pattern = "1_square.xml"
        
        elif pattern == 2:
            pattern = "2_corners.xml"
            
        elif pattern == 3:
            pattern = "3_large.xml"
            
        else:
            raise ValueError(f"Invalid pattern value: {pattern}. Must be 1, 2, or 3.")
        
        geometry_path = Path(__file__).resolve().parent.parent / "array_geometries" / pattern
        mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
        
        # MVDR Beamformer (same config as test_pipeline.py)
        beamformer = MVDRBeamformer(
            logger=logger,
            mic_channel_numbers=mic_channel_numbers,
            sample_rate=sample_rate,
            mic_positions_m=mic_positions,
            covariance_alpha=0.95,
            diagonal_loading=0.15,
            spectral_whitening_factor=0.12,
            weight_smooth_alpha=0.72,
            max_adaptive_loading_scale=4.0,
            coherence_suppression_strength=0.8,
            weight_smooth_alpha_min=0.45,
            weight_smooth_alpha_max=0.82,
            snr_threshold_for_sharpening=2.0,
            backward_null_strength=0.9,
        )

        # DOA estimator (mirrors Array_RealTime behavior when beamformer is unfrozen)
        # Use a DAS beamformer for fast scanning.
        doa_estimator = IterativeDOAEstimator(
            logger=logger,
            update_rate=3.0,
            angle_range=(-25.0, 25.0),
            beamformer=DASBeamformer(
                logger=logger,
                mic_channel_numbers=mic_channel_numbers,
                sample_rate=sample_rate,
                mic_positions_m=mic_positions,
            ),
            scan_step_deg=5.0,
            local_search_radius_deg=10.0,
            periodic_full_scan_blocks=20,
        )
        
        # Filters (same config as test_pipeline.py)
        filters: list[object] = [
            BandPassFilter(
                logger=logger,
                sample_rate=sample_rate,
                low_cutoff=300.0,
                high_cutoff=4000.0,
                order=4
            ),
        ]

        if enable_spectral_filter:
            filters.append(
                SpectralSubtractionFilter(
                    logger=logger,
                    sample_rate=sample_rate,
                    noise_factor=0.65,
                    gain_floor=0.55,  # Updated from 0.35 (prevents over-suppression of low freqs)
                    noise_alpha=0.995,
                    noise_update_snr_db=8.0,
                    gain_smooth_alpha=0.92,
                )
            )
        
        # AGC chain (same config as test_pipeline.py)
        if enable_agc:
            agc = AGCChain(logger=logger, stages=[
                AdaptiveAmplifier(
                    logger=logger,
                    target_rms=0.08,
                    min_gain=1.0,
                    max_gain=6.0,  # Updated from 12.0 (prevent oscillation)
                    adapt_alpha=0.04,
                    speech_activity_rms=0.00012,
                    silence_decay_alpha=0.008,
                    activity_hold_ms=600.0,
                    peak_protect_threshold=0.30,  # Updated from 0.35
                    peak_protect_strength=1.0,  # Updated from 0.85 (maximum protection)
                    max_gain_warn_rms_min=0.001,
                ),
                PedalboardAGC(
                    logger=logger,
                    sample_rate=sample_rate,
                    threshold_db=-20.0,
                    ratio=2.0,  # Updated from 3.5 (gentler compression)
                    attack_ms=3.0,
                    release_ms=140.0,
                    limiter_threshold_db=-7.0,  # Updated from -1.4 (much lower headroom)
                    limiter_release_ms=50.0  # Updated from 100.0
                ),
            ])
        else:
            agc = None
            
        codec = G711Codec(logger=logger)

        mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
        array = Array_RealTime(
            id_vendor=0x2752,
            id_product=0x0019,
            logger=logger,
            mic_list=mic_list,
            sampling_rate=sample_rate,
            doa_estimator=doa_estimator,
            beamformer=beamformer,
            echo_canceller=None,
            filters=filters,
            agc=agc,
            codec=codec,
            monitor_gain=1.0,
            output_mode="local",
            output_boundary_fade_ms=0.0,
            downsample_rate=None,
            initial_silence_duration=0.0,
        )
        array.start_realtime(blocksize=process_block_samples)
        
        print("Pipeline initialized.\n")
    else:
        beamformer = None
        doa_estimator = None
        filters = []
        agc = None
        array = None

    def _process_capture_block(audio_block: np.ndarray) -> tuple[np.ndarray, float | None, float | None]:
        """Process captured audio in short chunks to mimic realtime; optionally returns last DOA/confidence."""
        audio_block = np.asarray(audio_block, dtype=np.float32)

        if not use_pipeline:
            if audio_block.ndim == 2 and audio_block.shape[1] > 0:
                return np.asarray(audio_block[:, 0], dtype=np.float32).reshape(-1), None, None
            return np.asarray(audio_block, dtype=np.float32).reshape(-1), None, None

        if audio_block.ndim != 2:
            raise ValueError("Expected audio block with shape (samples, channels) when use_pipeline=True")

        if process_block_samples <= 0 or audio_block.shape[0] <= process_block_samples:
            if doa_estimator is not None and beamformer is not None:
                try:
                    doa_value = doa_estimator.estimate_doa(audio_block)
                    doa_conf = getattr(doa_estimator, 'latest_confidence_db', None)
                    if (
                        doa_value is not None
                        and hasattr(beamformer, "set_steering_angle")
                        and (doa_min_confidence_db is None or (doa_conf is not None and doa_conf >= float(doa_min_confidence_db)))
                    ):
                        beamformer.set_steering_angle(float(doa_value))
                except Exception:
                    doa_value = None
                    doa_conf = None
            else:
                doa_value = None
                doa_conf = None

            out = apply_realtime_processing_chain(
                block=audio_block,
                beamformer=beamformer,
                filters=filters,
                agc=agc,
                sample_rate=sample_rate,
                monitor_gain=1.0,
                theta_deg=None,
                freeze_beamformer=False,
                freeze_angle_deg=None,
            )
            return np.asarray(out, dtype=np.float32).reshape(-1), doa_value, doa_conf

        out_chunks: list[np.ndarray] = []
        last_doa: float | None = None
        last_conf: float | None = None
        n = audio_block.shape[0]
        for start in range(0, n, process_block_samples):
            chunk = audio_block[start:start + process_block_samples, :]
            if chunk.shape[0] == 0:
                continue

            if doa_estimator is not None and beamformer is not None:
                try:
                    doa_value = doa_estimator.estimate_doa(chunk)
                    doa_conf = getattr(doa_estimator, 'latest_confidence_db', None)
                    if (
                        doa_value is not None
                        and hasattr(beamformer, "set_steering_angle")
                        and (doa_min_confidence_db is None or (doa_conf is not None and doa_conf >= float(doa_min_confidence_db)))
                    ):
                        beamformer.set_steering_angle(float(doa_value))
                        last_doa = float(doa_value)
                        last_conf = float(doa_conf) if doa_conf is not None else last_conf
                except Exception:
                    pass

            out = apply_realtime_processing_chain(
                block=chunk,
                beamformer=beamformer,
                filters=filters,
                agc=agc,
                sample_rate=sample_rate,
                monitor_gain=1.0,
                theta_deg=None,
                freeze_beamformer=False,
                freeze_angle_deg=None,
            )
            out_chunks.append(np.asarray(out, dtype=np.float32).reshape(-1))

        if not out_chunks:
            return np.zeros(0, dtype=np.float32), last_doa, last_conf

        return np.concatenate(out_chunks), last_doa, last_conf
    
    # Storage for all measurements across all passes
    all_measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Warm up filters/AGC before measurements to avoid low initial gain frames.
    warmup_seconds = 2.0
    print(f"Warm-up: letting the live processing chain settle for {warmup_seconds:.1f}s...")
    interrupted = False
    try:
        if use_pipeline and array is not None:
            time.sleep(warmup_seconds)
    except KeyboardInterrupt:
        print("Warm-up interrupted by user. Exiting.")
        return None
    except Exception as warmup_err:
        print(f"Warm-up warning: {type(warmup_err).__name__}: {warmup_err}")

    input("Warm-up complete. Press Enter to begin fixed-angle measurements...")

    print(f"Starting {num_passes}-pass fixed-angle measurements...")
    print("For each angle: press Enter to start a live measurement window, Backspace to undo the last result")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for angle_idx, angle_deg in enumerate(measurement_angles):
            absolute_angle = (reference_angle + float(angle_deg)) % 360.0
            print(f"\n{'='*70}")
            print(f"ANGLE {angle_idx + 1}/{len(measurement_angles)}: {angle_deg:.1f}° (absolute {absolute_angle:.1f}°)")
            print(f"{'='*70}\n")
            
            skip_angle = False
            for pass_num in range(num_passes):
                if skip_angle:
                    break  # Skip to next angle
                
                retry_pass = True  # Allow retries for this pass
                while retry_pass:
                    retry_pass = False  # Reset for next iteration
                    
                    input(f"  Pass {pass_num + 1}/{num_passes} at {angle_deg:.1f}°: press Enter to record...")

                    try:
                        if not use_pipeline or array is None:
                            raise RuntimeError("Protocol 3 live side-door mode requires use_pipeline=True")

                        if reset_doa_each_capture and doa_estimator is not None:
                            doa_estimator.latest_doa = None
                            if hasattr(doa_estimator, "_last_update_time"):
                                try:
                                    doa_estimator._last_update_time = 0.0
                                except Exception:
                                    pass
                            if hasattr(doa_estimator, "_latest_gain"):
                                try:
                                    doa_estimator._latest_gain = None
                                except Exception:
                                    pass

                        if settle_seconds > 0:
                            print(f"    Settling live stream for {settle_seconds:.1f}s...")
                            time.sleep(settle_seconds)

                        array.start_side_door_measurement()
                        try:
                            print(f"    Measuring live stream for {sample_duration:.1f}s...", end="", flush=True)
                            measure_started = time.monotonic()
                            while (time.monotonic() - measure_started) < sample_duration:
                                time.sleep(0.05)
                            snapshot = array.get_side_door_measurement_snapshot(reset=False)
                        finally:
                            array.stop_side_door_measurement()
                        print()

                        input_rms_level = float(snapshot["input_rms"])
                        input_rms_dbfs = float(snapshot["input_rms_dbfs"])
                        input_rms_level_avgch = float(snapshot["input_allch_rms"])
                        input_rms_dbfs_avgch = float(snapshot["input_allch_rms_dbfs"])
                        rms_level = float(snapshot["output_rms"])
                        rms_dbfs = float(snapshot["output_rms_dbfs"])
                        peak_level = float(snapshot["output_peak"])
                        peak_dbfs = float(snapshot["output_peak_dbfs"])
                        doa_deg = snapshot["doa_deg"]
                        doa_conf_db = snapshot["doa_conf_db"]

                    except KeyboardInterrupt:
                        print("\nMeasurement interrupted by user")
                        interrupted = True
                        raise
                    except Exception as e:
                        logger.error(f"Error measuring angle {absolute_angle:.1f}°, pass {pass_num + 1}: {e}")
                        continue

                    gain_inout_db = rms_dbfs - input_rms_dbfs
                    gain_inout_db_avgch = rms_dbfs - input_rms_dbfs_avgch

                    measurement = {
                        'measurement_index': len(all_measurements),
                        'pass_number': pass_num + 1,
                        'angle_deg': float(angle_deg),
                        'expected_angle': absolute_angle,
                        'timestamp': datetime.now().isoformat(),
                        'reference_angle': reference_angle,
                        'doa_deg': float(doa_deg) if doa_deg is not None else np.nan,
                        'doa_conf_db': float(doa_conf_db) if doa_conf_db is not None else np.nan,
                        'input_rms_level': input_rms_level,
                        'input_rms_dbfs': input_rms_dbfs,
                        'input_rms_level_avgch': input_rms_level_avgch,
                        'input_rms_dbfs_avgch': input_rms_dbfs_avgch,
                        'rms_level': rms_level,
                        'rms_dbfs': rms_dbfs,
                        'peak_level': peak_level,
                        'peak_dbfs': peak_dbfs,
                        'gain_inout_db': gain_inout_db,
                        'gain_inout_db_avgch': gain_inout_db_avgch,
                    }

                    all_measurements.append(measurement)

                    print(
                        f"    ✓ Pass {pass_num + 1}/{num_passes} | "
                        f"Input: {input_rms_dbfs:7.2f} dBFS | Output: {rms_dbfs:7.2f} dBFS | "
                        f"Out/In: {gain_inout_db:+7.2f} dB"
                    )
                    
                    # Offer option to reject this measurement after it's recorded
                    while True:
                        accept_input = input("    Keep? (press Enter) or Backspace to redo/skip/abort >> ").strip().lower()
                        if accept_input == '' or accept_input == 'y':
                            # Measurement accepted, move on
                            break
                        elif accept_input == 'backspace' or accept_input == 'b':
                            # Reject and offer options
                            removed_measurement = all_measurements.pop()
                            print(f"    Measurement rejected and removed.")
                            
                            while True:
                                option_input = input("    Options: 'r'=re-record this pass, 's'=skip this angle, 'a'=abort >> ").strip().lower()
                                if option_input == 'r':
                                    print(f"    Re-recording pass {pass_num + 1}/{num_passes}...\n")
                                    retry_pass = True  # Will re-enter the recording loop for this pass
                                    break
                                elif option_input == 's':
                                    print(f"    Skipping remaining passes for {angle_deg:.1f}°\n")
                                    skip_angle = True
                                    break
                                elif option_input == 'a':
                                    print(f"    Aborting measurement. {len(all_measurements)} measurements saved so far.")
                                    interrupted = True
                                    raise KeyboardInterrupt
                                else:
                                    print("    Invalid. Type 'r', 's', or 'a'.")
                            
                            if skip_angle or retry_pass:
                                break  # Exit accept loop - either skip angle or re-record
                        else:
                            print("    Press Enter to accept or type 'b' to backspace.")
        
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
        interrupted = True
        if len(all_measurements) == 0:
            print("No data to save.")
            return None

    if array is not None and getattr(array, "is_running", False):
        array.stop_realtime()

    if interrupted and not save_on_interrupt:
        print("Interrupted run detected. Partial data discarded (save_on_interrupt is OFF).")
        return None
    
    # Convert all measurements to DataFrame
    df_all = pd.DataFrame(all_measurements)

    # Match Protocol 1 plotting convention: express gain as output RMS relative to the run's peak.
    # This guarantees the maximum response is 0 dB and other angles are negative.
    if not df_all.empty and 'rms_dbfs' in df_all.columns:
        ref_rms_dbfs = float(pd.to_numeric(df_all['rms_dbfs'], errors='coerce').max())
        df_all['gain_db'] = pd.to_numeric(df_all['rms_dbfs'], errors='coerce') - ref_rms_dbfs

    # Save raw data with all passes
    raw_csv_file = output_path / f"angle_gain_{test_timestamp}.csv"
    df_all.to_csv(raw_csv_file, index=False)
    print(f"\n{'='*70}")
    print(f"Raw data saved to: {raw_csv_file}")
    
    # Calculate averaged results by angle
    print("Averaging measurements across passes...")
    if df_all.empty:
        print("No measurements collected.")
        return None

    grouped = df_all.groupby(['angle_deg', 'expected_angle'], as_index=False).agg({
        'measurement_index': 'min',
        'pass_number': 'count',
        'reference_angle': 'first',
        'doa_deg': 'mean',
        'doa_conf_db': 'mean',
        'input_rms_level': 'mean',
        'input_rms_dbfs': 'mean',
        'input_rms_level_avgch': 'mean',
        'input_rms_dbfs_avgch': 'mean',
        'rms_level': 'mean',
        'rms_dbfs': ['mean', 'std'],
        'peak_level': 'mean',
        'peak_dbfs': 'mean',
        'gain_inout_db': ['mean', 'std'],
        'gain_inout_db_avgch': ['mean', 'std'],
    })

    grouped.columns = [
        'angle_deg',
        'expected_angle',
        'first_measurement_index',
        'num_passes',
        'reference_angle',
        'doa_deg',
        'doa_conf_db',
        'input_rms_level',
        'input_rms_dbfs',
        'input_rms_level_avgch',
        'input_rms_dbfs_avgch',
        'rms_level',
        'rms_dbfs',
        'rms_dbfs_std',
        'peak_level',
        'peak_dbfs',
        'gain_inout_db',
        'gain_inout_db_std',
        'gain_inout_db_avgch',
        'gain_inout_db_avgch_std',
    ]

    # Recompute gain_db on the averaged table as "relative RMS" (0 dB at the best angle).
    if not grouped.empty:
        ref_rms_dbfs_avg = float(pd.to_numeric(grouped['rms_dbfs'], errors='coerce').max())
        grouped['gain_db'] = pd.to_numeric(grouped['rms_dbfs'], errors='coerce') - ref_rms_dbfs_avg
        grouped['gain_db_std'] = pd.to_numeric(grouped.get('rms_dbfs_std', np.nan), errors='coerce')

    grouped.sort_values('angle_deg', inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    averaged_csv_file = output_path / f"angle_gain_averaged_{test_timestamp}.csv"
    grouped.to_csv(averaged_csv_file, index=False)

    gain_min = grouped['gain_db'].min()
    gain_max = grouped['gain_db'].max()
    gain_range = gain_max - gain_min
    gain_min_angle = grouped.loc[grouped['gain_db'].idxmin(), 'expected_angle']
    gain_max_angle = grouped.loc[grouped['gain_db'].idxmax(), 'expected_angle']

    print(f"\n{'='*70}")
    print(f"Fixed-angle measurement complete!")
    print(f"{'='*70}")
    print(f"  Total measurements: {len(df_all)} ({len(grouped)} angles × {num_passes} passes)")
    print(f"\n  Gain (relative RMS, ref = peak angle):")
    print(f"    Range: {gain_range:.2f} dB ({gain_min:.2f} to {gain_max:.2f} dB; max should be ~0 dB)")
    print(f"    Maximum at: {gain_max_angle:.1f}°")
    print(f"    Minimum at: {gain_min_angle:.1f}°")
    print(f"\n  Data saved to:")
    print(f"    Raw data (all passes):  {raw_csv_file}")
    print(f"    Averaged results:       {averaged_csv_file}")
    print(f"{'='*70}\n")
    
    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SNR Signal Test Protocol - Measure fixed-angle signal strength with full audio processing')
    
    parser.add_argument('--passes', type=int, default=1,
                        help='Number of repeated captures per angle (default: 1)')
    parser.add_argument('--angles', type=str, default=','.join(str(v) for v in DEFAULT_MEASUREMENT_ANGLES),
                        help='Comma-separated angle list in degrees (default fixed list)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: system default input device)')
    
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Audio sample duration in seconds (default: 2.0). '
                             'Longer durations allow DOA estimator to converge to accurate steering. '
                             'Use 0.8 for quick tests (faster but less accurate DOA convergence).')
    parser.add_argument('--settle-seconds', type=float, default=1.0,
                        help='Live settle time before each measurement window in seconds (default: 1.0)')
    parser.add_argument('--output', type=str, default='data/test_protocol/3_free_beam_loss',
                        help='Output directory (default: data/test_protocol/3_free_beam_loss)')
    parser.add_argument('--reference-angle', type=int, default=0,
                        help='Initial microphone pointing direction (default: 0°)')
    parser.add_argument('--no-pipeline', action='store_true',
                        help='Disable processing pipeline, record raw audio only')
    parser.add_argument('--wait-between-passes', action=argparse.BooleanOptionalAction, default=True,
                        help='Wait for Enter key before starting each new pass (default: enabled)')
    parser.add_argument('--process-block-ms', type=float, default=20.0,
                        help='Processing chunk size in ms (default: 20.0). Important for overlap-add filters.')
    parser.add_argument('--doa-min-confidence-db', type=float, default=1.0,
                        help='Minimum DOA confidence in dB (best-vs-second-best beam power) required to update steering. '
                             'Use 0 to disable gating; increase to reduce DOA jitter in diffuse/reverberant noise. (default: 1.0)')
    parser.add_argument('--reset-doa-each-capture', action=argparse.BooleanOptionalAction, default=False,
                        help='Reset DOA estimator state before each manual capture (default: disabled). '
                             'Enable only if you frequently reposition far and want fast re-acquisition.')
    parser.add_argument('--enable-agc', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable both AGC stages (AdaptiveAmplifier + PedalboardAGC) (default: disabled)')
    parser.add_argument('--enable-spectral-filter', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable spectral subtraction filter (default: enabled)')
    parser.add_argument('--save-on-interrupt', action=argparse.BooleanOptionalAction, default=False,
                        help='Save partial data when interrupted by Ctrl+C (default: disabled)')
    
    args = parser.parse_args()

    angle_list = _parse_angle_list(args.angles)
    
    # Run test
    test_di_signal(
        num_passes=args.passes,
        angles=angle_list,
        device_index=args.device,
        sample_duration=args.duration,
        settle_seconds=args.settle_seconds,
        output_dir=args.output,
        reference_angle=args.reference_angle,
        use_pipeline=not args.no_pipeline,
        wait_between_passes=args.wait_between_passes,
        enable_agc=args.enable_agc,
        enable_spectral_filter=args.enable_spectral_filter,
        save_on_interrupt=args.save_on_interrupt,
        process_block_ms=args.process_block_ms,
        doa_min_confidence_db=(None if float(args.doa_min_confidence_db) <= 0.0 else float(args.doa_min_confidence_db)),
        reset_doa_each_capture=bool(args.reset_doa_each_capture),
    )
    
    # Usage examples:
    #
    # Standard measurement with the fixed angle list:
    # python prototype/test_protocol/3_Free_Beam_Loss.py --device 1 --passes 3 --duration 0.8
    #
    # Override the angle list if needed:
    # python prototype/test_protocol/3_Free_Beam_Loss.py --device 1 --angles 0,15,30,45,90,180,270,315,330,345