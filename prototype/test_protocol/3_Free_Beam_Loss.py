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
from classes.Array_RealTime import apply_realtime_processing_chain
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, PedalboardAGC, AGCChain


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
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
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
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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
    sample_duration=0.8,
    output_dir='data/test_protocol/3_free_beam_loss',
    pattern=1,
    reference_angle=0,
    use_pipeline=True,
    wait_between_passes=False,
    freeze_beamformer=True,
    freeze_angle_deg=0.0,
    enable_agc=False,
    enable_spectral_filter=True,
    save_on_interrupt=False,
    process_block_ms=20.0,
):
    """
    Measure signal strength at a fixed set of angles using the processing pipeline.
    
    Args:
        num_passes: Number of repeated captures to take at each angle.
        angles: List of angles to measure. Defaults to the fixed base list requested.
        device_index: Audio device index for 4-channel array (None = use default input)
        sample_duration: Duration to record audio at each point in seconds.
        output_dir: Directory to save measurement data
        reference_angle: Reference offset applied to the fixed angle list.
        use_pipeline: Whether to apply full processing pipeline with beamformer (default: True)
        wait_between_passes: If True, pause between repeated passes.
        freeze_beamformer: If True, keep steering fixed to freeze_angle_deg for all measurements
        freeze_angle_deg: Steering angle used when freeze_beamformer is enabled
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
    if use_pipeline:
        print(f"  Processing chunk: {process_block_ms:.1f} ms ({process_block_samples} samples)")
    print(f"  Microphone reference angle: {reference_angle}°")
    print(f"  Total measurements: {num_passes * len(measurement_angles)}")
    print(f"  Output directory: {output_path}")
    if use_pipeline:
        print(f"  Processing: Full pipeline")
        print(f"    - Beamformer: MVDR")
        print(f"    - BandPass filter: ON")
        print(f"    - Spectral subtraction: {'ON' if enable_spectral_filter else 'OFF'}")
        print(f"    - AGC chain (both stages): {'ON' if enable_agc else 'OFF'}")
        print(f"  Beamformer freeze: {'ON' if freeze_beamformer else 'OFF'}")
        if freeze_beamformer:
            print(f"  Beamformer freeze angle: {float(freeze_angle_deg):.1f}°")
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
            normalize_channels=True,
            bootstrap_full_scan=True,
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
        
        print("Pipeline initialized.\n")
    else:
        beamformer = None
        doa_estimator = None
        filters = []
        agc = None

    def _process_capture_block(audio_block: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Process captured audio in short chunks to mimic realtime; optionally returns last DOA."""
        audio_block = np.asarray(audio_block, dtype=np.float32)

        if not use_pipeline:
            if audio_block.ndim == 2 and audio_block.shape[1] > 0:
                return np.asarray(audio_block[:, 0], dtype=np.float32).reshape(-1), None
            return np.asarray(audio_block, dtype=np.float32).reshape(-1), None

        if audio_block.ndim != 2:
            raise ValueError("Expected audio block with shape (samples, channels) when use_pipeline=True")

        if process_block_samples <= 0 or audio_block.shape[0] <= process_block_samples:
            if not bool(freeze_beamformer) and doa_estimator is not None and beamformer is not None:
                try:
                    doa_value = doa_estimator.estimate_doa(audio_block)
                    if doa_value is not None and hasattr(beamformer, "set_steering_angle"):
                        beamformer.set_steering_angle(float(doa_value))
                except Exception:
                    doa_value = None
            else:
                doa_value = None

            out = apply_realtime_processing_chain(
                block=audio_block,
                beamformer=beamformer,
                filters=filters,
                agc=agc,
                sample_rate=sample_rate,
                monitor_gain=1.0,
                theta_deg=None,
                freeze_beamformer=bool(freeze_beamformer),
                freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
            )
            return np.asarray(out, dtype=np.float32).reshape(-1), doa_value

        out_chunks: list[np.ndarray] = []
        last_doa: float | None = None
        n = audio_block.shape[0]
        for start in range(0, n, process_block_samples):
            chunk = audio_block[start:start + process_block_samples, :]
            if chunk.shape[0] == 0:
                continue

            if not bool(freeze_beamformer) and doa_estimator is not None and beamformer is not None:
                try:
                    doa_value = doa_estimator.estimate_doa(chunk)
                    if doa_value is not None and hasattr(beamformer, "set_steering_angle"):
                        beamformer.set_steering_angle(float(doa_value))
                    last_doa = float(doa_value) if doa_value is not None else last_doa
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
                freeze_beamformer=bool(freeze_beamformer),
                freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
            )
            out_chunks.append(np.asarray(out, dtype=np.float32).reshape(-1))

        if not out_chunks:
            return np.zeros(0, dtype=np.float32), last_doa

        return np.concatenate(out_chunks), last_doa
    
    # Storage for all measurements across all passes
    all_measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Warm up filters/AGC before measurements to avoid low initial gain frames.
    warmup_seconds = 2.0
    print(f"Warm-up: running {warmup_seconds:.1f}s pre-capture through processing chain...")
    interrupted = False
    try:
        warmup_audio = sd.rec(
            int(warmup_seconds * sample_rate),
            samplerate=sample_rate,
            channels=num_channels,
            device=device_index,
            blocking=True,
        )
        warmup_audio = np.asarray(warmup_audio).astype(np.float32)
        warmup_peak = np.max(np.abs(warmup_audio)) if warmup_audio.size > 0 else 0.0
        if warmup_peak > 1.0:
            warmup_audio = warmup_audio / warmup_peak

        if use_pipeline:
            _processed, _ = _process_capture_block(warmup_audio)
    except Exception as warmup_err:
        print(f"Warm-up warning: {type(warmup_err).__name__}: {warmup_err}")

    input("Warm-up complete. Press Enter to begin fixed-angle measurements...")

    print(f"Starting {num_passes}-pass fixed-angle measurements...")
    print("For each angle: press Enter to record, Backspace to undo last measurement")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for angle_idx, angle_deg in enumerate(measurement_angles):
            absolute_angle = (reference_angle + float(angle_deg)) % 360.0
            print(f"\n{'='*70}")
            print(f"ANGLE {angle_idx + 1}/{len(measurement_angles)}: {angle_deg:.1f}° (absolute {absolute_angle:.1f}°)")
            print(f"{'='*70}\n")
            
            for pass_num in range(num_passes):
                prompt = (
                    f"  Pass {pass_num + 1}/{num_passes} at {angle_deg:.1f}°: "
                    f"press Enter to record, Backspace to undo last... "
                )
                while True:
                    key_action = _wait_for_enter_or_backspace(prompt)
                    if key_action == 'enter':
                        break
                    if key_action == 'backspace':
                        if all_measurements:
                            removed_measurement = all_measurements.pop()
                            removed_angle = removed_measurement.get('expected_angle', removed_measurement.get('angle_deg', 0.0))
                            removed_pass = removed_measurement.get('pass_number', '?')
                            print(
                                f"  Backspace detected: removed last measurement (angle {removed_angle:.1f}°, pass {removed_pass})."
                            )
                        else:
                            print("  Backspace detected: no previous measurement to remove.")
                        print(f"  Waiting for Enter to record pass {pass_num + 1}/{num_passes}...\n")
                        continue

                try:
                    audio_data = sd.rec(
                        int(sample_duration * sample_rate),
                        samplerate=sample_rate,
                        channels=num_channels,
                        device=device_index,
                        blocking=True,
                    )
                except Exception as e:
                    logger.error(f"Error recording audio at angle {absolute_angle:.1f}°, pass {pass_num + 1}: {e}")
                    continue

                audio_data = np.asarray(audio_data).astype(np.float32)
                max_val = np.max(np.abs(audio_data)) if audio_data.size > 0 else 0.0
                if max_val > 1.0:
                    audio_data = audio_data / max_val

                raw_mono = audio_data[:, 0] if audio_data.ndim > 1 else audio_data
                input_rms_level = float(np.sqrt(np.mean(raw_mono ** 2)))
                input_rms_dbfs = _safe_dbfs(input_rms_level)

                # More angle-invariant input reference: average power across all mic channels.
                # This avoids the "gain" peaking just because mic channel 0 happens to be
                # in a null for a given rotation angle.
                if audio_data.ndim == 2 and audio_data.shape[1] > 0:
                    input_power_avgch = float(np.mean(audio_data ** 2))
                    input_rms_level_avgch = float(np.sqrt(max(input_power_avgch, 0.0)))
                else:
                    input_rms_level_avgch = input_rms_level
                input_rms_dbfs_avgch = _safe_dbfs(input_rms_level_avgch)

                if use_pipeline:
                    # Reset DOA state between captures so a large reposition does not require
                    # many small hill-climb updates to reacquire.
                    if not bool(freeze_beamformer) and doa_estimator is not None:
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

                    processed_audio, doa_deg = _process_capture_block(audio_data)
                else:
                    processed_audio = np.asarray(raw_mono, dtype=np.float32)
                    doa_deg = None

                processed_audio = np.asarray(processed_audio, dtype=np.float32)
                rms_level = float(np.sqrt(np.mean(processed_audio ** 2)))
                peak_level = float(np.max(np.abs(processed_audio))) if processed_audio.size > 0 else 0.0
                rms_dbfs = _safe_dbfs(rms_level)
                peak_dbfs = _safe_dbfs(peak_level)
                # Diagnostic: output/input ratio (depends on input channel level).
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
        
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
        interrupted = True
        if len(all_measurements) == 0:
            print("No data to save.")
            return None

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
    
    parser.add_argument('--duration', type=float, default=0.8,
                        help='Audio sample duration in seconds (default: 0.8)')
    parser.add_argument('--output', type=str, default='data/test_protocol/3_free_beam_loss',
                        help='Output directory (default: data/test_protocol/3_free_beam_loss)')
    parser.add_argument('--reference-angle', type=int, default=0,
                        help='Initial microphone pointing direction (default: 0°)')
    parser.add_argument('--no-pipeline', action='store_true',
                        help='Disable processing pipeline, record raw audio only')
    parser.add_argument('--wait-between-passes', action=argparse.BooleanOptionalAction, default=True,
                        help='Wait for Enter key before starting each new pass (default: enabled)')
    parser.add_argument('--freeze-beamformer', action=argparse.BooleanOptionalAction, default=False,
                        help='Freeze beamformer steering during the full measurement run (default: disabled)')
    parser.add_argument('--freeze-angle', type=float, default=0.0,
                        help='Steering angle used when beamformer freeze is enabled (default: 0.0)')
    parser.add_argument('--process-block-ms', type=float, default=20.0,
                        help='Processing chunk size in ms (default: 20.0). Important for overlap-add filters.')
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
        output_dir=args.output,
        reference_angle=args.reference_angle,
        use_pipeline=not args.no_pipeline,
        wait_between_passes=args.wait_between_passes,
        freeze_beamformer=args.freeze_beamformer,
        freeze_angle_deg=args.freeze_angle,
        enable_agc=args.enable_agc,
        enable_spectral_filter=args.enable_spectral_filter,
        save_on_interrupt=args.save_on_interrupt,
        process_block_ms=args.process_block_ms,
    )
    
    # Usage examples:
    #
    # Standard measurement with the fixed angle list:
    # python prototype/test_protocol/3_Free_Beam_Loss.py --device 1 --passes 3 --duration 0.8
    #
    # Override the angle list if needed:
    # python prototype/test_protocol/3_Free_Beam_Loss.py --device 1 --angles 0,15,30,45,90,180,270,315,330,345