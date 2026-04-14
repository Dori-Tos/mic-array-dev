"""
Polar Pattern Test Protocol
===========================

Measures the polar pattern (directivity) of a microphone array using the full audio processing pipeline.
Performs multi-pass measurements at different angles, applies the complete beamforming and filtering chain,
and saves results in standardized CSV format for analysis and comparison.

Combines:
- Beamformer: MVDR with adaptive smoothing and coherence-based sidelobe suppression
- Pipeline: Same processing chain as test_pipeline.py (Beamformer + Filters + AGC)
- Methodology: Dynamic microphone measurement approach (rotating array, recording at each angle)
- Approach: Multi-pass measurements with averaging (like test_directivity_multipass.py)
- Output: Standardized CSV format with RMS, peak levels, angles, timestamps
- Audio: 4-channel array input for beamforming at each measurement point
"""

from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
import argparse
import os
import select
import time
import math
import logging
import sys

if os.name == 'nt':
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
else:
    msvcrt = None
    import termios
    import tty

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


def _consume_backspace_request() -> bool:
    """Return True if Backspace was pressed since the last poll."""
    if os.name == 'nt':
        if msvcrt is None:
            return False
        requested = False
        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b'\x08', b'\x7f'):
                requested = True
        return requested

    if not sys.stdin.isatty():
        return False

    requested = False
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        key = sys.stdin.read(1)
        if key in ('\x08', '\x7f'):
            requested = True
    return requested


@contextmanager
def _pass_keyboard_mode():
    """Enable key polling during a pass without affecting normal prompts."""
    if os.name != 'nt' and sys.stdin.isatty():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        yield


def test_polar_pattern(
    num_passes=3,
    resolution=50,
    seconds_per_rotation=120,
    device_index=None,
    sample_duration=0.8,
    rotation_direction='counterclockwise',
    output_dir='data/polar_pattern',
    reference_angle=0,
    use_pipeline=True,
    wait_between_passes=False,
    quarter_rotation=False,
    edge_padding_points=2,
    alternate_rotation_direction=True,
    freeze_beamformer=True,
    freeze_angle_deg=0.0,
    enable_agc=False,
    enable_spectral_filter=True,
    save_on_interrupt=False,
):
    """
    Measure microphone array polar pattern using the complete processing pipeline.
    
    Performs multi-pass directivity measurements with full audio processing at each angle:
    - Beamforming: MVDR with adaptive smoothing and coherence-based sidelobe suppression
    - Filtering: BandPass (300-4000 Hz) + Spectral Subtraction
    - AGC chain: Adaptive Amplifier + Pedalboard AGC
    
    Records 4-channel audio from microphone array at each measurement point and applies
    the complete processing pipeline matching test_pipeline.py configuration.
    
    Args:
        num_passes: Number of complete rotations to perform (default: 3)
        resolution: Number of measurement points around 360° per pass (default: 50)
        seconds_per_rotation: Time for one complete turntable rotation in seconds (default: 120)
        device_index: Audio device index for 4-channel array (None = use default input)
        sample_duration: Duration to record audio at each point in seconds (default: 0.8)
        rotation_direction: Direction of turntable rotation ('clockwise' or 'counterclockwise')
        output_dir: Directory to save measurement data
        reference_angle: Angle where array is pointing initially (default: 0°)
        use_pipeline: Whether to apply full processing pipeline with beamformer (default: True)
        wait_between_passes: If True, wait for Enter key before starting next pass
        quarter_rotation: If True, measure only front 90° instead of full 360° (faster)
                         Useful for measuring front directivity and mirroring afterwards
        edge_padding_points: Number of extra measurements before 0° and after limit (e.g., 2).
                    These edge captures stabilize boundary bins; final averaged output keeps
                    only the in-range window [0..total_degrees].
        alternate_rotation_direction: If True, alternate direction each pass.
                         If False, keep the same rotation_direction for all passes.
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
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate timing
    total_degrees = 90 if quarter_rotation else 360
    degrees_per_measurement = total_degrees / resolution
    interval = seconds_per_rotation * (degrees_per_measurement / 360)  # Time per measurement point
    pass_duration = seconds_per_rotation * (total_degrees / 360.0)
    edge_padding_points = int(max(0, edge_padding_points))
    padded_resolution = resolution + (2 * edge_padding_points)
    speed_deg_per_sec = 360.0 / max(seconds_per_rotation, 1e-9)
    pass_duration_padded = pass_duration + (2.0 * edge_padding_points * interval)
    
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
    
    print(f"\n{'='*70}")
    print(f"Polar Pattern Test Protocol Configuration:")
    print(f"{'='*70}")
    print(f"  Audio device: {device_name} (index {device_index})")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {num_channels} (array-based beamforming)")
    print(f"  Number of passes: {num_passes}")
    print(f"  Measurement range: {total_degrees}° {f'(FRONT ONLY - will be mirrored)' if quarter_rotation else '(FULL CIRCLE)'}")
    print(f"  Resolution: {resolution} points ({degrees_per_measurement:.1f}° per measurement)")
    print(f"  Edge padding points: {edge_padding_points} (effective captures/pass: {padded_resolution})")
    print(f"  Turntable speed: {seconds_per_rotation} seconds/full rotation (adjusted for {total_degrees}° range)")
    print(f"  Measurement interval: {interval:.2f} seconds")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  Rotation direction: {rotation_direction}")
    print(f"  Alternate direction each pass: {'ON' if alternate_rotation_direction else 'OFF'}")
    print(f"  Microphone reference angle: {reference_angle}°")
    print(f"  Total test duration: ~{num_passes * resolution * interval / 60:.1f} minutes")
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
    print("  Rollback current pass key: Backspace (active only while a pass is running)")

    if sample_duration > interval:
        print(
            f"  WARNING: sample_duration ({sample_duration:.3f}s) is greater than interval ({interval:.3f}s)."
            " Requested rotation cadence cannot be reached with blocking captures;"
            " effective spacing will be >= sample_duration + processing time."
        )
    theoretical_max_points = int(max(1, np.floor(pass_duration_padded / max(sample_duration, 1e-6))))
    if theoretical_max_points < padded_resolution:
        print(
            f"  WARNING: Requested captures ({padded_resolution}) are higher than time budget allows "
            f"for this pass duration/sample duration (theoretical max ~{theoretical_max_points} before processing overhead)."
        )
    if edge_padding_points > 0:
        pre_rel = [-(k * degrees_per_measurement) for k in range(edge_padding_points, 0, -1)]
        post_rel = [total_degrees + (k * degrees_per_measurement) for k in range(1, edge_padding_points + 1)]
        pre_abs = [((reference_angle + a) % 360.0) for a in pre_rel]
        post_abs = [((reference_angle + a) % 360.0) for a in post_rel]
        print(f"  Edge pre-window angles (relative): {[round(v, 2) for v in pre_rel]}")
        print(f"  Edge pre-window angles (absolute): {[round(v, 2) for v in pre_abs]}")
        print(f"  Edge post-window angles (relative): {[round(v, 2) for v in post_rel]}")
        print(f"  Edge post-window angles (absolute): {[round(v, 2) for v in post_abs]}")
    print(f"{'='*70}\n")
    
    # Setup logging
    logger = logging.getLogger("PolarPatternTest")
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
        geometry_path = Path(__file__).resolve().parent.parent / "array_geometries" / "1_square.xml"
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
        filters = []
        agc = None
    
    # Storage for all measurements across all passes
    all_measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Warm up filters/AGC before timed measurements to avoid low initial gain frames.
    # This also gives a deterministic moment for the operator to start rotation.
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
            _ = apply_realtime_processing_chain(
                block=warmup_audio,
                beamformer=beamformer,
                filters=filters,
                agc=agc,
                sample_rate=sample_rate,
                monitor_gain=1.0,
                theta_deg=0.0,
                freeze_beamformer=bool(freeze_beamformer),
                freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
            )
    except Exception as warmup_err:
        print(f"Warm-up warning: {type(warmup_err).__name__}: {warmup_err}")

    input("Warm-up complete. Press Enter to begin measurement and start rotation...")
    
    print(f"Starting {num_passes}-pass polar pattern measurements...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for pass_num in range(num_passes):
            # Optional alternating direction across passes.
            if alternate_rotation_direction:
                pass_rotation_direction = rotation_direction if pass_num % 2 == 0 else (
                    'clockwise' if rotation_direction == 'counterclockwise' else 'counterclockwise'
                )
            else:
                pass_rotation_direction = rotation_direction
            
            print(f"\n{'='*70}")
            print(f"PASS {pass_num + 1}/{num_passes} - Direction: {pass_rotation_direction.upper()}")
            print(f"{'='*70}\n")
            
            # Wait for user input before starting this pass (except for the first pass)
            if wait_between_passes and pass_num > 0:
                input(f"Press Enter to start pass {pass_num + 1}/{num_passes}...")
                print()

            while True:
                with _pass_keyboard_mode():
                    # Time-based synchronization: angle is derived from real elapsed time,
                    # with edge-padding captures before/after the in-range window.
                    pass_measurement_start_idx = len(all_measurements)
                    pass_start_time = time.time()
                    pass_end_time = pass_start_time + pass_duration_padded
                    next_measurement_time = pass_start_time
                    meas_idx = 0
                    rollback_requested = False

                    while meas_idx < padded_resolution:
                        if _consume_backspace_request():
                            rollback_requested = True
                            break

                        now = time.time()
                        if now >= pass_end_time:
                            break

                        # Keep capture starts roughly paced by requested interval when possible.
                        time_to_wait = next_measurement_time - now
                        while time_to_wait > 0:
                            if _consume_backspace_request():
                                rollback_requested = True
                                break
                            sleep_chunk = min(0.05, time_to_wait)
                            time.sleep(sleep_chunk)
                            time_to_wait = next_measurement_time - time.time()
                        if rollback_requested:
                            break

                        # Preview angle (for logging/errors) based on current elapsed time.
                        now_for_angle = time.time()
                        elapsed_preview = np.clip(now_for_angle - pass_start_time, 0.0, pass_duration_padded)
                        logical_ccw_preview = (-edge_padding_points * degrees_per_measurement) + (elapsed_preview * speed_deg_per_sec)
                        if pass_rotation_direction == 'counterclockwise':
                            relative_angle_preview = logical_ccw_preview
                        else:
                            relative_angle_preview = total_degrees - logical_ccw_preview
                        expected_angle_preview = (reference_angle + relative_angle_preview) % 360

                        # Record audio sample (4-channel for beamforming)
                        try:
                            audio_data = sd.rec(
                                int(sample_duration * sample_rate),
                                samplerate=sample_rate,
                                channels=num_channels,
                                device=device_index,
                                blocking=True
                            )
                        except Exception as e:
                            logger.error(f"Error recording audio at angle {expected_angle_preview:.1f}°: {e}")
                            continue

                        if _consume_backspace_request():
                            rollback_requested = True
                            break

                        # Ensure audio is float32 and normalized to [-1, 1]
                        audio_data = np.asarray(audio_data).astype(np.float32)
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 1.0:
                            audio_data = audio_data / max_val

                        # Apply processing pipeline using the shared Array_RealTime chain helper
                        if use_pipeline:
                            processed_audio = apply_realtime_processing_chain(
                                block=audio_data,
                                beamformer=beamformer,
                                filters=filters,
                                agc=agc,
                                sample_rate=sample_rate,
                                monitor_gain=1.0,
                                theta_deg=0.0,
                                freeze_beamformer=bool(freeze_beamformer),
                                freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
                            )
                        else:
                            processed_audio = np.asarray(audio_data[:, 0], dtype=np.float32)

                        # Calculate RMS and peak levels
                        rms_level = np.sqrt(np.mean(processed_audio ** 2))
                        peak_level = np.max(np.abs(processed_audio))

                        # Compute expected angle from elapsed real time at the CENTER of the capture window.
                        # This keeps angle labels synchronized with turntable motion even under load.
                        capture_mid_time = time.time() - (sample_duration * 0.5)
                        elapsed = np.clip(capture_mid_time - pass_start_time, 0.0, pass_duration_padded)
                        logical_ccw = (-edge_padding_points * degrees_per_measurement) + (elapsed * speed_deg_per_sec)
                        if pass_rotation_direction == 'counterclockwise':
                            relative_angle = logical_ccw
                        else:
                            relative_angle = total_degrees - logical_ccw
                        expected_angle = (reference_angle + relative_angle) % 360

                        # Store measurement
                        measurement = {
                            'measurement_index': len(all_measurements),
                            'pass_number': pass_num + 1,
                            'expected_angle': expected_angle,
                            'relative_angle': relative_angle,
                            'timestamp': datetime.now().isoformat(),
                            'reference_angle': reference_angle,
                            'rms_level': rms_level,
                            'peak_level': peak_level,
                        }

                        all_measurements.append(measurement)

                        # Progress indicator for every measurement so the cadence matches the resolution.
                        print(f"  Pass {pass_num + 1}, Measurement {meas_idx + 1}/{padded_resolution}: "
                            f"Angle: {expected_angle:6.1f}° (rel {relative_angle:6.1f}°) | RMS: {20*np.log10(max(rms_level, 1e-10)):7.2f} dB | "
                                f"Peak: {20*np.log10(max(peak_level, 1e-10)):7.2f} dB")

                        # Schedule next measurement from absolute timeline, not loop execution time.
                        next_measurement_time += interval
                        meas_idx += 1

                    if rollback_requested:
                        removed = len(all_measurements) - pass_measurement_start_idx
                        if removed > 0:
                            del all_measurements[pass_measurement_start_idx:]
                        print(
                            f"  Backspace detected: rolled back pass {pass_num + 1} "
                            f"(removed {removed} measurements)."
                        )
                    else:
                        if meas_idx < padded_resolution:
                            print(
                                f"  Pass {pass_num + 1}: captured {meas_idx}/{padded_resolution} samples before padded rotation window ended "
                                f"({pass_duration_padded:.2f}s)."
                            )

                if rollback_requested:
                    input(f"Press Enter to restart pass {pass_num + 1}/{num_passes}...")
                    print()
                    continue

                break
        
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
    
    # Calculate dB values
    min_level = 1e-10
    df_all['rms_dbfs'] = 20 * np.log10(np.maximum(df_all['rms_level'], min_level))
    df_all['peak_dbfs'] = 20 * np.log10(np.maximum(df_all['peak_level'], min_level))
    
    # Save raw data with all passes
    raw_csv_file = output_path / f"polar_pattern_raw_{test_timestamp}.csv"
    df_all.to_csv(raw_csv_file, index=False)
    print(f"\n{'='*70}")
    print(f"Raw data saved to: {raw_csv_file}")
    
    # Calculate averaged results by angle
    print("Averaging measurements across passes...")

    # Keep in-range window plus half-bin overscan so slight post-limit samples
    # can contribute to the endpoint bin (e.g., 90°) and avoid abrupt cutoff.
    lower_bound = -0.5 * degrees_per_measurement
    upper_bound = total_degrees + (0.5 * degrees_per_measurement)
    df_avg = df_all[(df_all['relative_angle'] >= lower_bound) & (df_all['relative_angle'] <= upper_bound)].copy()
    if df_avg.empty:
        print("No in-range samples collected after filtering edge-padding points.")
        return None

    # relative_angle is time-derived float, so values differ slightly across passes.
    # Bin to intended grid before averaging to avoid sawtooth high/low alternation.
    # Use an extra endpoint bin so the upper-limit angle (e.g., 90°) is explicitly represented.
    rel_angle = df_avg['relative_angle']
    bin_idx = np.floor((rel_angle + (degrees_per_measurement * 0.5)) / degrees_per_measurement).astype(int)
    bin_idx = np.clip(bin_idx, 0, resolution)
    df_avg['angle_bin'] = bin_idx
    df_avg['expected_angle_binned'] = (reference_angle + df_avg['angle_bin'] * degrees_per_measurement) % 360.0
    df_avg['relative_angle_binned'] = df_avg['angle_bin'] * degrees_per_measurement
    
    # Group by binned angles and calculate mean for numeric columns
    grouped = df_avg.groupby(['angle_bin', 'expected_angle_binned', 'relative_angle_binned'], as_index=False).agg({
        'rms_level': 'mean',
        'rms_dbfs': 'mean',
        'peak_level': 'mean',
        'peak_dbfs': 'mean',
        'reference_angle': 'first',
        'pass_number': 'count'
    })

    grouped.rename(columns={
        'expected_angle_binned': 'expected_angle',
        'relative_angle_binned': 'relative_angle',
    }, inplace=True)
    
    # Rename count column
    grouped.rename(columns={'pass_number': 'num_passes'}, inplace=True)
    
    # Sort by bin index to preserve monotonic angle order
    grouped.sort_values('angle_bin', inplace=True)
    grouped.drop(columns=['angle_bin'], inplace=True)

    # DI uses power, not dB. Compute the average noise power in the linear domain
    # from the uniformly spaced angle bins so each direction contributes equally.
    noise_avg_power_linear = float(np.mean(np.square(grouped['rms_level'].to_numpy(dtype=float))))
    noise_avg_rms_linear = float(np.sqrt(noise_avg_power_linear))
    noise_avg_rms_dbfs = float(20.0 * np.log10(max(noise_avg_rms_linear, min_level)))

    # Store the summary value on every row so the CSV remains backward-compatible
    # while still carrying the total noise power needed for DI calculations.
    grouped['noise_avg_power_linear'] = noise_avg_power_linear
    grouped['noise_avg_rms_linear'] = noise_avg_rms_linear
    grouped['noise_avg_rms_dbfs'] = noise_avg_rms_dbfs
    
    # Save averaged results
    averaged_csv_file = output_path / f"polar_pattern_averaged_{test_timestamp}.csv"
    grouped.to_csv(averaged_csv_file, index=False)
    
    # Calculate statistics
    rms_min = grouped['rms_dbfs'].min()
    rms_max = grouped['rms_dbfs'].max()
    rms_range = rms_max - rms_min
    rms_min_angle = grouped.loc[grouped['rms_dbfs'].idxmin(), 'expected_angle']
    rms_max_angle = grouped.loc[grouped['rms_dbfs'].idxmax(), 'expected_angle']
    
    peak_min = grouped['peak_dbfs'].min()
    peak_max = grouped['peak_dbfs'].max()
    peak_range = peak_max - peak_min
    peak_max_angle = grouped.loc[grouped['peak_dbfs'].idxmax(), 'expected_angle']
    
    print(f"\n{'='*70}")
    print(f"Multi-pass measurement complete!")
    print(f"{'='*70}")
    print(f"  Total measurements: {len(df_all)} ({len(grouped)} unique angles × {num_passes} passes)")
    print(f"\n  RMS Level:")
    print(f"    Range: {rms_range:.1f} dB ({rms_min:.1f} to {rms_max:.1f} dBFS)")
    print(f"    Maximum at: {rms_max_angle:.1f}°")
    print(f"    Minimum at: {rms_min_angle:.1f}°")
    print(f"\n  Peak Level:")
    print(f"    Range: {peak_range:.1f} dB ({peak_min:.1f} to {peak_max:.1f} dBFS)")
    print(f"    Maximum at: {peak_max_angle:.1f}°")
    print(f"\n  Noise Power Summary (for DI):")
    print(f"    Average noise power (linear): {noise_avg_power_linear:.8f}")
    print(f"    Average noise RMS: {noise_avg_rms_linear:.6f}")
    print(f"    Average noise RMS: {noise_avg_rms_dbfs:.2f} dBFS")
    print(f"\n  Data saved to:")
    print(f"    Raw data (all passes):  {raw_csv_file}")
    print(f"    Averaged results:       {averaged_csv_file}")
    print(f"{'='*70}\n")
    
    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Polar Pattern Test Protocol - Measure microphone directivity with full audio processing')
    
    parser.add_argument('--passes', type=int, default=3,
                        help='Number of complete rotations (default: 3)')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Number of measurement points per rotation (default: 50)')
    parser.add_argument('--rotation-time', type=float, default=120.0,
                        help='Time for one rotation in seconds (default: 120.0)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: system default input device)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--duration', type=float, default=0.8,
                        help='Audio sample duration in seconds (default: 0.8)')
    parser.add_argument('--rotation-direction', type=str,
                        choices=['clockwise', 'counterclockwise'],
                        default='counterclockwise',
                        help='Direction of turntable rotation (default: counterclockwise)')
    parser.add_argument('--output', type=str, default='data/test_protocol/polar_pattern',
                        help='Output directory (default: data/test_protocol/polar_pattern)')
    parser.add_argument('--reference-angle', type=int, default=0,
                        help='Initial microphone pointing direction (default: 0°)')
    parser.add_argument('--no-pipeline', action='store_true',
                        help='Disable processing pipeline, record raw audio only')
    parser.add_argument('--wait-between-passes', action='store_true',
                        help='Wait for Enter key before starting each new pass')
    parser.add_argument('--quarter-rotation', action='store_true',
                        help='Measure only front 90° instead of full 360° (useful for measuring front directivity and mirroring afterwards)')
    parser.add_argument('--edge-padding-points', type=int, default=2,
                        help='Extra measurements before 0° and after the rotation limit (default: 2)')
    parser.add_argument('--alternate-rotation-direction', action=argparse.BooleanOptionalAction, default=False,
                        help='Alternate rotation direction at each pass (default: disabled)')
    parser.add_argument('--freeze-beamformer', action=argparse.BooleanOptionalAction, default=True,
                        help='Freeze beamformer steering during the full measurement run (default: enabled)')
    parser.add_argument('--freeze-angle', type=float, default=0.0,
                        help='Steering angle used when beamformer freeze is enabled (default: 0.0)')
    parser.add_argument('--enable-agc', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable both AGC stages (AdaptiveAmplifier + PedalboardAGC) (default: disabled)')
    parser.add_argument('--enable-spectral-filter', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable spectral subtraction filter (default: enabled)')
    parser.add_argument('--save-on-interrupt', action=argparse.BooleanOptionalAction, default=False,
                        help='Save partial data when interrupted by Ctrl+C (default: disabled)')
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print("=" * 70)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            try:
                is_input = dev['max_input_channels'] > 0
                channels = dev['max_input_channels']
                samplerate = int(dev['default_samplerate'])
                if is_input:
                    default_marker = ' [DEFAULT INPUT]' if i == sd.default.device[0] else ''
                    print(f"  {i:2d}: {dev['name']:<40} ({channels} channels, {samplerate} Hz){default_marker}")
            except (IndexError, TypeError):
                pass
        print("=" * 70)
        exit(0)
    
    # Run test
    test_polar_pattern(
        num_passes=args.passes,
        resolution=args.resolution,
        seconds_per_rotation=args.rotation_time,
        device_index=args.device,
        sample_duration=args.duration,
        rotation_direction=args.rotation_direction,
        output_dir=args.output,
        reference_angle=args.reference_angle,
        use_pipeline=not args.no_pipeline,
        wait_between_passes=args.wait_between_passes,
        quarter_rotation=args.quarter_rotation,
        edge_padding_points=args.edge_padding_points,
        alternate_rotation_direction=args.alternate_rotation_direction,
        freeze_beamformer=args.freeze_beamformer,
        freeze_angle_deg=args.freeze_angle,
        enable_agc=args.enable_agc,
        enable_spectral_filter=args.enable_spectral_filter,
        save_on_interrupt=args.save_on_interrupt,
    )
    
    # Usage examples:
    # 
    # List available audio devices:
    # python prototype/test_protocol/1_Polar_Pattern.py --list-devices
    #
    # Standard measurement with full pipeline (3 passes, 50 points, 120 sec/rotation):
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --duration 0.8
    #
    # Quick test (1 pass, 36 points, 60 sec/rotation):
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --passes 1 --resolution 36 --rotation-time 60 --duration 0.8
    #
    # Front directivity only (90°, mirrored afterwards) - 3 passes, 50 points:
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --duration 0.8 --quarter-rotation
    #
    # Raw audio only (no processing pipeline):
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --no-pipeline --passes 3 --resolution 50 --rotation-time 120
    #
    # With manual synchronization between passes:
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --wait-between-passes
    #
    # With manual synchronization and front directivity only:
    # python prototype/test_protocol/1_Polar_Pattern.py --device 1 --passes 3 --resolution 30 --rotation-time 22.5 --wait-between-passes --quarter-rotation