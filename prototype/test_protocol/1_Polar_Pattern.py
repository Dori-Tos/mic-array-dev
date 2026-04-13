"""
Polar Pattern Test Protocol
===========================

Measures the polar pattern (directivity) of a microphone using the full audio processing pipeline.
Performs multi-pass measurements at different angles, applies the complete beamforming and filtering chain,
and saves results in standardized CSV format for analysis and comparison.

Combines:
- Pipeline: Same processing chain as test_pipeline.py (beamformer, filters, AGC)
- Methodology: Dynamic microphone measurement approach (rotating microphone, recording at each angle)
- Approach: Multi-pass measurements with averaging (like test_directivity_multipass.py)
- Output: Standardized CSV format with RMS, peak levels, angles, timestamps
"""

from datetime import datetime
from pathlib import Path
import argparse
import time
import math
import logging
import sys

import numpy as np
import pandas as pd
import sounddevice as sd

# Import processing classes from the shared codebase
sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.Microphone import Microphone
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, PedalboardAGC, AGCChain


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
    quarter_rotation=False
):
    """
    Measure microphone polar pattern using the complete processing pipeline.
    
    Performs multi-pass directivity measurements with full audio processing:
    - Beamforming (optional MVDR for array measurements)
    - Filtering (BandPass + Spectral Subtraction)
    - AGC chain (Adaptive + Pedalboard)
    
    Args:
        num_passes: Number of complete rotations to perform (default: 3)
        resolution: Number of measurement points around 360° per pass (default: 50)
        seconds_per_rotation: Time for one complete turntable rotation in seconds (default: 120)
        device_index: Audio device index (None = use default input device)
        sample_duration: Duration to record audio at each point in seconds (default: 0.8)
        rotation_direction: Direction of turntable rotation ('clockwise' or 'counterclockwise')
        output_dir: Directory to save measurement data
        reference_angle: Angle where microphone is pointing initially (default: 0°)
        use_pipeline: Whether to apply full processing pipeline (default: True)
        wait_between_passes: If True, wait for Enter key before starting next pass
        quarter_rotation: If True, measure only front 90° instead of full 360° (default: False)
                         Useful for measuring front directivity and mirroring afterwards
    
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
    
    # Get device info
    if device_index is None:
        device_info = sd.query_devices(kind='input')
        device_name = device_info['name']
        device_index = device_info['index']
    else:
        device_info = sd.query_devices(device_index)
        device_name = device_info['name']
    
    sample_rate = int(device_info['default_samplerate'])
    
    print(f"\n{'='*70}")
    print(f"Polar Pattern Test Protocol Configuration:")
    print(f"{'='*70}")
    print(f"  Audio device: {device_name} (index {device_index})")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Number of passes: {num_passes}")
    print(f"  Measurement range: {total_degrees}° {f'(FRONT ONLY - will be mirrored)' if quarter_rotation else '(FULL CIRCLE)'}")
    print(f"  Resolution: {resolution} points ({degrees_per_measurement:.1f}° per measurement)")
    print(f"  Turntable speed: {seconds_per_rotation} seconds/full rotation (adjusted for {total_degrees}° range)")
    print(f"  Measurement interval: {interval:.2f} seconds")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  Rotation direction: {rotation_direction}")
    print(f"  Microphone reference angle: {reference_angle}°")
    print(f"  Total test duration: ~{num_passes * resolution * interval / 60:.1f} minutes")
    print(f"  Output directory: {output_path}")
    if use_pipeline:
        print(f"  Processing: Full pipeline (BandPass + SpectralSubtraction + AGC)")
    else:
        print(f"  Processing: Raw audio only")
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
        
        # Filters
        filters = [
            BandPassFilter(
                logger=logger,
                sample_rate=sample_rate,
                low_cutoff=300.0,
                high_cutoff=4000.0,
                order=4
            ),
            SpectralSubtractionFilter(
                logger=logger,
                sample_rate=sample_rate,
                noise_factor=0.65,
                gain_floor=0.35,
                noise_alpha=0.995,
                noise_update_snr_db=8.0,
                gain_smooth_alpha=0.92,
            ),
        ]
        
        # AGC chain
        agc = AGCChain(logger=logger, stages=[
            AdaptiveAmplifier(
                logger=logger,
                target_rms=0.08,
                min_gain=1.0,
                max_gain=12.0,
                adapt_alpha=0.04,
                speech_activity_rms=0.00012,
                silence_decay_alpha=0.008,
                activity_hold_ms=600.0,
                peak_protect_threshold=0.35,
                peak_protect_strength=0.85,
                max_gain_warn_rms_min=0.001,
            ),
            PedalboardAGC(
                logger=logger,
                sample_rate=sample_rate,
                threshold_db=-20.0,
                ratio=3.5,
                attack_ms=3.0,
                release_ms=140.0,
                limiter_threshold_db=-1.4,
                limiter_release_ms=100.0
            ),
        ])
        
        print("Pipeline initialized.\n")
    else:
        filters = []
        agc = None
    
    # Storage for all measurements across all passes
    all_measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print(f"Starting {num_passes}-pass polar pattern measurements...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for pass_num in range(num_passes):
            # Alternate direction: odd passes = counterclockwise, even passes = clockwise
            pass_rotation_direction = rotation_direction if pass_num % 2 == 0 else (
                'clockwise' if rotation_direction == 'counterclockwise' else 'counterclockwise'
            )
            
            print(f"\n{'='*70}")
            print(f"PASS {pass_num + 1}/{num_passes} - Direction: {pass_rotation_direction.upper()}")
            print(f"{'='*70}\n")
            
            # Wait for user input before starting this pass (except for the first pass)
            if wait_between_passes and pass_num > 0:
                input(f"Press Enter to start pass {pass_num + 1}/{num_passes}...")
                print()
            
            for meas_idx in range(resolution):
                measurement_start = time.time()
                
                # Calculate expected angle based on rotation direction and measurement range
                if pass_rotation_direction == 'counterclockwise':
                    angle_offset = (meas_idx * degrees_per_measurement)
                else:
                    angle_offset = total_degrees - (meas_idx * degrees_per_measurement)
                
                expected_angle = (reference_angle + angle_offset) % 360
                
                # Wait for the measurement interval
                time_to_wait = interval - (time.time() - measurement_start)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                
                # Record audio sample
                try:
                    audio_data = sd.rec(
                        int(sample_duration * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        device=device_index,
                        blocking=True
                    )
                except Exception as e:
                    logger.error(f"Error recording audio at angle {expected_angle:.1f}°: {e}")
                    continue
                
                # Ensure audio is float32 and normalized to [-1, 1]
                audio_data = np.squeeze(audio_data).astype(np.float32)
                if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                    audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
                
                # Apply processing pipeline if enabled
                processed_audio = audio_data.copy()
                
                if use_pipeline and filters:
                    # Apply filters
                    for filt in filters:
                        processed_audio = filt.apply(processed_audio)
                
                if use_pipeline and agc:
                    # Apply AGC
                    processed_audio = agc.process(processed_audio)
                
                # Calculate RMS and peak levels
                rms_level = np.sqrt(np.mean(processed_audio ** 2))
                peak_level = np.max(np.abs(processed_audio))
                
                # Store measurement
                measurement = {
                    'measurement_index': len(all_measurements),
                    'pass_number': pass_num + 1,
                    'expected_angle': expected_angle,
                    'relative_angle': expected_angle,  # Same as expected_angle for single mic
                    'timestamp': datetime.now().isoformat(),
                    'reference_angle': reference_angle,
                    'rms_level': rms_level,
                    'peak_level': peak_level,
                }
                
                all_measurements.append(measurement)
                
                # Progress indicator
                if (meas_idx + 1) % 10 == 0:
                    print(f"  Pass {pass_num + 1}, Measurement {meas_idx + 1}/{resolution}: "
                          f"Angle: {expected_angle:6.1f}° | RMS: {20*np.log10(max(rms_level, 1e-10)):7.2f} dB | "
                          f"Peak: {20*np.log10(max(peak_level, 1e-10)):7.2f} dB")
        
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
        if len(all_measurements) == 0:
            print("No data to save.")
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
    
    # Group by expected_angle and calculate mean for numeric columns
    grouped = df_all.groupby(['expected_angle', 'relative_angle'], as_index=False).agg({
        'rms_level': 'mean',
        'rms_dbfs': 'mean',
        'peak_level': 'mean',
        'peak_dbfs': 'mean',
        'reference_angle': 'first',
        'pass_number': 'count'
    })
    
    # Rename count column
    grouped.rename(columns={'pass_number': 'num_passes'}, inplace=True)
    
    # Sort by expected_angle
    grouped.sort_values('expected_angle', inplace=True)
    
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
    parser.add_argument('--output', type=str, default='data/polar_pattern',
                        help='Output directory (default: data/polar_pattern)')
    parser.add_argument('--reference-angle', type=int, default=0,
                        help='Initial microphone pointing direction (default: 0°)')
    parser.add_argument('--no-pipeline', action='store_true',
                        help='Disable processing pipeline, record raw audio only')
    parser.add_argument('--wait-between-passes', action='store_true',
                        help='Wait for Enter key before starting each new pass')
    parser.add_argument('--quarter-rotation', action='store_true',
                        help='Measure only front 90° instead of full 360° (useful for measuring front directivity and mirroring afterwards)')
    
    args = parser.parse_args()
    
    # List devices if requested
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
        quarter_rotation=args.quarter_rotation
    )
    
    # Usage examples:
    # 
    # List available audio devices:
    # python 1_Polar_Pattern.py --list-devices
    #
    # Standard measurement with full pipeline (3 passes, 50 points, 120 sec/rotation):
    # python 1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --duration 0.8
    #
    # Quick test (1 pass, 36 points, 60 sec/rotation):
    # python 1_Polar_Pattern.py --device 1 --passes 1 --resolution 36 --rotation-time 60 --duration 0.8
    #
    # Front directivity only (90°, mirrored afterwards) - 3 passes, 50 points:
    # python 1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --duration 0.8 --quarter-rotation
    #
    # Raw audio only (no processing pipeline):
    # python 1_Polar_Pattern.py --device 1 --no-pipeline --passes 3 --resolution 50 --rotation-time 120
    #
    # With manual synchronization between passes:
    # python 1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 120 --wait-between-passes
    #
    # With manual synchronization and front directivity only:
    # python 1_Polar_Pattern.py --device 1 --passes 3 --resolution 50 --rotation-time 90 --wait-between-passes --quarter-rotation
