from custom_tuning import Tuning
import usb.core
import numpy as np
import pandas as pd
import math
import sounddevice
import time
from datetime import datetime
from pathlib import Path


def test_directivity_multipass(
    num_passes=3,  # Number of complete rotations to perform
    resolution=36,  # Number of measurement points per rotation (36 = 10° steps)
    seconds_per_rotation=90,  # Time for one complete rotation in seconds
    device_index=1,  # Audio device index
    sample_duration=2.0,  # Audio sample duration in seconds
    output_dir='data/directivity',
    doa_lock_angle=180  # Angle the DOA is locked at
):
    """
    Perform multi-pass directivity measurements of the ReSpeaker mic array.
    Each pass alternates direction (counterclockwise, then clockwise, etc.)
    and results are averaged across all passes to reduce noise.
    
    Args:
        num_passes: Number of complete rotations to perform (default: 3)
        resolution: Number of measurement points around 360° per pass
        seconds_per_rotation: Time for one complete turntable rotation (seconds)
        device_index: Audio device index for ReSpeaker
        sample_duration: Duration to record audio at each point (seconds)
        output_dir: Directory to save measurement data
        doa_lock_angle: Angle to lock the DOA at (default: 180°)
    
    Returns:
        DataFrame with averaged measurements
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate timing
    degrees_per_measurement = 360 / resolution
    interval = seconds_per_rotation * (degrees_per_measurement / 360)
    
    print(f"Multi-Pass Directivity Test Configuration:")
    print(f"  Number of passes: {num_passes}")
    print(f"  Resolution: {resolution} points ({degrees_per_measurement:.1f}° per measurement)")
    print(f"  Turntable speed: {seconds_per_rotation} seconds/rotation")
    print(f"  Measurement interval: {interval:.2f} seconds")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  DOA locked at: {doa_lock_angle}°")
    print(f"  Total test duration: ~{num_passes * resolution * interval / 60:.1f} minutes")
    print(f"  Output directory: {output_path}")
    print("-" * 70)
    
    # Find the Respeaker 4 mic array USB device
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return None
    
    # Audio parameters
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1
    
    # Storage for all measurements across all passes
    all_measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("\nStarting multi-pass measurements...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for pass_num in range(num_passes):
            # Alternate direction: odd passes = counterclockwise, even passes = clockwise
            rotation_direction = 'counterclockwise' if pass_num % 2 == 0 else 'clockwise'
            
            print(f"\n{'='*70}")
            print(f"PASS {pass_num + 1}/{num_passes} - Direction: {rotation_direction.upper()}")
            print(f"{'='*70}\n")
            
            for meas_idx in range(resolution):
                measurement_start = time.time()
                
                # Expected angle based on position
                if rotation_direction == 'counterclockwise':
                    expected_angle = (meas_idx * degrees_per_measurement) % 360
                else:
                    # Clockwise: reverse the angle progression
                    expected_angle = (360 - (meas_idx * degrees_per_measurement)) % 360
                
                # Read tuning parameters
                tuning = Tuning(mic_array)
                agc_gain = tuning.read('AGCGAIN')
                agc_gain_db = 20 * math.log10(agc_gain) if agc_gain and agc_gain > 0 else -float('inf')
                doa_angle = tuning.direction
                voice_activity = tuning.is_voice
                tuning.close()
                
                # Small delay to release USB control
                time.sleep(0.05)
                
                # Record audio sample
                audio_data = sounddevice.rec(
                    int(sample_duration * RESPEAKER_RATE),
                    samplerate=RESPEAKER_RATE,
                    channels=RESPEAKER_CHANNELS,
                    dtype='int16',
                    device=device_index
                )
                sounddevice.wait()
                
                # Process audio - calculate overall RMS for continuous signal
                audio_data = audio_data.flatten()
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Convert to dBFS with noise floor protection
                full_scale = 32768.0
                min_level = 1e-10
                
                rms = max(rms, min_level)
                
                rms_dbfs = 20 * math.log10(rms / full_scale)
                
                # Calculate relative angle (source angle relative to locked beamformer direction)
                relative_angle = (expected_angle - doa_lock_angle) % 360
                
                # Store measurement with pass information
                measurement = {
                    'pass_number': pass_num + 1,
                    'measurement_index': meas_idx,
                    'rotation_direction': rotation_direction,
                    'expected_angle': expected_angle,
                    'relative_angle': relative_angle,
                    'timestamp': datetime.now().isoformat(),
                    'doa_angle': doa_angle,
                    'locked_doa': doa_lock_angle,
                    'agc_gain': agc_gain,
                    'agc_gain_db': agc_gain_db,
                    'voice_activity': voice_activity,
                    'rms_level': rms,
                    'rms_dbfs': rms_dbfs,
                }
                all_measurements.append(measurement)
                
                # Progress report
                print(f"[Pass {pass_num+1}/{num_passes}][{meas_idx+1}/{resolution}] "
                      f"Angle: {expected_angle:6.1f}° | "
                      f"Relative: {relative_angle:6.1f}° | "
                      f"DOA: {doa_angle:3d}° (locked) | "
                      f"RMS: {rms_dbfs:6.1f} dBFS | "
                
                # Wait for next measurement (accounting for processing time)
                elapsed = time.time() - measurement_start
                remaining = interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
        print("No data saved.")
        return None
    
    # Convert all measurements to DataFrame
    df_all = pd.DataFrame(all_measurements)
    
    # Save raw data with all passes
    raw_csv_file = output_path / f"directivity_multipass_raw_{test_timestamp}.csv"
    df_all.to_csv(raw_csv_file, index=False)
    print(f"\n{'='*70}")
    print(f"Raw data saved to: {raw_csv_file}")
    
    # Calculate averaged results by angle
    print("\nAveraging measurements across passes...")
    
    # Group by expected_angle and calculate mean for numeric columns
    numeric_columns = ['rms_level', 'rms_dbfs', 'agc_gain', 'agc_gain_db']
    
    # Group by both expected_angle and relative_angle to preserve both
    grouped = df_all.groupby(['expected_angle', 'relative_angle'], as_index=False).agg({
        'rms_level': 'mean',
        'rms_dbfs': 'mean',
        'agc_gain': 'mean',
        'agc_gain_db': 'mean',
        'doa_angle': 'mean',
        'voice_activity': 'mean',
        'locked_doa': 'first',  # Same for all measurements
        'pass_number': 'count'  # Count how many passes contributed
    })
    
    # Rename count column
    grouped.rename(columns={'pass_number': 'num_passes'}, inplace=True)
    
    # Sort by expected_angle
    grouped.sort_values('expected_angle', inplace=True)
    
    # Save averaged results
    averaged_csv_file = output_path / f"directivity_multipass_averaged_{test_timestamp}.csv"
    grouped.to_csv(averaged_csv_file, index=False)
    
    # Calculate statistics
    rms_min = grouped['rms_dbfs'].min()
    rms_max = grouped['rms_dbfs'].max()
    rms_range = rms_max - rms_min
    rms_min_angle = grouped.loc[grouped['rms_dbfs'].idxmin(), 'expected_angle']
    rms_max_angle = grouped.loc[grouped['rms_dbfs'].idxmax(), 'expected_angle']
    
    print(f"\n{'='*70}")
    print(f"Multi-pass measurement complete!")
    print(f"  Total measurements: {len(df_all)} ({len(grouped)} unique angles × {num_passes} passes)")
    print(f"  RMS Level Range: {rms_range:.1f} dB ({rms_min:.1f} to {rms_max:.1f} dBFS)")
    print(f"    Maximum at: {rms_max_angle:.1f}°")
    print(f"    Minimum at: {rms_min_angle:.1f}°")
    print(f"\n  Data saved to:")
    print(f"    Raw data (all passes):  {raw_csv_file}")
    print(f"    Averaged results:       {averaged_csv_file}")
    print(f"{'='*70}")
    
    return grouped


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-pass directivity test for ReSpeaker microphone array')
    parser.add_argument('--passes', type=int, default=3, 
                        help='Number of complete rotations (default: 3)')
    parser.add_argument('--resolution', type=int, default=50, 
                        help='Number of measurement points per rotation (default: 50)')
    parser.add_argument('--rotation-time', type=float, default=120.0,
                        help='Time for one rotation in seconds (default: 120.0)')
    parser.add_argument('--device', type=int, default=1,
                        help='Audio device index (default: 1)')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Audio sample duration in seconds (default: 2.0)')
    parser.add_argument('--output', type=str, default='data/directivity',
                        help='Output directory (default: data/directivity)')
    parser.add_argument('--doa-lock-angle', type=int, default=180,
                        help='Angle to lock the DOA at (default: 180°)')
   
    args = parser.parse_args()
    
    # Recommended settings:
    # 3 passes with 50 points each:
    # sudo .venv/bin/python3 respeaker/test_directivity_multipass.py --passes 3 --resolution 50 --rotation-time 120 --duration 2 --doa-lock-angle 180
    
    test_directivity_multipass(
        num_passes=args.passes,
        resolution=args.resolution,
        seconds_per_rotation=args.rotation_time,
        device_index=args.device,
        sample_duration=args.duration,
        output_dir=args.output,
        doa_lock_angle=args.doa_lock_angle
    )
