from custom_tuning import Tuning
import usb.core
import numpy as np
import pandas as pd
import math
import sounddevice
import time
from datetime import datetime
from pathlib import Path


def test_directivity(
    resolution=36,  # Number of measurement points (36 = 10° steps)
    seconds_per_rotation=90,  # Time for one complete rotation in seconds
    device_index=1,  # Audio device index
    sample_duration=2.0,  # Audio sample duration in seconds
    rotation_direction='counterclockwise',  # Rotation direction
    output_dir='data/directivity',
    save_peaks=False,
    signal_type='burst',  # Type of test signal ('burst' or 'continuous')
    burst_period=0.4,  # Duration of burst signal in seconds (if signal_type='burst')
    doa_locked=True,  # Whether the beamformer is locked during measurements (default: True)
    doa_lock_angle=0  # Angle the DOA is locked at (default: 0°)
):
    """
    Perform directivity measurements of the ReSpeaker mic array.
    
    Args:
        resolution: Number of measurement points around 360°
        seconds_per_rotation: Time for one complete turntable rotation (seconds)
        device_index: Audio device index for ReSpeaker
        sample_duration: Duration to record audio at each point (needed to calculate RMS/Peak levels)
        rotation_direction: Direction of turntable rotation ('clockwise' or 'counterclockwise')
        output_dir: Directory to save measurement data
        save_peaks: Whether to save peak measurements
        signal_type: Type of test signal ('burst' for short bursts, 'continuous' for steady tone)
        burst_period: Duration of each burst in seconds (only used if signal_type='burst')
        doa_locked: Whether to lock the beamformer during measurements (default: True)
        doa_lock_angle: Angle to lock the DOA at if locking is enabled (default: 0°)
    
    Returns:
        DataFrame with measurements
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate timing
    degrees_per_measurement = 360 / resolution
    interval = seconds_per_rotation * (degrees_per_measurement / 360)
    
    print(f"Directivity Test Configuration:")
    print(f"  Resolution: {resolution} points ({degrees_per_measurement:.1f}° per measurement)")
    print(f"  Turntable speed: {seconds_per_rotation} seconds/rotation")
    print(f"  Rotation direction: {rotation_direction}")
    print(f"  Measurement interval: {interval:.2f} seconds")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  Total test duration: ~{resolution * interval / 60:.1f} minutes")
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
    
    # Storage for measurements
    measurements = []
    
    # Get timestamp for this test run
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    print("\nStarting measurements...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for meas_idx in range(resolution):
            measurement_start = time.time()
            
            # Expected angle based on position (assuming starting at 0°)
            if rotation_direction == 'clockwise':
                expected_angle = (meas_idx * degrees_per_measurement) % 360
            else:
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
            
            # Process audio
            audio_data = audio_data.flatten()
            
            if signal_type == 'burst':
                # For burst signals: Calculate RMS in sliding windows to find peak RMS during active burst
                window_size = int(0.05 * RESPEAKER_RATE)  # 50ms window
                hop_size = max(1, window_size // 4)  # 25% overlap, minimum 1
                
                # Calculate RMS for each window
                window_rms_values = []
                for i in range(0, len(audio_data) - window_size + 1, hop_size):
                    window = audio_data[i:i+window_size]
                    window_rms_values.append(np.sqrt(np.mean(window**2)))
                
                if window_rms_values:
                    # Use the maximum RMS window (captures the burst peak)
                    rms = max(window_rms_values)
                else:
                    # Fallback for very short samples
                    rms = np.sqrt(np.mean(audio_data**2))
                    
            else:
                # For continuous signals: Calculate overall RMS
                rms = np.sqrt(np.mean(audio_data**2))

            # Peak value
            peak = np.max(np.abs(audio_data))
            
            # Convert to dBFS with noise floor protection
            full_scale = 32768.0
            min_level = 1e-10  # Minimum level to avoid log(0)
            
            rms = max(rms, min_level)  # Apply noise floor
            peak = max(peak, min_level)
            
            rms_dbfs = 20 * math.log10(rms / full_scale)
            peak_dbfs = 20 * math.log10(peak / full_scale)
            
            # Calculate relative angle (source angle relative to locked beamformer direction)
            if doa_locked:
                # Relative angle = how far the source is from the beamformer's locked direction
                relative_angle = (expected_angle - doa_lock_angle) % 360
            else:
                relative_angle = None
            
            # Store measurement
            if save_peaks:
                measurement = {
                    'measurement_index': meas_idx,
                    'expected_angle': expected_angle,
                    'relative_angle': relative_angle,
                    'timestamp': datetime.now().isoformat(),
                    'doa_angle': doa_angle,
                    'locked_doa': doa_lock_angle if doa_locked else None,
                    'agc_gain': agc_gain,
                    'agc_gain_db': agc_gain_db,
                    'voice_activity': voice_activity,
                    'rms_level': rms,
                    'rms_dbfs': rms_dbfs,
                    'peaks': True,
                    'peak_level': peak,
                    'peak_dbfs': peak_dbfs
                }
            else:
                measurement = {
                    'measurement_index': meas_idx,
                    'expected_angle': expected_angle,
                    'relative_angle': relative_angle,
                    'timestamp': datetime.now().isoformat(),
                    'doa_angle': doa_angle,
                    'locked_doa': doa_lock_angle if doa_locked else None,
                    'agc_gain': agc_gain,
                    'agc_gain_db': agc_gain_db,
                    'voice_activity': voice_activity,
                    'rms_level': rms,
                    'rms_dbfs': rms_dbfs,
                    'peaks': False
                }
            measurements.append(measurement)
            
            # Progress report
            if doa_locked and relative_angle is not None:
                print(f"[{meas_idx+1}/{resolution}] Angle: {expected_angle:6.1f}° | "
                      f"Relative: {relative_angle:6.1f}° | "
                      f"DOA: {doa_angle:3d}° (locked) | "
                      f"RMS: {rms_dbfs:6.1f} dBFS | "
                      f"Peak: {peak_dbfs:6.1f} dBFS")
            else:
                print(f"[{meas_idx+1}/{resolution}] Angle: {expected_angle:6.1f}° | "
                      f"DOA: {doa_angle:3d}° | "
                      f"RMS: {rms_dbfs:6.1f} dBFS | "
                      f"Peak: {peak_dbfs:6.1f} dBFS")
            
            # Wait for next measurement (accounting for processing time)
            elapsed = time.time() - measurement_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
    
    # Convert to Pandas DataFrame for optimized storage
    df = pd.DataFrame(measurements)
    
    csv_file = output_path / f"directivity_{test_timestamp}.csv"
    
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Measurement complete!")
    print(f"  Total measurements: {len(measurements)}")
    print(f"  Data saved to:")
    print(f"    CSV:    {csv_file}")
    print(f"{'='*70}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ReSpeaker microphone array directivity')
    parser.add_argument('--resolution', type=int, default=36, 
                        help='Number of measurement points (default: 36)')
    parser.add_argument('--rotation-time', type=float, default=90.0,
                        help='Time for one rotation in seconds (default: 90.0)')
    parser.add_argument('--device', type=int, default=0,
                        help='Audio device index (default: 1)')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Audio sample duration in seconds (default: 2.0)')
    parser.add_argument('--rotation-direction', type=str, choices=['clockwise', 'counterclockwise'], default='counterclockwise',
                        help='Direction of turntable rotation (default: counterclockwise)')
    parser.add_argument('--output', type=str, default='data/directivity',
                        help='Output directory (default: data/directivity)')
    parser.add_argument('--save-peaks', action='store_true',
                        help='Save peak measurements (default: False)')
    parser.add_argument('--signal-type', type=str, choices=['burst', 'continuous'], default='burst',
                        help='Type of test signal (default: burst)')
    parser.add_argument('--burst-period', type=float, default=0.4,
                        help='Duration of burst signal in seconds (default: 0.4, only used if signal-type is burst)')
    parser.add_argument('--doa-locked', action='store_true',
                        help='Lock the beamformer during measurements (default: False)')
    parser.add_argument('--doa-lock-angle', type=int, default=0,
                        help='Angle to lock the DOA at if locking is enabled (default: 0°)')
   
    args = parser.parse_args()
    
    config = {
        'resolution': args.resolution,
        'seconds_per_rotation': args.rotation_time,
        'device_index': args.device,
        'sample_duration': args.duration,
        'rotation_direction': args.rotation_direction,
        'output_dir': args.output,
        'save_peaks': args.save_peaks,
        'signal_type': args.signal_type,
        'burst_period': args.burst_period,
        'doa_locked': args.doa_locked,
        'doa_lock_angle': args.doa_lock_angle
    }
    
    # Recommanded settings for Measurements:
    # 1:
    #   - resolution: 36 (10° steps)
    #   - rotation_time: 90 seconds (1.5 minutes per rotation)
    #   - sample_duration: 2 seconds
    #   => 2.5 seconds per measurement
    # = sudo .venv/bin/python3 respeaker/test_directivity.py --resolution 36 --rotation-time 90 --duration 2
    
    # 2:
    #   - resolution: 50 (7.2° steps)
    #   - rotation_time: 120 seconds (2 minutes per rotation)
    #   - sample_duration: 2 second 
    #   => 2.5 seconds per measurement
    #   + Set AGC gain to 26 for similar performances
    # = sudo .venv/bin/python3 respeaker/test_directivity.py --resolution 50 --rotation-time 120 --duration 2 --signal-type continuous --doa-locked true --doa-lock-angle 180
    
    test_directivity(
        resolution=config['resolution'],
        seconds_per_rotation=config['seconds_per_rotation'],
        device_index=config['device_index'],
        sample_duration=config['sample_duration'],
        rotation_direction=config['rotation_direction'],
        output_dir=config['output_dir'],
        save_peaks=config['save_peaks'],
        signal_type=config['signal_type'],
        burst_period=config['burst_period'],
        doa_locked=config['doa_locked'],
        doa_lock_angle=config['doa_lock_angle']
    )
