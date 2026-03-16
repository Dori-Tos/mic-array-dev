import numpy as np
import pandas as pd
import math
import sounddevice
import time
from datetime import datetime
from pathlib import Path


def test_directivity_standard_mic(
    resolution=50,  # Number of measurement points (50 = 7.2° steps)
    seconds_per_rotation=60,  # Time for one complete rotation in seconds
    device_index=None,  # Audio device index (None = default)
    sample_duration=0.8,  # Audio sample duration in seconds
    rotation_direction='counterclockwise',  # Rotation direction
    output_dir='data/directivity/directivity_dynamic_mic',
    reference_angle=0,  # Reference angle (where microphone points at start)
    use_full_scale=False,  # Whether to use full scale (1.0) for dBFS calculation
    reference_max_rms=None  # Optional reference max RMS value for normalizing across measurements      
):
    """
    Perform directivity measurements with a standard microphone and audio interface.
    
    This is a simplified version for testing with dynamic/condenser microphones
    connected to standard audio interfaces (e.g., Roland UA-101, Focusrite, etc.)
    
    Args:
        resolution: Number of measurement points around 360°
        seconds_per_rotation: Time for one complete turntable rotation (seconds)
        device_index: Audio device index (None = use default input device)
        sample_duration: Duration to record audio at each point (seconds)
        rotation_direction: Direction of turntable rotation ('clockwise' or 'counterclockwise')
        output_dir: Directory to save measurement data
        reference_angle: Angle where microphone is pointing initially (default: 0°)
        use_full_scale: If True, use 1.0 as full scale. If False, normalize to max RMS in measurements.
        reference_max_rms: Optional max RMS value to use for normalization (allows comparing multiple measurements)
    Returns:
        DataFrame with measurements
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate timing
    degrees_per_measurement = 360 / resolution
    interval = seconds_per_rotation * (degrees_per_measurement / 360)
    
    # Get device info
    if device_index is None:
        device_info = sounddevice.query_devices(kind='input')
        device_name = device_info['name']
        device_index = device_info['index']
    else:
        device_info = sounddevice.query_devices(device_index)
        device_name = device_info['name']
    
    # Determine sample rate from device
    default_samplerate = int(device_info['default_samplerate'])
    
    print(f"Standard Microphone Directivity Test Configuration:")
    print(f"  Audio device: {device_name} (index {device_index})")
    print(f"  Sample rate: {default_samplerate} Hz")
    print(f"  Resolution: {resolution} points ({degrees_per_measurement:.1f}° per measurement)")
    print(f"  Turntable speed: {seconds_per_rotation} seconds/rotation")
    print(f"  Rotation direction: {rotation_direction}")
    print(f"  Measurement interval: {interval:.2f} seconds")
    print(f"  Sample duration: {sample_duration} seconds")
    print(f"  Total test duration: ~{resolution * interval / 60:.1f} minutes")
    print(f"  Microphone reference angle: {reference_angle}°")
    print(f"  Output directory: {output_path}")
    print(f"  Use full scale: {use_full_scale}")
    if reference_max_rms is not None:
        print(f"  Reference max RMS: {reference_max_rms:.6f} ({20*math.log10(reference_max_rms):.1f} dB)")
    print("-" * 70)
    
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
            
            # Calculate relative angle (source angle relative to microphone direction)
            relative_angle = (expected_angle - reference_angle) % 360
            
            # Record audio sample
            audio_data = sounddevice.rec(
                int(sample_duration * default_samplerate),
                samplerate=default_samplerate,
                channels=1,  # Mono
                dtype='float32',  # Standard audio interfaces use float
                device=device_index
            )
            sounddevice.wait()
            
            # Process audio
            audio_data = audio_data.flatten()
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Peak value
            peak = np.max(np.abs(audio_data))
            
            # Store measurement (dB values will be calculated after collecting all data)
            measurement = {
                'measurement_index': meas_idx,
                'expected_angle': expected_angle,
                'relative_angle': relative_angle,
                'timestamp': datetime.now().isoformat(),
                'reference_angle': reference_angle,
                'rms_level': rms,
                'peak_level': peak
            }
            measurements.append(measurement)
            
            # Progress report (temporary dB calculation for display)
            temp_ref = reference_max_rms if reference_max_rms is not None else (1.0 if use_full_scale else rms)
            temp_rms_db = 20 * math.log10(max(rms, 1e-10) / temp_ref) if temp_ref > 0 else -math.inf
            temp_peak_db = 20 * math.log10(max(peak, 1e-10) / temp_ref) if temp_ref > 0 else -math.inf
            
            print(f"[{meas_idx+1}/{resolution}] "
                  f"Angle: {expected_angle:6.1f}° | "
                  f"Relative: {relative_angle:6.1f}° | "
                  f"RMS: {temp_rms_db:6.1f} dB | "
                  f"Peak: {temp_peak_db:6.1f} dB")
            
            # Wait for next measurement (accounting for processing time)
            elapsed = time.time() - measurement_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    
    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user")
        return None
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(measurements)
    
    # Determine reference value for dB calculation
    if use_full_scale:
        reference_value = 1.0
        reference_source = "full scale (1.0)"
    elif reference_max_rms is not None:
        reference_value = reference_max_rms
        reference_source = f"provided reference ({reference_max_rms:.6f})"
    else:
        reference_value = df['rms_level'].max()
        reference_source = f"maximum RMS in measurements ({reference_value:.6f})"
    
    print(f"\n\nCalculating dB values using {reference_source}...")
    
    # Calculate dB values with chosen reference
    min_level = 1e-10
    df['rms_dbfs'] = 20 * np.log10(np.maximum(df['rms_level'], min_level) / reference_value)
    df['peak_dbfs'] = 20 * np.log10(np.maximum(df['peak_level'], min_level) / reference_value)
    
    # Store the reference value used (this will be the same for all rows)
    df['reference_rms_used'] = reference_value
    
    # Save CSV
    csv_file = output_path / f"directivity_{test_timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Calculate statistics
    if len(measurements) > 0:
        rms_min = df['rms_dbfs'].min()
        rms_max = df['rms_dbfs'].max()
        rms_range = rms_max - rms_min
        rms_min_angle = df.loc[df['rms_dbfs'].idxmin(), 'expected_angle']
        rms_max_angle = df.loc[df['rms_dbfs'].idxmax(), 'expected_angle']
        
        print(f"\n{'='*70}")
        print(f"Measurement complete!")
        print(f"  Total measurements: {len(measurements)}")
        print(f"  Reference value used: {reference_value:.6f} ({reference_source})")
        print(f"  RMS Level Range: {rms_range:.1f} dB ({rms_min:.1f} to {rms_max:.1f} dB)")
        print(f"    Maximum at: {rms_max_angle:.1f}°")
        print(f"    Minimum at: {rms_min_angle:.1f}°")
        print(f"\n  Data saved to: {csv_file}")
        print(f"\n  To compare with this measurement in future runs, use:")
        print(f"    --reference-max-rms {reference_value:.8f}")
        print(f"{'='*70}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test standard microphone directivity (e.g., AKG, Shure, etc.)')
    parser.add_argument('--resolution', type=int, default=50, 
                        help='Number of measurement points (default: 50)')
    parser.add_argument('--rotation-time', type=float, default=120.0,
                        help='Time for one rotation in seconds (default: 120.0)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--duration', type=float, default=2.0,
                        help='Audio sample duration in seconds (default: 2.0)')
    parser.add_argument('--rotation-direction', type=str, 
                        choices=['clockwise', 'counterclockwise'], 
                        default='counterclockwise',
                        help='Direction of turntable rotation (default: counterclockwise)')
    parser.add_argument('--output', type=str, default='data/directivity_standard_mic',
                        help='Output directory (default: data/directivity_standard_mic)')
    parser.add_argument('--reference-angle', type=int, default=0,
                        help='Initial microphone pointing direction (default: 0°)')
    parser.add_argument('--use-full-scale', action='store_true',
                        help='Use full scale (1.0) for dB calculation instead of max RMS (default: False)')
    parser.add_argument('--reference-max-rms', type=float, default=None,
                        help='Reference max RMS value for normalization (allows comparing multiple measurements)')
   
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print("=" * 70)
        devices = sounddevice.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default_marker = " [DEFAULT]" if i == sounddevice.default.device[0] else ""
                print(f"  [{i}] {dev['name']}{default_marker}")
                print(f"      Max input channels: {dev['max_input_channels']}")
                print(f"      Default sample rate: {dev['default_samplerate']} Hz")
                print()
        print("=" * 70)
        exit(0)
    
    # Usage examples:
    # 
    # List available devices:
    # python dynamic_mic/test_directivity_dynamic_mic.py --list-devices
    #
    # Compare with previous measurement using its max RMS value:
    # python dynamic_mic/test_directivity_dynamic_mic.py --device 1 --resolution 50 --rotation-time 60 --duration 0.8 --reference-max-rms 0.12345678
    #
    # AKG FGN99E gooseneck microphone test:
    # python dynamic_mic/test_directivity_dynamic_mic.py --device 1 --resolution 50 --rotation-time 60 --duration 0.8 --reference-angle 0
    
    test_directivity_standard_mic(
        resolution=args.resolution,
        seconds_per_rotation=args.rotation_time,
        device_index=args.device,
        sample_duration=args.duration,
        rotation_direction=args.rotation_direction,
        output_dir=args.output,
        reference_angle=args.reference_angle,
        use_full_scale=args.use_full_scale,
        reference_max_rms=args.reference_max_rms
    )
