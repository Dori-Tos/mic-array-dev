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
    sample_duration=0.5,  # Audio sample duration in seconds
    output_dir='data/directivity'
):
    """
    Perform directivity measurements of the ReSpeaker mic array.
    
    Args:
        resolution: Number of measurement points around 360°
        seconds_per_rotation: Time for one complete turntable rotation (seconds)
        device_index: Audio device index for ReSpeaker
        sample_duration: Duration to record audio at each point (needed to calculate RMS/Peak levels)
        output_dir: Directory to save measurement data
    
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
    test_timestamp = datetime.now().strftime("%Y-%m-%d %H%:M%S")
    
    print("\nStarting measurements...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        for i in range(resolution):
            measurement_start = time.time()
            
            # Expected angle based on position (assuming starting at 0°)
            expected_angle = (i * degrees_per_measurement) % 360
            
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
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            
            # Convert to dBFS
            full_scale = 32768.0
            rms_dbfs = 20 * math.log10(rms / full_scale) if rms > 0 else -float('inf')
            peak_dbfs = 20 * math.log10(peak / full_scale) if peak > 0 else -float('inf')
            
            # Store measurement
            measurement = {
                'measurement_index': i,
                'expected_angle': expected_angle,
                'timestamp': datetime.now().isoformat(),
                'doa_angle': doa_angle,
                'agc_gain': agc_gain,
                'agc_gain_db': agc_gain_db,
                'voice_activity': voice_activity,
                'rms_level': rms,
                'rms_dbfs': rms_dbfs,
                'peak_level': peak,
                'peak_dbfs': peak_dbfs
            }
            measurements.append(measurement)
            
            # Progress report
            print(f"[{i+1}/{resolution}] Angle: {expected_angle:6.1f}° | "
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
    
    # Convert to DataFrame
    df = pd.DataFrame(measurements)
    
    # Save data in multiple formats
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
    parser.add_argument('--device', type=int, default=1,
                        help='Audio device index (default: 1)')
    parser.add_argument('--duration', type=float, default=0.5,
                        help='Audio sample duration in seconds (default: 0.5)')
    parser.add_argument('--output', type=str, default='data/directivity',
                        help='Output directory (default: data/directivity)')
    
    args = parser.parse_args()
    
    test_directivity(
        resolution=args.resolution,
        seconds_per_rotation=args.rotation_time,
        device_index=args.device,
        sample_duration=args.duration,
        output_dir=args.output
    )
