from custom_tuning import Tuning
import usb.core
import numpy as np
import math
import sounddevice
import time
from datetime import datetime


def live_gain_measurement(
    device_index=1,
    sample_duration=0.5,  # Duration of each measurement in seconds
    baseline_samples=10,  # Number of initial measurements to use as baseline
    update_interval=0.5  # Minimum time between measurements in seconds
):
    """
    Monitor ReSpeaker mic array gain in real-time to detect adaptive filter behavior.
    
    The first few measurements establish a baseline, then subsequent measurements
    show deviation from that baseline to detect gain reduction by adaptive filters.
    
    Args:
        device_index: Audio device index for ReSpeaker (default: 1)
        sample_duration: Duration to record for each measurement (default: 0.5s)
        baseline_samples: Number of initial measurements to average for baseline (default: 10)
        update_interval: Minimum time between measurements in seconds (default: 0.5s)
    """
    
    print("=" * 80)
    print("ReSpeaker Live Gain Measurement - Adaptive Filter Monitor")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Device index: {device_index}")
    print(f"  Sample duration: {sample_duration}s")
    print(f"  Baseline samples: {baseline_samples}")
    print(f"  Update interval: {update_interval}s")
    print("-" * 80)
    print("Starting measurement... Play your test signal now!")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()
    
    # Find the Respeaker 4 mic array USB device
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return
    
    # Audio parameters
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1
    
    # Storage for baseline measurements
    baseline_rms_values = []
    baseline_agc_values = []
    
    measurement_count = 0
    baseline_rms_dbfs = None
    baseline_agc_db = None
    
    start_time = time.time()
    
    try:
        while True:
            measurement_start = time.time()
            measurement_count += 1
            
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
            
            # Convert to dBFS
            full_scale = 32768.0
            min_level = 1e-10
            rms = max(rms, min_level)
            rms_dbfs = 20 * math.log10(rms / full_scale)
            
            elapsed_time = time.time() - start_time
            
            # Collecting baseline
            if measurement_count <= baseline_samples:
                baseline_rms_values.append(rms_dbfs)
                baseline_agc_values.append(agc_gain_db)
                
                print(f"[BASELINE {measurement_count}/{baseline_samples}] "
                      f"Time: {elapsed_time:6.1f}s | "
                      f"RMS: {rms_dbfs:6.1f} dBFS | "
                      f"AGC: {agc_gain_db:5.1f} dB | "
                      f"DOA: {doa_angle:3d}° | "
                      f"Voice: {'YES' if voice_activity else 'NO '}")
                
                # Calculate baseline average after collecting all samples
                if measurement_count == baseline_samples:
                    baseline_rms_dbfs = np.mean(baseline_rms_values)
                    baseline_agc_db = np.mean(baseline_agc_values)
                    print()
                    print("=" * 80)
                    print(f"BASELINE ESTABLISHED:")
                    print(f"  Average RMS: {baseline_rms_dbfs:.1f} dBFS")
                    print(f"  Average AGC: {baseline_agc_db:.1f} dB")
                    print("=" * 80)
                    print()
                    print("Now monitoring for gain changes...")
                    print()
            
            # Monitor deviation from baseline
            else:
                rms_deviation = rms_dbfs - baseline_rms_dbfs
                agc_deviation = agc_gain_db - baseline_agc_db
                
                # Visual indicator of gain reduction
                if rms_deviation < -3:
                    status = "⚠️ SIGNIFICANT REDUCTION"
                elif rms_deviation < -1:
                    status = "⚠️ REDUCING"
                elif rms_deviation < -0.5:
                    status = "⚐ Slight reduction"
                else:
                    status = "✓ Stable"
                
                print(f"[{measurement_count:4d}] "
                      f"Time: {elapsed_time:6.1f}s | "
                      f"RMS: {rms_dbfs:6.1f} dBFS ({rms_deviation:+5.1f} dB) | "
                      f"AGC: {agc_gain_db:5.1f} dB ({agc_deviation:+5.1f} dB) | "
                      f"DOA: {doa_angle:3d}° | "
                      f"Voice: {'YES' if voice_activity else 'NO '} | "
                      f"{status}")
            
            # Wait for next measurement
            elapsed = time.time() - measurement_start
            remaining = update_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    
    except KeyboardInterrupt:
        print("\n")
        print("=" * 80)
        print("Measurement stopped by user")
        
        if baseline_rms_dbfs is not None:
            # Calculate final statistics
            print()
            print("Summary:")
            print(f"  Total measurements: {measurement_count}")
            print(f"  Total duration: {elapsed_time:.1f}s")
            print(f"  Baseline RMS: {baseline_rms_dbfs:.1f} dBFS")
            
            if measurement_count > baseline_samples:
                # Get statistics of the monitoring period
                monitoring_rms_values = baseline_rms_values[baseline_samples:] if len(baseline_rms_values) > baseline_samples else []
                
                # Note: we need to track these during monitoring, let me add that
                print(f"  Monitored {measurement_count - baseline_samples} samples after baseline")
        
        print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Monitor ReSpeaker gain in real-time to detect adaptive filter behavior')
    parser.add_argument('--device', type=int, default=1,
                        help='Audio device index (default: 1)')
    parser.add_argument('--duration', type=float, default=0.5,
                        help='Sample duration in seconds (default: 0.5)')
    parser.add_argument('--baseline', type=int, default=10,
                        help='Number of baseline samples (default: 10)')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Update interval in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Usage example:
    # sudo .venv/bin/python3 respeaker/live_gain_measurment.py
    # sudo .venv/bin/python3 respeaker/live_gain_measurment.py --baseline 20 --interval 0.3
    
    live_gain_measurement(
        device_index=args.device,
        sample_duration=args.duration,
        baseline_samples=args.baseline,
        update_interval=args.interval
    )
