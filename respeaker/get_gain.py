from custom_tuning import Tuning
import usb.core
import pyaudio
import numpy as np
import math
import sounddevice


def get_gain(device_index=2):  # Card2 device 0
    # Find the Respeaker 4 mic array USB device
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("Respeaker 4 mic array not found")
        return None
    
    # Audio parameters (Card2 device 0)
    RESPEAKER_RATE = 16000
    RESPEAKER_CHANNELS = 1
    RESPEAKER_WIDTH = 2
    RESPEAKER_INDEX = device_index
    CHUNK = 1024
    SAMPLE_DURATION = 0.5  # seconds to sample for level measurement
    
    # Create Tuning object to interact with the device
    tuning = Tuning(mic_array)
    
    # Read all tuning parameters first
    agc_gain = tuning.read('AGCGAIN')
    agc_gain_db = 20 * math.log10(agc_gain) if agc_gain and agc_gain > 0 else -float('inf')
    agc_max_gain = tuning.read('AGCMAXGAIN')
    agc_desired_level = tuning.read('AGCDESIREDLEVEL')
    doa_angle = tuning.direction
    voice_activity = tuning.is_voice
    
    # Close the USB control connection before opening audio stream
    # This prevents segmentation faults from simultaneous USB device access
    tuning.close()
    
    print(f"AGC Gain (applied): {agc_gain:.2f} (dB: {agc_gain_db:.2f})")
    print(f"AGC Max Gain: {agc_max_gain:.2f}")
    print(f"AGC Desired Level: {agc_desired_level:.6f}")
    print(f"DOA Angle: {doa_angle}°")
    print(f"Voice Activity: {'Yes' if voice_activity else 'No'}")
    
    # Now measure the actual audio signal level
    print(f"\nSampling audio for {SAMPLE_DURATION}s to measure real signal level...")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,
    )
    
    frames = []
    num_chunks = int(RESPEAKER_RATE / CHUNK * SAMPLE_DURATION)
    
    for i in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    
    # Calculate RMS (Root Mean Square) level
    rms = np.sqrt(np.mean(audio_data**2))
    
    # Calculate peak level
    peak = np.max(np.abs(audio_data))
    
    # Convert to dBFS (dB relative to full scale)
    # For 16-bit audio, full scale is 32768
    full_scale = 32768.0
    rms_dbfs = 20 * math.log10(rms / full_scale) if rms > 0 else -float('inf')
    peak_dbfs = 20 * math.log10(peak / full_scale) if peak > 0 else -float('inf')
    
    print(f"\nReal Audio Signal Levels:")
    print(f"  RMS Level: {rms:.2f} ({rms_dbfs:.2f} dBFS)")
    print(f"  Peak Level: {peak:.2f} ({peak_dbfs:.2f} dBFS)")
    
    return {
        'agc_gain': agc_gain,
        'agc_gain_db': agc_gain_db,
        'agc_max_gain': agc_max_gain,
        'agc_desired_level': agc_desired_level,
        'doa_angle': doa_angle,
        'voice_activity': voice_activity,
        'rms_level': rms,
        'rms_dbfs': rms_dbfs,
        'peak_level': peak,
        'peak_dbfs': peak_dbfs
    }


if __name__ == '__main__':
    print("Reading gain and DOA from Respeaker 4 mic array...")
    print("-" * 50)
    get_gain()  # Uses Card2 device 0 (index 2) by default
