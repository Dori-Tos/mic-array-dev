import collections
import threading
import wave

import numpy as np
import pandas as pd
import math
import sounddevice as sd
import time
from datetime import datetime
from pathlib import Path

def _bytes_to_human(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"

def save_output_dynamic_mic(
    device_index=None,
    output_dir='data/test_protocol/4_save_output_dynamic_mic',
    blocksize=1024,
    listen_output=True,
	warn_size_mb: float = 512.0,
	hard_limit_mb: float = 2048.0,
    monitor_gain: float = 0.22,
):
    """
    Save raw audio from a standard dynamic microphone to a WAV file.
    
    :param device_index: Optional index of the audio input device to use (default: system default)
    :param output_dir: Directory where the output CSV file will be saved (default: data/test_protocol/4_save_output_dynamic_mic)
    :param blocksize: Size of each audio block to read (default: 1024)
    :param listen_output: Whether to listen to the output audio in real-time (default: True)
    :param warn_size_mb: Size in megabytes at which to print a warning about large file size (default: 512 MB)
    :param hard_limit_mb: Size in megabytes at which to stop recording to prevent excessive disk usage (default: 2048 MB)
    """
    
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
    
    # Determine sample rate from device
    default_samplerate = int(device_info['default_samplerate'])
    
    print(f"Standard Microphone Directivity Test Configuration:")
    print(f"  Audio device: {device_name} (index {device_index})")
    print(f"  Sample rate: {default_samplerate} Hz")
    print(f"  Output directory: {output_path}")
    print("-" * 70)
    
    
    warn_bytes = int(max(1.0, float(warn_size_mb)) * 1024 * 1024)
    hard_bytes = int(max(float(warn_size_mb) + 1.0, float(hard_limit_mb)) * 1024 * 1024)
    
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    final_file = output_path / f"raw_output_{test_timestamp}.wav"
    temp_file = output_path / f"raw_output_{test_timestamp}.tmp.wav"
    
    print("\nStarting recording...")
    print("Press Ctrl+C to stop early\n")
    
    bytes_written = 0
    warned = False
    save_allowed = True
    interrupted_by_user = False
    start_time = time.time()
    monitor_lock = threading.Lock()
    monitor_fifo: collections.deque[np.ndarray] = collections.deque()
    monitor_current = np.zeros(0, dtype=np.float32)
    monitor_stream = None

    def _monitor_callback(outdata, frames, time_info, status):
        nonlocal monitor_current
        if status:
            pass
        chunk = np.zeros(frames, dtype=np.float32)
        write_idx = 0
        with monitor_lock:
            while write_idx < frames:
                if monitor_current.size == 0:
                    if not monitor_fifo:
                        break
                    monitor_current = monitor_fifo.popleft()
                remaining = frames - write_idx
                take = min(remaining, monitor_current.size)
                if take <= 0:
                    break
                chunk[write_idx:write_idx + take] = monitor_current[:take]
                monitor_current = monitor_current[take:]
                write_idx += take
        outdata[:, 0] = np.clip(chunk * float(monitor_gain), -1.0, 1.0)

    if listen_output:
        monitor_stream = sd.OutputStream(
            samplerate=default_samplerate,
            channels=1,
            dtype="float32",
            latency="low",
            blocksize=int(blocksize),
            callback=_monitor_callback,
        )
        
    wav_writer = wave.open(str(temp_file), "wb")
    wav_writer.setnchannels(1)
    wav_writer.setsampwidth(2)
    wav_writer.setframerate(default_samplerate)
    
    try:
        
        with sd.InputStream(
            samplerate=default_samplerate,
            channels=1,
            device=int(device_index),
            dtype="float32",
            blocksize=int(blocksize),
        ) as stream:

            if monitor_stream is not None:
                monitor_stream.start()

            while True:
                block, _overflowed = stream.read(int(blocksize))
                block = np.asarray(block, dtype=np.float32)

                mono = np.asarray(block, dtype=np.float32).reshape(-1)
                mono = np.clip(mono, -1.0, 1.0)
                if listen_output:
                    with monitor_lock:
                        monitor_fifo.append(mono.copy())
                        while len(monitor_fifo) > 12:
                            monitor_fifo.popleft()
                pcm16 = (mono * 32767.0).astype(np.int16)

                wav_writer.writeframes(pcm16.tobytes())
                bytes_written += int(pcm16.nbytes)
                if (not warned) and bytes_written >= warn_bytes:
                    warned = True
                    print(
                        f"WARNING: output is getting large ({_bytes_to_human(bytes_written)}). "
                        f"Hard stop at {_bytes_to_human(hard_bytes)}."
                    )

                if bytes_written >= hard_bytes:
                    print(
                        f"ERROR: hard output limit exceeded ({_bytes_to_human(bytes_written)}). "
                        f"Aborting and discarding recording."
                    )
                    save_allowed = False
                    break
    
    
    except KeyboardInterrupt:
        interrupted_by_user = True
        print("\nCtrl+C received. Finalizing output...")

    finally:
        if monitor_stream is not None:
            try:
                monitor_stream.stop()
                monitor_stream.close()
            except Exception:
                pass
        wav_writer.close()
        
    if not save_allowed:
        try:
            if temp_file.exists():
                temp_file.unlink()
        finally:
            print("Recording aborted due to hard size limit. No file saved.")
        return
      
    temp_file.replace(final_file)
    elapsed = time.time() - start_time
    print("\nSaved processed output WAV:")
    print(f"  File: {final_file}")
    print(f"  Duration (approx): {elapsed:.1f} s")
    print(f"  Audio bytes: {_bytes_to_human(bytes_written)}")
    if not interrupted_by_user:
        print("  Capture ended normally.")
     
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test standard microphone directivity (e.g., AKG, Shure, etc.)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: system default)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--output', type=str, default='data/directivity_standard_mic',
                        help='Output directory (default: data/directivity_standard_mic)')
    parser.add_argument('--blocksize', type=int, default=1024,
                        help='Audio block size (default: 1024)')
    parser.add_argument('--listen-output', action='store_true',
                        help='Listen to the output audio in real-time (default: True)')
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print("=" * 70)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                default_marker = " [DEFAULT]" if i == sd.default.device[0] else ""
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
    
    save_output_dynamic_mic(
        device_index=args.device,
        output_dir=args.output,
    )
