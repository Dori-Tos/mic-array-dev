"""
Single Microphone Diagnostic Test
Tests each microphone channel individually with NO filtering, NO beamforming.
Allows listening to raw microphone output to identify hardware issues.

Usage:
    python test_single_mic.py [channel_number]
    
Example:
    python test_single_mic.py 0  # Listen to mic 0
    python test_single_mic.py 1  # Listen to mic 1
    python test_single_mic.py 2  # Listen to mic 2
    python test_single_mic.py 3  # Listen to mic 3
    python test_single_mic.py    # Cycle through all 4 mics
"""

from classes.Microphone import Microphone
import sounddevice as sd
import numpy as np
import logging
import time
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    blocksize = 2048
    
    logger = logging.getLogger("MicDiagnostic")
    logger.setLevel(logging.INFO)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Determine which channel to test
    if len(sys.argv) > 1:
        try:
            test_channel = int(sys.argv[1])
            channels_to_test = [test_channel]
        except ValueError:
            logger.error(f"Invalid channel: {sys.argv[1]}")
            sys.exit(1)
    else:
        channels_to_test = [0, 1, 2, 3, 4, 5, 6, 7]  # Test all 8 mics in sequence
    
    mic_channel_numbers = [0, 1, 2, 3, 4, 5, 6, 7]
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    
    try:
        # Start audio capture (8-channel)
        stream = sd.InputStream(
            channels=8,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        stream.start()
        logger.info(f"Audio stream started: {sample_rate}Hz, 8 channels")
        
        # Create output stream (mono playback)
        output_stream = sd.OutputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        output_stream.start()
        logger.info("Output stream started for playback")
        
        time.sleep(1.0)  # Let streams stabilize
        
        for channel in channels_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Microphone Channel {channel}")
            logger.info(f"{'='*60}")
            logger.info("Raw audio from this channel will be played. Listen for artifacts...")
            logger.info("Press Ctrl+C to skip to next mic or exit")
            
            time.sleep(2.0)  # Give user time to prepare
            
            block_count = 0
            start_time = time.time()
            max_duration = 30.0  # 30 seconds per mic
            
            try:
                while time.time() - start_time < max_duration:
                    # Read 4-channel block
                    audio_block, _ = stream.read(blocksize)  # Shape: (blocksize, 4)
                    
                    # Extract single channel (mono)
                    mono = audio_block[:, channel].astype(np.float32)
                    
                    # Display statistics
                    rms = float(np.sqrt(np.mean(mono ** 2)))
                    peak = float(np.max(np.abs(mono)))
                    
                    if block_count % 10 == 0:  # Log every 10 blocks (~0.1s)
                        logger.info(f"[Ch{channel}] RMS: {rms:.6f}, Peak: {peak:.6f}")
                    
                    # Check for NaN or Inf
                    if np.any(np.isnan(mono)) or np.any(np.isinf(mono)):
                        logger.warning(f"[Ch{channel}] ALERT: NaN or Inf detected!")
                    
                    # Check for extreme values
                    if peak > 1.5:
                        logger.warning(f"[Ch{channel}] CLIPPING: Peak value {peak:.4f} exceeds ±1.5")
                    
                    # Play to output (mono)
                    output_stream.write(mono.reshape(-1, 1))
                    
                    block_count += 1
                
                elapsed = time.time() - start_time
                logger.info(f"[Ch{channel}] Completed: {block_count} blocks in {elapsed:.1f}s")
                
            except KeyboardInterrupt:
                logger.info(f"[Ch{channel}] Skipped by user")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("Microphone diagnostic complete")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        
    finally:
        stream.stop()
        stream.close()
        output_stream.stop()
        output_stream.close()
        logger.info("Streams closed")
