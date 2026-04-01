"""
Full Pipeline Diagnostic Test
Tests the complete real-time pipeline with beamformer, downsampling, and AGC.

This script tests configurations in order to narrow down the culprit:
1. Beamformer only (48kHz) 
2. Beamformer + downsampling to 16kHz (no AGC)
3. Beamformer + downsampling + AGC (full pipeline)

Modify TEST_MODE to test each configuration.
"""

from classes.Microphone import Microphone
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.AGC import AGC, TwoStageAGC
import sounddevice as sd
import numpy as np
import logging
import time
from pathlib import Path
from scipy import signal

# DIAGNOSTIC: Choose test mode
# TEST_MODE = 1  # Beamformer only (48kHz) - baseline
# TEST_MODE = 2  # Beamformer + downsampling (no AGC)
TEST_MODE = 3  # Beamformer + downsampling + AGC (full)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    downsample_rate = 16000
    blocksize = 2048
    
    logger = logging.getLogger("PipelineDiagnostic")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    mic_channel_numbers = [0, 1, 2, 3]
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    # Create MVDR beamformer
    beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.93,
        diagonal_loading=0.08,
        spectral_whitening_factor=0.06,
        weight_smooth_alpha=0.55,
        max_adaptive_loading_scale=5.0,
    )
    
    # Create AGC stages
    agc_fast = AGC(
        logger=logger,
        target_rms=0.003,
        min_gain=0.6,
        max_gain=8.0,
        attack_ms=35.0,
        release_ms=200.0,
        noise_floor_rms=0.0,
        gate_gain=1.0,
        gate_open_ms=30.0,
        gate_close_ms=150.0,
        gate_hold_ms=20.0,
    )
    agc_slow = AGC(
        logger=logger,
        target_rms=0.012,
        min_gain=0.8,
        max_gain=7.0,
        attack_ms=125.0,
        release_ms=2200.0,
        noise_floor_rms=0.00012,
        gate_gain=0.35,
        gate_open_ms=30.0,
        gate_close_ms=150.0,
        gate_hold_ms=100.0,
    )
    agc = TwoStageAGC(logger=logger, stage1=agc_fast, stage2=agc_slow)
    
    # Resampler state
    resampler_state = None
    
    try:
        # Input: 4 channels, 48kHz
        input_stream = sd.InputStream(
            channels=4,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        input_stream.start()
        
        # Output: mono, 48kHz (will upsample 16kHz if needed)
        output_stream = sd.OutputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        output_stream.start()
        
        time.sleep(1.0)
        
        mode_desc = {
            1: "Beamformer only (48kHz)",
            2: "Beamformer + downsampling to 16kHz (no AGC)",
            3: "Beamformer + downsampling + AGC (full pipeline)"
        }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Test Mode {TEST_MODE}: {mode_desc[TEST_MODE]}")
        logger.info("Listen for artifacts...")
        logger.info("Press Ctrl+C to stop")
        logger.info(f"{'='*70}\n")
        
        block_count = 0
        
        try:
            while True:
                # Read 4-channel, 48kHz
                audio_block, _ = input_stream.read(blocksize)  # (blocksize, 4)
                
                # Apply beamformer
                beamformer.set_steering_angle(0.0)
                beamformed = beamformer.apply(audio_block)  # (blocksize,)
                mono = np.asarray(beamformed, dtype=np.float32)
                
                # MODE 1: Beamformer only
                if TEST_MODE == 1:
                    output_data = mono
                    output_sr = sample_rate
                
                # MODE 2: Beamformer + downsampling (no AGC)
                elif TEST_MODE == 2:
                    # Downsample 48kHz → 16kHz using polyphase (matches Array_RealTime)
                    downsampled = signal.resample_poly(mono, downsample_rate, sample_rate)
                    
                    # Upsample 16kHz → 48kHz for playback using polyphase
                    upsampled = signal.resample_poly(downsampled, sample_rate, downsample_rate)
                    output_data = upsampled.astype(np.float32)
                    output_sr = sample_rate
                
                # MODE 3: Full pipeline (beamformer + downsampling + AGC)
                elif TEST_MODE == 3:
                    # Downsample 48kHz → 16kHz using polyphase (matches Array_RealTime)
                    downsampled = signal.resample_poly(mono, downsample_rate, sample_rate)
                    
                    # Apply AGC at 16kHz
                    agc_out = agc.process(downsampled, sample_rate=downsample_rate)
                    
                    # Upsample 16kHz → 48kHz for playback using polyphase
                    upsampled = signal.resample_poly(agc_out, sample_rate, downsample_rate)
                    output_data = upsampled.astype(np.float32)
                    output_sr = sample_rate
                
                # Log statistics
                if block_count % 20 == 0:
                    rms = float(np.sqrt(np.mean(output_data ** 2)))
                    peak = float(np.max(np.abs(output_data)))
                    has_nan = np.any(np.isnan(output_data))
                    has_inf = np.any(np.isinf(output_data))
                    logger.info(
                        f"[Block {block_count:5d}] RMS: {rms:.6f}  Peak: {peak:.6f}  "
                        f"NaN:{has_nan}  Inf:{has_inf}"
                    )
                
                # Play output
                output_stream.write(np.clip(output_data * 0.35, -0.95, 0.95).reshape(-1, 1))
                
                block_count += 1
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Test Mode {TEST_MODE} Complete: {block_count} blocks processed")
        logger.info(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        input_stream.stop()
        input_stream.close()
        output_stream.stop()
        output_stream.close()
        logger.info("Streams closed")
