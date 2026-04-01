"""
Beamformer Diagnostic Test
Tests DAS vs MVDR to isolate beamformer instability.

This script:
1. Captures raw 4-mic audio
2. Applies ONLY beamforming (no other processing)
3. Plays back beamformed output
4. Logs statistics to identify numerical issues

Modify BEAMFORMER_TYPE to test:
- "DAS": Simple delay-and-sum (baseline)
- "MVDR": Adaptive MVDR (potentially unstable)

If DAS is clean and MVDR has artifacts → MVDR is causing the problem
If both have artifacts → Problem is elsewhere (downsampling, AGC, etc.)
"""

from classes.Microphone import Microphone
from classes.Beamformer import DASBeamformer, MVDRBeamformer
import sounddevice as sd
import numpy as np
import logging
import time
import sys
from pathlib import Path

# DIAGNOSTIC: Choose beamformer type
BEAMFORMER_TYPE = "MVDR"  # Change to "DAS" for comparison
# BEAMFORMER_TYPE = "DAS"

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    blocksize = 2048
    
    logger = logging.getLogger("BeamformerDiagnostic")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    mic_channel_numbers = [0, 1, 2, 3]
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    
    # Load mic positions from XML
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    try:
        # Create beamformer based on selection
        if BEAMFORMER_TYPE == "DAS":
            beamformer = DASBeamformer(
                logger=logger,
                mic_channel_numbers=mic_channel_numbers,
                sample_rate=sample_rate,
                mic_positions_m=mic_positions,
            )
            logger.info("Using DAS Beamformer (simple baseline)")
        else:
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
            logger.info("Using MVDR Beamformer (adaptive)")
        
        # Start 4-channel input
        input_stream = sd.InputStream(
            channels=4,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        input_stream.start()
        logger.info(f"Input stream started: {sample_rate}Hz, 4 channels")
        
        # Start mono output
        output_stream = sd.OutputStream(
            channels=1,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype=np.float32
        )
        output_stream.start()
        logger.info("Output stream started")
        
        time.sleep(1.0)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Beamformer Diagnostic: {BEAMFORMER_TYPE}")
        logger.info("Raw beamformed output (mono) will be played. Listen for artifacts...")
        logger.info("Press Ctrl+C to stop")
        logger.info(f"{'='*70}\n")
        
        block_count = 0
        artifact_count = 0
        
        try:
            while True:
                # Read 4-channel block
                audio_block, _ = input_stream.read(blocksize)  # Shape: (blocksize, 4)
                
                # Apply beamformer (DOA frozen at 0°)
                beamformer.set_steering_angle(0.0)
                beamformed = beamformer.apply(audio_block)
                
                # Convert to mono float32
                mono = np.asarray(beamformed, dtype=np.float32).reshape(-1)
                
                # Compute statistics
                rms = float(np.sqrt(np.mean(mono ** 2)))
                peak = float(np.max(np.abs(mono)))
                
                # Check for numerical pathologies
                has_nan = np.any(np.isnan(mono))
                has_inf = np.any(np.isinf(mono))
                num_extreme = np.sum(np.abs(mono) > 10.0)
                
                # Log periodically
                if block_count % 20 == 0:
                    status = "OK"
                    if has_nan or has_inf or num_extreme > 0:
                        status = "ALERT"
                        artifact_count += 1
                    logger.info(
                        f"[Block {block_count:5d}] RMS: {rms:.6f}  Peak: {peak:.6f}  "
                        f"NaN:{has_nan}  Inf:{has_inf}  Extreme:{num_extreme}  [{status}]"
                    )
                
                # Play beamformed output
                output_stream.write(mono.reshape(-1, 1))
                
                block_count += 1
        
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Diagnostic Complete")
        logger.info(f"Processed {block_count} blocks, {artifact_count} blocks with issues")
        logger.info(f"Beamformer type: {BEAMFORMER_TYPE}")
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
