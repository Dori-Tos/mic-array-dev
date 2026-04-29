"""
Quick test to check raw microphone input RMS vs beamformer output RMS
"""
from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.DOAEstimator import IterativeDOAEstimator
from classes.AGC import AGC
from classes.EchoCanceller import EchoCanceller
from classes.Filter import LowPassFilter
from classes.Codec import G711Codec

import time
import numpy as np
from pathlib import Path
import logging
import sounddevice as sd

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    downsample_rate = 16000
    mic_channel_numbers = [0, 1, 2, 3]
    
    logger = logging.getLogger("InputLevelTest")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    geometry_path = script_dir / "array_geometries" / "1_Square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    mvdr_beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        doa_beamformer=das_beamformer,
        scan_step_deg=5.0,
    )
    
    filters = [LowPassFilter(logger=logger, sample_rate=sample_rate, cutoff_freq=6000.0, order=4)]
    
    agc = AGC(
        logger=logger,
        target_rms=0.08,
        min_gain=0.5,
        max_gain=50.0,
        attack_ms=20.0,
        release_ms=500.0,
        noise_floor_rms=0.0001,
        gate_gain=1.0,
    )
    
    echo_canceller = EchoCanceller(logger=logger, sample_rate=sample_rate, channels=4)
    
    codec = G711Codec(logger=logger)
    
    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=mvdr_beamformer,
        monitor_gain=0.2,
        downsample_rate=downsample_rate,
        filters=filters,
        agc=agc,
        echo_canceller=echo_canceller,
        codec=codec,
    )

    array.start_realtime(blocksize=2048)

    try:
        print("Capturing audio... Speak at normal level. Press Ctrl+C to stop.\n")
        start = time.time()
        
        while time.time() - start < 5:
            time.sleep(0.5)
            
            # Get raw input block
            raw_block = array.get_latest_block()
            if raw_block is not None:
                # Calculate RMS for each channel
                rms_per_channel = [np.sqrt(np.mean(raw_block[:, i]**2)) for i in range(raw_block.shape[1])]
                raw_rms_avg = np.mean(rms_per_channel)
                raw_peak = np.max(np.abs(raw_block))
                
                # Get beamformed output
                beamformed = array.get_latest_beamformed()
                if beamformed is not None:
                    beamformed_rms = np.sqrt(np.mean(beamformed**2))
                    beamformed_peak = np.max(np.abs(beamformed))
                else:
                    beamformed_rms = 0
                    beamformed_peak = 0
                
                elapsed = time.time() - start
                print(f"[{elapsed:.1f}s] RAW (int16) RMS: {raw_rms_avg:.0f} | Peak: {raw_peak:.0f} | "
                      f"BEAMFORMED (float) RMS: {beamformed_rms:.6f} | Peak: {beamformed_peak:.6f}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        array.stop_realtime()
