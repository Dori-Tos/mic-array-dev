from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter
from classes.AGC import AGC
from classes.Codec import G711Codec, OpusCodec

import time
import numpy as np
from pathlib import Path
import logging

import sounddevice as sd


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    downsample_rate = 16000  # Downsample from 48 kHz to 16 kHz for faster processing
    cutoff_freq = 6000.0
    order = 4
    monitor_gain = 0.2
    
    mic_channel_numbers = [0, 1, 2, 3]
    
    logger = logging.getLogger("MicArrayTest")
    logger.setLevel(logging.DEBUG)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    geometry_path = script_dir / "array_geometries" / "1_Square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    # DAS for DOA estimation (faster than MVDR)
    das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    # 
    mvdr_beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        doa_beamformer=mvdr_beamformer,  # Use DAS for fast DOA scanning
        scan_step_deg=5.0,
    )
    echo_canceller = EchoCanceller(logger=logger, sample_rate=sample_rate, channels=4)
    filters = [LowPassFilter(logger=logger, sample_rate=sample_rate, cutoff_freq=cutoff_freq, order=order)]
    agc = AGC()
    codec = G711Codec(logger=logger)

    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=mvdr_beamformer,  # Use MVDR for high-quality output beamforming
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec,
        monitor_gain=monitor_gain,
        downsample_rate=downsample_rate  # Process at 16kHz instead of 48kHz for ~3x speedup
    )

    array.start_realtime(blocksize=2048)
    array.start_output_monitoring()

    try:
        time.sleep(1)
        print("Realtime beamformed monitoring started. Press Ctrl+C to stop.")
        while True:
            doa = array.get_latest_doa()
            if doa is not None:
                print(f"DOA: {doa:.1f}°")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("Stopping...")
        
    finally:
        array.stop_realtime()