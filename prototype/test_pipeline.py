from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter, WienerFilter
from classes.AGC import AGC, TwoStageAGC
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
    order = 4
    monitor_gain = 0.22
    
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
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    # DAS for DOA estimation (faster than MVDR)
    das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    # MVDR for main beamforming (higher quality than DAS)
    mvdr_beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.97,
        diagonal_loading=5e-2,
    )
    
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        beamformer=das_beamformer,  # Use DAS for fast DOA scanning
        scan_step_deg=5.0,
    )
    echo_canceller = EchoCanceller(logger=logger, sample_rate=sample_rate, channels=4)
    
    # Passband to eliminate low-frequency rumble and high-frequency hiss
    # Wiener filter for continuous noise reduction
    filter_rate = downsample_rate if downsample_rate is not None else sample_rate
    filters = [
        BandPassFilter(
            logger=logger, 
            sample_rate=filter_rate, 
            low_cutoff=300.0, 
            high_cutoff=3200.0, 
            order=order),
        
        WienerFilter(
            logger=logger,
            sample_rate=filter_rate,
            noise_alpha=0.985,
            gain_floor=0.05,
            gain_smooth_alpha=0.86,
            noise_update_snr_db=6.0,
            noise_update_rms=8e-4,
        ),
    ]
    
    agc_fast = AGC(
        logger=logger,
        target_rms=0.0025,
        min_gain=0.7,
        max_gain=12.0,
        attack_ms=8.0,
        release_ms=200.0,
        noise_floor_rms=0.0,
        gate_gain=1.0,
    )
    agc_slow = AGC(
        logger=logger,
        target_rms=0.009,
        min_gain=0.9,
        max_gain=10.0,
        attack_ms=40.0,
        release_ms=2200.0,
        noise_floor_rms=0.00012,
        gate_gain=0.35,
    )
    agc = TwoStageAGC(logger=logger, stage1=agc_fast, stage2=agc_slow)
    codec = G711Codec(logger=logger)

    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=mvdr_beamformer, 
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
        print("Realtime beamformed monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        array.stop_realtime()
        array.stop_output_monitoring()