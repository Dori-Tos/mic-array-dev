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
    cutoff_freq = 6000.0
    order = 4
    monitor_gain = 0.2
    
    mic_channel_numbers = [0, 1, 2, 3]
    
    logger = logging.getLogger("MicArrayTest")
    logger.setLevel(logging.DEBUG)

    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=1.0,
        angle_range=(-25, 25),
        doa_beamformer=beamformer,
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
        beamformer=beamformer,
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec
    )
    
    array.start_realtime(blocksize=2048)

    try:
        array.test_all_microphones(duration_seconds=2.0)

    except KeyboardInterrupt:
        print("Stopping...")
        
    finally:
        array.stop_realtime()