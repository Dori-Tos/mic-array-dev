from classes.Array import Array
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter
from classes.AGC import AGC
from classes.Codec import G711Codec, OpusCodec

import sounddevice as sd
import numpy as np


if __name__ == "__main__":
    sample_rate = 16000
    cutoff_freq = 6000.0
    order = 4
    
    mic_channel_numbers = [0, 1, 2, 3]

    
    mic_list = [Microphone(channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    doa_estimator = IterativeDOAEstimator(update_rate=3.0)
    beamformer = DASBeamformer(mic_channel_numbers=mic_channel_numbers)
    echo_canceller = EchoCanceller(sample_rate=sample_rate, channels=4)
    filters = [LowPassFilter(sample_rate=sample_rate, cutoff_freq=cutoff_freq, order=order)]
    agc = AGC()
    codec = G711Codec()

    array = Array(
        id_vendor=0x2886,
        id_product=0x0018,
        mic_list=mic_list,
        sampling_rate=16000,
        doa_estimator=doa_estimator,
        beamformer=beamformer,
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec
    )

    array.start_realtime(blocksize=1024)

    try:
        while True:
            doa = array.get_latest_doa()
            if doa is not None:
                print(f"Estimated DOA: {doa}°")
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopping...")