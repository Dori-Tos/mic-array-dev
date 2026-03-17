from .Microphone import Microphone
from .DOAEstimator import DOAEstimator
from .Beamformer import Beamformer
from .EchoCanceller import EchoCanceller
from .Filter import Filter
from .AGC import AGC
from .Codec import Codec

from typing import Sequence
import numpy as np
import threading

import sounddevice as sd


class Array:
    def __init__(self, id_vendor, id_product, mic_list: list[Microphone], sampling_rate: int,
                 doa_estimator: DOAEstimator, beamformer: Beamformer, 
                 echo_canceller: EchoCanceller, filters: Sequence[Filter], agc: AGC, codec: Codec,
                 device_index: int | None = None):
        
        self.id_vendor: int = id_vendor
        self.id_product: int = id_product
        self.mic_list: list[Microphone] = mic_list
        self.sampling_rate: int = sampling_rate
        self.doa_estimator: DOAEstimator = doa_estimator
        self.beamformer: Beamformer = beamformer
        self.echo_canceller: EchoCanceller = echo_canceller
        self.filters: Sequence[Filter] = filters
        self.agc: AGC = agc
        self.codec: Codec = codec
        self.device_index: int | None = device_index

        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._latest_block: np.ndarray | None = None
        self._latest_per_mic: dict[int, np.ndarray] = {}
        self._latest_doa = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start_realtime(self, blocksize: int = 0):
        """ 
        Start real-time audio processing. This will open the audio stream and begin calling the callback function.
        The callback will store the latest audio block, per-microphone samples, and DOA estimates for retrieval.
        """
        if self._is_running:
            return
        if not self.mic_list:
            raise ValueError("Cannot start stream without microphones")

        self._stream = sd.InputStream(
            samplerate=self.sampling_rate,
            channels=len(self.mic_list),
            dtype='int16',
            device=self.device_index,
            callback=self._audio_callback,
            latency='low',
            blocksize=blocksize
        )
        self._stream.start()
        self._is_running = True

    def stop_realtime(self):
        if not self._is_running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._is_running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """
        This callback is called by the audio stream for each block of audio data. It processes the incoming audio,
        extracts per-microphone samples, estimates DOA, and stores the latest data for retrieval.
        """
        block = np.copy(indata)

        if status:
            print(f"[Array callback] {status}")

        per_mic = {}
        for idx, mic in enumerate(self.mic_list):
            channel_index = mic.channel_number if mic.channel_number is not None else idx
            if 0 <= channel_index < block.shape[1]:
                per_mic[mic.channel_number] = block[:, channel_index].copy()

        doa_value = None
        if callable(self.doa_estimator.estimate_doa):
            try:
                doa_value = self.doa_estimator.estimate_doa(block)
            except Exception:
                doa_value = None

        with self._lock:
            self._latest_block = block
            self._latest_per_mic = per_mic
            self._latest_doa = doa_value

    def get_latest_block(self) -> np.ndarray | None:
        """
        Get the latest audio block received from the stream. This will return a copy of the data to ensure thread safety.
        """
        
        with self._lock:
            if self._latest_block is None:
                return None
            return self._latest_block.copy()

    def get_latest_mic_samples(self, mic_channel_number: int) -> np.ndarray | None:
        """
        Get the latest audio samples for a specific microphone channel. This will return a copy of the data to ensure thread safety.
        """
        with self._lock:
            data = self._latest_per_mic.get(mic_channel_number)
            if data is None:
                return None
            return data.copy()

    def get_latest_doa(self):
        """
        Get the latest DOA estimate. This will return a copy of the data to ensure thread safety.
        """
        with self._lock:
            if self._latest_block is not None:
                self._latest_doa = self.doa_estimator.estimate_doa(self._latest_block)
                return self._latest_doa.copy() if self._latest_doa is not None else None
            else:
                return None

        
    
    
