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
import time
import logging

import sounddevice as sd


class Array:
    """
    Class representing a microphone array system that captures audio, estimates DOA, applies beamforming,
    performs echo cancellation, filtering, AGC, and encoding.

    :param id_vendor: USB vendor ID for the microphone array device.
    :param id_product: USB product ID for the microphone array device.
    :param mic_list: List of Microphone instances representing the individual microphones in the array.
    :param sampling_rate: The sampling rate for audio capture and processing.
    :param doa_estimator: An instance of a DOAEstimator subclass for estimating direction of arrival.
    :param beamformer: An instance of a Beamformer subclass for applying spatial filtering.
    :param echo_canceller: An instance of EchoCanceller for removing echo from the beamformed signal.
    :param filters: A sequence of Filter instances to apply to the beamformed signal before AGC.
    :param agc: An instance of AGC for automatic gain control on the processed signal.
    :param codec: An instance of Codec for encoding the final output audio.
    :param device_index: Optional index of the audio input device to use. If None, the default device is used.
    """
    
    
    def __init__(self, logger: logging.Logger,
                 id_vendor, id_product, mic_list: list[Microphone], sampling_rate: int,
                 doa_estimator: DOAEstimator, beamformer: Beamformer, 
                 echo_canceller: EchoCanceller, filters: Sequence[Filter], agc: AGC, codec: Codec,
                 device_index: int | None = None):
        
        self.logger: logging.Logger = logger
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

    def get_latest_mic_samples(self, mic_channel_number: int) -> np.ndarray | None:
        """
        Get the latest audio samples for a specific microphone channel. This will return a copy of the data to ensure thread safety.
        """
        with self._lock:
            data = self._latest_per_mic.get(mic_channel_number)
            if data is None:
                return None
            return data.copy()
            
    def test_all_microphones(self, duration_seconds: float = 2.0, poll_interval: float = 0.05):
        """
        Measure RMS for each microphone over a fixed time window.

        Args:
            duration_seconds: Measurement duration in seconds.
            poll_interval: Time between polling the latest microphone buffers.

        Returns:
            Dictionary keyed by microphone channel with RMS summary.
        """
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        mic_buffers: dict[int, list[np.ndarray]] = {
            mic.channel_number: [] for mic in self.mic_list
        }

        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            with self._lock:
                snapshot = {
                    mic.channel_number: self._latest_per_mic.get(mic.channel_number)
                    for mic in self.mic_list
                }

            for channel_number, samples in snapshot.items():
                if samples is not None and len(samples) > 0:
                    mic_buffers[channel_number].append(samples.astype(np.float64, copy=False))

            time.sleep(poll_interval)

        full_scale = 32768.0
        min_level = 1e-10
        results = {}

        for mic in self.mic_list:
            channel_number = mic.channel_number
            chunks = mic_buffers[channel_number]

            if not chunks:
                results[channel_number] = {
                    'rms': None,
                    'rms_dbfs': None,
                    'num_samples': 0,
                    'duration_seconds': duration_seconds
                }
                print(f"Mic {channel_number}: No data captured in {duration_seconds:.2f}s")
                continue

            all_samples = np.concatenate(chunks)
            rms = float(np.sqrt(np.mean(all_samples ** 2)))
            rms_dbfs = 20 * np.log10(max(rms, min_level) / full_scale)

            results[channel_number] = {
                'rms': rms,
                'rms_dbfs': float(rms_dbfs),
                'num_samples': int(all_samples.size),
                'duration_seconds': duration_seconds
            }

            print(
                f"Mic {channel_number}: RMS={rms:.2f} | "
                f"RMS={rms_dbfs:.2f} dBFS | "
                f"Samples={all_samples.size}"
            )

        return results
    
    def rec_audio_microphone(self, mic_channel_number: int, duration_seconds: float = 2.0) -> np.ndarray | None:
        """
        Capture raw audio samples from a specific microphone channel for a fixed duration.

        Args:
            mic_channel_number: The channel number of the microphone to capture from.
            duration_seconds: Duration to capture audio in seconds.
        
        Returns:
            A numpy array of captured audio samples, or None if no data was captured.
        """
        
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")

        samples_list = []
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            with self._lock:
                samples = self._latest_per_mic.get(mic_channel_number)

            if samples is not None and len(samples) > 0:
                samples_list.append(samples.astype(np.float64, copy=False))

            time.sleep(0.05)

        if not samples_list:
            print(f"Mic {mic_channel_number}: No data captured in {duration_seconds:.2f}s")
            return None

        all_samples = np.concatenate(samples_list)
        print(f"Mic {mic_channel_number}: Captured {all_samples.size} samples in {duration_seconds:.2f}s")
        return all_samples

    def test_beamformer(
        self,
        duration_seconds: float = 2.0,
        poll_interval: float = 0.05,
        theta_deg: float | None = None,
        reference_channel: int | None = None,
    ):
        """
        Test beamformer output over a fixed time window.

        Args:
            duration_seconds: Measurement duration in seconds.
            poll_interval: Time between polling the latest audio block.
            theta_deg: Steering angle in degrees. If None, uses latest DOA when available,
                otherwise uses the beamformer's current steering angle.
            reference_channel: Channel number used as baseline for comparison.
                If None, defaults to the first beamformer channel.

        Returns:
            Dictionary with RMS/dBFS for beamformed and reference signals.
        """
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        if reference_channel is None:
            reference_channel = int(self.beamformer.mic_channel_numbers[0])

        beamformed_chunks: list[np.ndarray] = []
        reference_chunks: list[np.ndarray] = []
        angles_used: list[float] = []

        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            with self._lock:
                block = None if self._latest_block is None else self._latest_block.copy()
                doa_snapshot = self._latest_doa

            if block is not None and block.ndim == 2 and block.shape[0] > 0:
                if theta_deg is not None:
                    angle = float(theta_deg)
                elif isinstance(doa_snapshot, (int, float, np.integer, np.floating)):
                    angle = float(doa_snapshot)
                else:
                    angle = float(self.beamformer.get_steering_angle())

                try:
                    beamformed = self.beamformer.apply(block, angle)
                except Exception:
                    beamformed = np.array([], dtype=np.float64)

                if beamformed.size > 0:
                    beamformed_chunks.append(beamformed.astype(np.float64, copy=False))
                    angles_used.append(angle)

                if block.shape[1] > reference_channel:
                    reference = block[:, reference_channel]
                    reference_chunks.append(reference.astype(np.float64, copy=False))

            time.sleep(poll_interval)

        if not beamformed_chunks or not reference_chunks:
            print(f"Beamformer test: No data captured in {duration_seconds:.2f}s")
            return {
                "beamformed": {
                    "rms": None,
                    "rms_dbfs": None,
                    "num_samples": 0,
                },
                "reference": {
                    "channel": int(reference_channel),
                    "rms": None,
                    "rms_dbfs": None,
                    "num_samples": 0,
                },
                "improvement_db": None,
                "avg_angle_deg": None,
                "duration_seconds": duration_seconds,
            }

        beamformed_all = np.concatenate(beamformed_chunks)
        reference_all = np.concatenate(reference_chunks)

        full_scale = 32768.0
        min_level = 1e-10

        beamformed_rms = float(np.sqrt(np.mean(beamformed_all ** 2)))
        beamformed_dbfs = float(20 * np.log10(max(beamformed_rms, min_level) / full_scale))

        reference_rms = float(np.sqrt(np.mean(reference_all ** 2)))
        reference_dbfs = float(20 * np.log10(max(reference_rms, min_level) / full_scale))

        improvement_db = float(beamformed_dbfs - reference_dbfs)
        avg_angle_deg = float(np.mean(angles_used)) if angles_used else None

        print(
            f"Beamformer: RMS={beamformed_rms:.2f} | RMS={beamformed_dbfs:.2f} dBFS | "
            f"Ref ch {reference_channel}: RMS={reference_rms:.2f} | RMS={reference_dbfs:.2f} dBFS | "
            f"Delta={improvement_db:.2f} dB"
        )

        return {
            "beamformed": {
                "rms": beamformed_rms,
                "rms_dbfs": beamformed_dbfs,
                "num_samples": int(beamformed_all.size),
            },
            "reference": {
                "channel": int(reference_channel),
                "rms": reference_rms,
                "rms_dbfs": reference_dbfs,
                "num_samples": int(reference_all.size),
            },
            "improvement_db": improvement_db,
            "avg_angle_deg": avg_angle_deg,
            "duration_seconds": duration_seconds,
        }
        

        
    
    
