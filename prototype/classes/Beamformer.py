import numpy as np
import logging

import xml.etree.ElementTree as ET


class Beamformer:
    """
    Base class for beamforming algorithms. Provides common init, channel selection, and steering matrix logic.

    Args:
    - mic_channel_numbers: List of channel indices.
    - sample_rate: Sample rate of the audio in Hz.
    - mic_spacing_m: Spacing between microphones in meters (used for steering calculations).
    - sound_speed_m_s: Speed of sound in meters per second (used for steering calculations).
    """
    def __init__(
        self, logger: logging.Logger,
        mic_channel_numbers: list[int], sample_rate: int = 48000,
        mic_spacing_m: float = 0.05, sound_speed_m_s: float = 343.0,
        mic_positions_m: np.ndarray | list[list[float]] | None = None,
    ):
        
        self.logger: logging.Logger = logger
        
        if len(mic_channel_numbers) < 2:
            raise ValueError("At least 2 microphones are required for beamforming")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if mic_spacing_m <= 0:
            raise ValueError("mic_spacing_m must be > 0")
        if sound_speed_m_s <= 0:
            raise ValueError("sound_speed_m_s must be > 0")

        self.mic_channel_numbers = mic_channel_numbers
        self.channel_count = len(mic_channel_numbers)
        self.sample_rate = int(sample_rate)
        self.mic_spacing_m = float(mic_spacing_m)
        self.sound_speed_m_s = float(sound_speed_m_s)

        if mic_positions_m is None:
            positions = np.zeros((self.channel_count, 3), dtype=np.float64)
            positions[:, 0] = np.arange(self.channel_count, dtype=np.float64) * self.mic_spacing_m
        else:
            positions = np.asarray(mic_positions_m, dtype=np.float64)
            if positions.shape != (self.channel_count, 3):
                raise ValueError(
                    f"mic_positions_m must have shape ({self.channel_count}, 3), got {positions.shape}"
                )
        self.mic_positions_m = positions
        self._steering_angle_deg = 0.0

    @staticmethod
    def load_positions_from_xml(xml_path: str) -> np.ndarray:
        """
        Load microphone positions from Acoular-style XML.

        Expected format:
            <MicArray>
              <pos x="..." y="..." z="..."/>
              ...
            </MicArray>

        Returns:
            np.ndarray with shape (num_mics, 3), columns [x, y, z] in meters.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        positions: list[list[float]] = []

        for node in root.findall(".//pos"):
            x_attr = node.attrib.get("x")
            y_attr = node.attrib.get("y")
            z_attr = node.attrib.get("z")
            if x_attr is None or y_attr is None or z_attr is None:
                continue
            positions.append([float(x_attr), float(y_attr), float(z_attr)])

        if not positions:
            raise ValueError(f"No <pos x=... y=... z=...> entries found in XML file: {xml_path}")

        return np.asarray(positions, dtype=np.float64)

    def set_steering_angle(self, theta_deg: float):
        self._steering_angle_deg = float(theta_deg)

    def get_steering_angle(self) -> float:
        return self._steering_angle_deg

    def _select_channels(self, block: np.ndarray) -> np.ndarray:
        if block.ndim != 2:
            raise ValueError("Expected block shape (samples, channels)")
        if block.shape[1] == self.channel_count:
            return block

        max_channel = max(self.mic_channel_numbers)
        if block.shape[1] <= max_channel:
            raise ValueError(
                f"Input block has {block.shape[1]} channels but requires at least {max_channel + 1}"
            )
        return block[:, self.mic_channel_numbers]

    def _steering_matrix(self, theta_rad: float, n_fft: int) -> np.ndarray:
        """
        Returns a steering matrix of shape (freq_bins, num_mics).
        """
        freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        direction = np.array(
            [
                np.sin(theta_rad),
                0.0,
                np.cos(theta_rad),
            ],
            dtype=np.float64,
        )

        relative_positions = self.mic_positions_m - self.mic_positions_m[0]
        delays = (relative_positions @ direction) / self.sound_speed_m_s
        return np.exp(-1j * 2.0 * np.pi * freq_bins[:, None] * delays[None, :])

    def process(self, block: np.ndarray, theta_deg: float) -> np.ndarray:
        raise NotImplementedError()
    
    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        raise NotImplementedError()


class DASBeamformer(Beamformer):
    def __init__(
        self, logger: logging.Logger,
        mic_channel_numbers: list[int], sample_rate: int = 48000,
        mic_spacing_m: float = 0.05, sound_speed_m_s: float = 343.0,
        mic_positions_m: np.ndarray | list[list[float]] | None = None,
    ):
        super().__init__(
            logger=logger,
            mic_channel_numbers=mic_channel_numbers,
            sample_rate=sample_rate,
            mic_spacing_m=mic_spacing_m,
            sound_speed_m_s=sound_speed_m_s,
            mic_positions_m=mic_positions_m,
        )

    def process(self, block: np.ndarray, theta_deg: float) -> np.ndarray:
        # Optimize: Avoid copy if block is already float64
        selected = self._select_channels(np.asarray(block, dtype=np.float64))
        n_samples = selected.shape[0]
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        theta_rad = np.deg2rad(theta_deg)
        steering = self._steering_matrix(theta_rad=theta_rad, n_fft=n_samples)

        # Compute FFT once
        spectrum = np.fft.rfft(selected, axis=0)
        
        # Apply steering weights and sum across channels
        weights = np.conj(steering) / self.channel_count
        output_spectrum = np.sum(spectrum * weights, axis=1)

        # OPTIMIZE: Use casting=False to avoid unnecessary conversions
        return np.fft.irfft(output_spectrum, n=n_samples)

    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)
        return self.process(block, angle)


class MVDRBeamformer(Beamformer):
    def __init__(
        self, logger: logging.Logger,
        mic_channel_numbers: list[int], sample_rate: int = 48000,
        mic_spacing_m: float = 0.05, sound_speed_m_s: float = 343.0,
        mic_positions_m: np.ndarray | list[list[float]] | None = None,
        covariance_alpha: float = 0.9,  diagonal_loading: float = 1e-3,
    ):
        super().__init__(
            logger=logger,
            mic_channel_numbers=mic_channel_numbers,
            sample_rate=sample_rate,
            mic_spacing_m=mic_spacing_m,
            sound_speed_m_s=sound_speed_m_s,
            mic_positions_m=mic_positions_m,
        )
        if not 0.0 <= covariance_alpha < 1.0:
            raise ValueError("covariance_alpha must be in [0, 1)")
        if diagonal_loading <= 0:
            raise ValueError("diagonal_loading must be > 0")

        self.covariance_alpha = float(covariance_alpha)
        self.diagonal_loading = float(diagonal_loading)
        self._covariance: np.ndarray | None = None

    def reset(self):
        self._covariance = None

    def _ensure_covariance(self, freq_bins: int):
        if self._covariance is None or self._covariance.shape[0] != freq_bins:
            identity = np.eye(self.channel_count, dtype=np.complex128)
            self._covariance = np.repeat(identity[None, :, :], freq_bins, axis=0)

    def process(self, block: np.ndarray, theta_deg: float) -> np.ndarray:
        selected = self._select_channels(np.asarray(block, dtype=np.float64))
        n_samples = selected.shape[0]
        
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        theta_rad = np.deg2rad(theta_deg)
        steering = self._steering_matrix(theta_rad=theta_rad, n_fft=n_samples)
        spectrum = np.fft.rfft(selected, axis=0).astype(np.complex128, copy=False)

        freq_bins = spectrum.shape[0]
        self._ensure_covariance(freq_bins)
        assert self._covariance is not None

        output_spectrum = np.zeros(freq_bins, dtype=np.complex128)
        identity = np.eye(self.channel_count, dtype=np.complex128)

        for i in range(freq_bins):
            x = spectrum[i, :].reshape(-1, 1)
            r_inst = x @ x.conj().T
            self._covariance[i] = (
                self.covariance_alpha * self._covariance[i]
                + (1.0 - self.covariance_alpha) * r_inst
            )

            trace_r = np.trace(self._covariance[i]).real
            loading = self.diagonal_loading * (trace_r / self.channel_count + 1e-12)
            r_loaded = self._covariance[i] + loading * identity

            a = steering[i, :].reshape(-1, 1)
            r_inv_a = np.linalg.pinv(r_loaded) @ a
            denom = (a.conj().T @ r_inv_a).item()
            if np.abs(denom) < 1e-12:
                w = np.ones((self.channel_count, 1), dtype=np.complex128) / self.channel_count
            else:
                w = r_inv_a / denom

            output_spectrum[i] = (w.conj().T @ x).item()

        return np.fft.irfft(output_spectrum, n=n_samples).astype(np.float64, copy=False)

    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)
        return self.process(block, angle)
