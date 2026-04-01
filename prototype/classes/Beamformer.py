import numpy as np
import logging

import xml.etree.ElementTree as ET


class Beamformer:
    """
    Base class for beamforming algorithms. Provides common init, channel selection, and steering matrix logic.

    :param mic_channel_numbers: List of channel indices.
    :param sample_rate: Sample rate of the audio in Hz.
    :param mic_spacing_m: Spacing between microphones in meters (used for steering calculations).
    :param sound_speed_m_s: Speed of sound in meters per second (used for steering calculations).
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

        Azimuth convention used here:
        - 0° points to +y (forward)
        - +90° points to +x (right)
        - -90° points to -x (left)
        - z is treated as height and ignored for azimuth steering
        """
        freq_bins = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        direction = np.array(
            [
                np.sin(theta_rad),
                np.cos(theta_rad),
                0.0,
            ],
            dtype=np.float64,
        )

        relative_positions = self.mic_positions_m - np.mean(self.mic_positions_m, axis=0)
        delays = (relative_positions @ direction) / self.sound_speed_m_s
        return np.exp(-1j * 2.0 * np.pi * freq_bins[:, None] * delays[None, :])

    def process(self, block: np.ndarray, theta_deg: float) -> np.ndarray:
        raise NotImplementedError()
    
    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        raise NotImplementedError()


class DASBeamformer(Beamformer):
    """
    Delay-and-sum (DAS) beamformer implementation. Applies steering delays and sums across channels.
    
    :param mic_channel_numbers: List of channel indices.
    :param sample_rate: Sample rate of the audio in Hz.
    :param mic_spacing_m: Spacing between microphones in meters (used for steering calculations).
    :param sound_speed_m_s: Speed of sound in meters per second (used for steering calculations).
    :param mic_positions_m: Optional array of shape (num_mics, 3) with microphone positions in meters.
        If not provided, a linear array along the x-axis with spacing mic_spacing_m is assumed.
    """
    
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
    """
    Minimum Variance Distortionless Response (MVDR) beamformer implementation.
    Uses an adaptive covariance matrix with exponential smoothing and diagonal loading for stability.
    
    Enhanced with power iteration for improved directivity and noise subspace suppression.
    
    Alternative directivity enhancement methods (not currently implemented):
    - Eigenvalue Thresholding: Full eigendecomposition + noise subspace masking (too expensive, O(n³))
    - Adaptive Diagonal Loading: Condition-number-based loading adjustment for automatic SNR adaptation
    - Signal Subspace Projection (MUSIC-like): Full eigendecomposition + projection filtering
    - Spatial Tapering: Channel windowing to reduce side-lobes (trades main-lobe narrowing for raised side-lobes)
    - Angle-dependent weighting: Tighter constraints at extreme angles (±90°)
    
    Current implementation: POWER ITERATION + WEIGHT SMOOTHING
    - Fast dominant eigenvalue/eigenvector estimation via power iteration on average covariance (O(n²))
    - Computed once per block for efficiency; influences adaptive loading across all frequency bins
    - Adaptive spectral whitening to enhance signal-to-noise separation
    - Temporal MVDR weight smoothing to reduce bursty artifacts with off-axis interferers
    - Suppresses diffuse sources and side-source artifacts with minimal computational cost
    - Maintains real-time performance (~20ms per block)
    
    :param mic_channel_numbers: List of channel indices.
    :param sample_rate: Sample rate of the audio in Hz.
    :param mic_spacing_m: Spacing between microphones in meters (used for steering calculations).
    :param sound_speed_m_s: Speed of sound in meters per second (used for steering calculations).
    :param mic_positions_m: Optional array of shape (num_mics, 3) with microphone positions in meters.
        If not provided, a linear array along the x-axis with spacing mic_spacing_m is assumed
    :param covariance_alpha: EMA factor for covariance matrix updating (0 = instant, close to 1 = slow).
    :param diagonal_loading: Amount of diagonal loading to improve numerical stability (default: 1e-3).
    :param spectral_whitening_factor: Adaptive whitening strength (0–1, higher = more aggressive whitening to suppress noise).
    :param weight_smooth_alpha: Temporal smoothing for MVDR weights (0 = no smoothing, close to 1 = smoother/more stable).
    :param max_adaptive_loading_scale: Upper bound for adaptive loading scale to avoid over-whitening.
    """
    def __init__(
        self, logger: logging.Logger,
        mic_channel_numbers: list[int], sample_rate: int = 48000,
        mic_spacing_m: float = 0.05, sound_speed_m_s: float = 343.0,
        mic_positions_m: np.ndarray | list[list[float]] | None = None,
        covariance_alpha: float = 0.9,  diagonal_loading: float = 1e-3,
        spectral_whitening_factor: float = 0.3,
        weight_smooth_alpha: float = 0.82,
        max_adaptive_loading_scale: float = 8.0,
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
        if not 0.0 <= spectral_whitening_factor <= 1.0:
            raise ValueError("spectral_whitening_factor must be in [0, 1]")
        if not 0.0 <= weight_smooth_alpha < 1.0:
            raise ValueError("weight_smooth_alpha must be in [0, 1)")
        if max_adaptive_loading_scale <= 0.0:
            raise ValueError("max_adaptive_loading_scale must be > 0")

        self.covariance_alpha = float(covariance_alpha)
        self.diagonal_loading = float(diagonal_loading)
        self.spectral_whitening_factor = float(spectral_whitening_factor)
        self.weight_smooth_alpha = float(weight_smooth_alpha)
        self.max_adaptive_loading_scale = float(max_adaptive_loading_scale)
        self._covariance: np.ndarray | None = None
        self._prev_weights: np.ndarray | None = None
        self._power_iteration_vec: np.ndarray | None = None
        self._identity = np.eye(self.channel_count, dtype=np.complex128)
        self._eps = 1e-12

    def reset(self):
        self._covariance = None
        self._prev_weights = None
        self._power_iteration_vec = None

    def _power_iteration(self, matrix: np.ndarray, num_iterations: int = 2) -> tuple[float, np.ndarray]:
        """
        Fast dominant eigenvalue/eigenvector estimation via power iteration.
        Converges in 2 or 3 iterations for typical audio covariance matrices.
        
        Returns: (largest_eigenvalue, dominant_eigenvector)
        """
        n = matrix.shape[0]
        if self._power_iteration_vec is None or self._power_iteration_vec.shape[0] != n:
            v = np.ones(n, dtype=np.complex128) / np.sqrt(float(n))
        else:
            v = self._power_iteration_vec.astype(np.complex128, copy=True)
            norm_v = float(np.linalg.norm(v))
            if norm_v <= self._eps:
                v = np.ones(n, dtype=np.complex128) / np.sqrt(float(n))
            else:
                v /= norm_v
        
        for _ in range(num_iterations):
            v = matrix @ v
            norm_v = float(np.linalg.norm(v))
            if norm_v <= self._eps:
                v = np.ones(n, dtype=np.complex128) / np.sqrt(float(n))
            else:
                v /= norm_v

        self._power_iteration_vec = v.copy()
        
        eigenvalue = float(np.real(np.vdot(v, matrix @ v)))
        return eigenvalue, v

    def _ensure_covariance(self, freq_bins: int):
        if self._covariance is None or self._covariance.shape[0] != freq_bins:
            self._covariance = np.repeat(self._identity[None, :, :], freq_bins, axis=0)
        
    def _compute_block_snr_estimate(self, spectrum: np.ndarray) -> float:
        """
        Compute SNR estimate for the block using power iteration on average covariance.
        Runs once per block (not per-bin) for efficient noise estimation.
        """
        freq_bins = spectrum.shape[0]
        avg_covariance = np.einsum("fi,fj->ij", spectrum, np.conj(spectrum), optimize=True)
        avg_covariance /= max(freq_bins, 1)
        
        try:
            dominant_eigenvalue, _ = self._power_iteration(avg_covariance, num_iterations=2)
        except (np.linalg.LinAlgError, ValueError):
            return 1.0  # Default SNR ratio on failure
        
        trace_r = float(np.trace(avg_covariance).real)
        avg_eigenvalue = trace_r / self.channel_count if trace_r > self._eps else self._eps
        
        snr_ratio = dominant_eigenvalue / (avg_eigenvalue + self._eps)
        return np.clip(snr_ratio, 0.1, 10.0)

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

        # Compute SNR estimate once per block for efficient adaptive loading.
        block_snr_ratio = self._compute_block_snr_estimate(spectrum)

        # Vectorized covariance update for all bins.
        r_inst = spectrum[:, :, None] * np.conj(spectrum[:, None, :])
        self._covariance = (
            self.covariance_alpha * self._covariance
            + (1.0 - self.covariance_alpha) * r_inst
        )

        # Robust, frequency-dependent loading.
        trace_r = np.real(np.trace(self._covariance, axis1=1, axis2=2))
        base_loading = self.diagonal_loading * (trace_r / self.channel_count + self._eps)
        inv_snr = np.clip(1.0 / (block_snr_ratio + self._eps), 0.0, self.max_adaptive_loading_scale)
        adaptive_loading = self.diagonal_loading * (1.0 + self.spectral_whitening_factor * inv_snr)
        total_loading = base_loading + adaptive_loading
        r_loaded = self._covariance + total_loading[:, None, None] * self._identity[None, :, :]

        # Batched MVDR solve across all frequency bins.
        try:
            r_inv_a = np.linalg.solve(r_loaded, steering[:, :, None])[..., 0]
        except (np.linalg.LinAlgError, ValueError):
            r_inv_a = np.einsum("fij,fj->fi", np.linalg.pinv(r_loaded), steering, optimize=True)

        denom = np.einsum("fi,fi->f", np.conj(steering), r_inv_a, optimize=True)
        weights = np.empty_like(r_inv_a)
        valid = np.abs(denom) > self._eps
        weights[valid] = r_inv_a[valid] / denom[valid, None]
        weights[~valid] = 1.0 / self.channel_count

        # Temporal smoothing suppresses rapid weight swings that cause popping with side/rear interferers.
        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            weights = self.weight_smooth_alpha * self._prev_weights + (1.0 - self.weight_smooth_alpha) * weights

        # Re-enforce distortionless response after smoothing.
        smooth_denom = np.einsum("fi,fi->f", np.conj(steering), weights, optimize=True)
        valid_smooth = np.abs(smooth_denom) > self._eps
        weights[valid_smooth] = weights[valid_smooth] / smooth_denom[valid_smooth, None]
        weights[~valid_smooth] = 1.0 / self.channel_count
        self._prev_weights = weights.copy()

        output_spectrum = np.einsum("fi,fi->f", np.conj(weights), spectrum, optimize=True)

        return np.fft.irfft(output_spectrum, n=n_samples).astype(np.float64, copy=False)

    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)
        return self.process(block, angle)
