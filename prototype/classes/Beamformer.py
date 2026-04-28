import numpy as np
import logging
import time

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
        self.last_process_time_ms = 0.0  # Track processing time per apply() call

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
        # Avoid redundant updates for near-equal angles to prevent unnecessary resets
        try:
            current = float(self._steering_angle_deg)
        except Exception:
            current = None

        new_angle = float(theta_deg)
        if current is not None and np.isclose(current, new_angle, atol=1e-3):
            return

        self._steering_angle_deg = new_angle
        # Log actual steering updates for easier debugging (rate-limited by callers)
        try:
            self.logger.debug(f"[Beamformer] Steering angle set to {self._steering_angle_deg:.2f}°")
        except Exception:
            pass

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
    
    def reset_weight_history(self):
        """
        Reset the weight smoothing history (if applicable for this beamformer).
        
        Used by DOAEstimator when DOA changes significantly (≥1°) to prevent
        lag-induced artifacts from steering/covariance/weight mismatch.
        
        Base implementation: no-op. Subclasses with weight smoothing override this.
        """
        pass    
    def reset_on_doa_change(self):
        """
        Reset both weight and covariance history for complete DOA change adaptation.
        
        Used by MVDRBeamformer to handle DOA changes by resetting both smoothing buffers.
        This prevents lag-induced artifacts from steering/covariance/weight mismatches.
        
        Base implementation: no-op. Only MVDR uses this.
        """
        pass

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
        beam_start = time.perf_counter()
        angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)
        result = self.process(block, angle)
        beam_time_ms = (time.perf_counter() - beam_start) * 1000.0
        self.last_process_time_ms = beam_time_ms
        result_arr = np.asarray(result)
        return result
    
    def apply_with_overlap_add_crossfade(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        """
        For DAS (stateless), this is identical to apply() since there is no internal filter state.
        DAS beamforming doesn't have covariance/weight smoothing, so steering changes don't cause discontinuities.
        """
        return self.apply(block, theta_deg)


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
        This is now a baseline; actual alpha adapts based on SNR for best directivity.
    :param max_adaptive_loading_scale: Upper bound for adaptive loading scale to avoid over-whitening.
    :param coherence_suppression_strength: Strength of diffuse noise suppression via coherence weighting (0–1).
        0.0 = no suppression (pure MVDR), 1.0 = maximum suppression (aggressive noise reduction).
        Default 0.5 balances directivity with speech preservation.
    :param backward_null_strength: Strength of spatial null constraint at 180° (back diffraction suppression).
        0.0 = disabled (pure MVDR with normal side-lobes at back).
        0.5 = moderate back suppression via MVDR null constraint (default).
        1.0 = aggressive null at 180°, strong back diffraction rejection.
        Recommendation: Use 0.5–0.8 to suppress back diffraction without over-constraining the beamformer.
        
        Implementation: Null-constrained MVDR that projects out any weight component pointing toward 180°
        while maintaining distortionless response at the steering angle. This is particularly effective
        at low frequencies where diffraction causes significant back lobes.
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
        coherence_suppression_strength: float = 0.5,
        weight_smooth_alpha_min: float = 0.45,
        weight_smooth_alpha_max: float = 0.82,
        snr_threshold_for_sharpening: float = 2.0,
        backward_null_strength: float = 0.5,
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
        if not 0.0 <= coherence_suppression_strength <= 1.0:
            raise ValueError("coherence_suppression_strength must be in [0, 1]")
        if not 0.0 <= weight_smooth_alpha_min < 1.0:
            raise ValueError("weight_smooth_alpha_min must be in [0, 1)")
        if not weight_smooth_alpha_min <= weight_smooth_alpha_max < 1.0:
            raise ValueError("weight_smooth_alpha_min must be <= weight_smooth_alpha_max < 1.0")
        if snr_threshold_for_sharpening <= 0:
            raise ValueError("snr_threshold_for_sharpening must be > 0")
        if not 0.0 <= backward_null_strength <= 1.0:
            raise ValueError("backward_null_strength must be in [0, 1]")

        self.covariance_alpha = float(covariance_alpha)
        self.diagonal_loading = float(diagonal_loading)
        self.spectral_whitening_factor = float(spectral_whitening_factor)
        self.weight_smooth_alpha_fixed = float(weight_smooth_alpha)  # Keep original for reference
        self.weight_smooth_alpha_min = float(weight_smooth_alpha_min)  # Sharp main lobe (steady state)
        self.weight_smooth_alpha_max = float(weight_smooth_alpha_max)  # Stable (noisy conditions)
        self.snr_threshold_for_sharpening = float(snr_threshold_for_sharpening)  # Transition point
        self.max_adaptive_loading_scale = float(max_adaptive_loading_scale)
        self.coherence_suppression_strength = float(coherence_suppression_strength)
        self.backward_null_strength = float(backward_null_strength)  # Null suppression at 180°
        self._covariance: np.ndarray | None = None
        self._prev_weights: np.ndarray | None = None
        self._power_iteration_vec: np.ndarray | None = None
        self._last_coherence: np.ndarray | None = None  # Store last computed coherence for external use (e.g., AGC)
        self._prev_steering_angle_deg: float | None = None  # Track steering angle for auto-reset on DOA changes
        self._identity = np.eye(self.channel_count, dtype=np.complex128)
        self._eps = 1e-12
        
        # Output-level crossfading for steering angle transitions (blend block outputs, not beamformer states)
        self._prev_output: np.ndarray | None = None  # Previous block's beamformed output for blending
        self._prev_steering_angle_for_blend: float | None = None  # Steering angle of previous block

    def reset(self):
        self._covariance = None
        self._prev_weights = None
        self._power_iteration_vec = None
        self._last_coherence = None
        self._prev_steering_angle_deg = None
        self._prev_output = None
        self._prev_steering_angle_for_blend = None
    
    # def reset_weight_history(self):
    #     """
    #     Reset the weight smoothing history to allow fresh weight computation.
        
    #     Used when DOA changes significantly (≥1°) to prevent lag-induced artifacts.
    #     Setting _prev_weights = None breaks the smoothing chain for exactly 1 block,
    #     allowing MVDR weights to recompute freely for the new steering direction
    #     without being constrained by old weights from a different DOA.
        
    #     This fixes the fundamental problem: MVDR is nonlinear in steering vector,
    #     and when DOA changes but old weights are applied to new steering matrix,
    #     the solution becomes mismatched → phase/amplitude discontinuities → artifacts.
    #     """
    #     self._prev_weights = None
    
    # def reset_on_doa_change(self):
    #     """
    #     Complete reset of weight and covariance history on significant DOA changes.
        
    #     Resets BOTH _prev_weights and _covariance to enable instant adaptation to new
    #     steering direction without lag-mismatch artifacts. This is stricter than 
    #     reset_weight_history() and should only be used when DOA changes significantly (≥1°).
        
    #     Rationale:
    #     - MVDR weights are nonlinear in both steering vector AND covariance matrix
    #     - Weight smoothing alpha=0.72-0.82 adapts in 2-3 blocks (worst: 0.6s @ 5Hz)
    #     - Covariance smoothing alpha=0.9 adapts in 10 blocks (worst: 2s @ 5Hz)
    #     - When DOA changes, both must adapt, not just weights
    #     - Resetting both allows instant recalculation in Block N+1
    #     - Covariance still smooths gradually from fresh spectrum (acceptable)
        
    #     Without this, new steering applied to old covariance creates discontinuities
    #     → output amplitude/phase swings at block boundaries → pops in audio
    #     """
    #     self._prev_weights = None
    #     self._covariance = None
    
    def get_last_coherence(self) -> np.ndarray | None:
        """
        Retrieve the coherence signal from the last process() call.
        Useful for passing to AGC for coherence-gated gain control.
        
        Returns:
            np.ndarray of shape (freq_bins,) with coherence values in [0, 1], or None if not computed.
        """
        return self._last_coherence

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
    
    def _compute_adaptive_weight_smooth_alpha(self, snr_ratio: float) -> float:
        """
        Compute adaptive weight smoothing factor based on SNR.
        
        Mechanism: During steady-state high-SNR conditions (clean speech), use lower smoothing
        (weight_smooth_alpha_min) for sharp main lobe directivity. During noisy low-SNR conditions,
        fall back to higher smoothing (weight_smooth_alpha_max) for stability.
        
        :param snr_ratio: SNR estimate from _compute_block_snr_estimate()
        :return: Adaptive alpha in range [weight_smooth_alpha_min, weight_smooth_alpha_max]
        """
        # Smooth interpolation between min (sharp) and max (stable) based on SNR relative to threshold
        # SNR 1.0 (low)  -> alpha_max (0.82, very smooth/stable)
        # SNR 2.0 (threshold) -> alpha_mid (0.63, balanced)
        # SNR 5.0+ (high) -> alpha_min (0.45, sharp directivity)
        
        # Clamp SNR to valid range for smooth interpolation
        snr_clipped = np.clip(snr_ratio, 0.5, 5.0)
        
        # Normalized position between threshold and high-SNR region: 0 (low) to 1 (high)
        # At snr_threshold (2.0): t = 0.0 -> use alpha_max
        # At snr_threshold * 2.5 (5.0): t = 1.0 -> use alpha_min
        t = np.clip(
            (snr_clipped - self.snr_threshold_for_sharpening) / 
            (2.5 * self.snr_threshold_for_sharpening - self.snr_threshold_for_sharpening),
            0.0, 1.0
        )
        
        # Interpolate: high t -> low alpha (sharp), low t -> high alpha (stable)
        # Use quadratic ease-out for smoother transition
        adaptive_alpha = self.weight_smooth_alpha_max - (self.weight_smooth_alpha_max - self.weight_smooth_alpha_min) * (t ** 1.5)
        
        return float(adaptive_alpha)
    
    def _compute_coherence_strength(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute inter-channel coherence as a measure of source coherence vs. diffuse noise.
        
        Coherence ρ between channels i,j at frequency f:
            ρ = |E[X_i * conj(X_j)]| / sqrt(E[|X_i|²] * E[|X_j|²])
        
        Returns: Average coherence per frequency bin, shape (freq_bins,) in range [0, 1]
            - 1.0 = perfectly coherent (point source)
            - 0.5 = moderately coherent
            - 0.0 = incoherent (diffuse noise)
        
        Used to suppress diffuse noise and room reflections while preserving main source.
        """
        freq_bins, num_mics = spectrum.shape
        
        # Compute magnitude spectrum for each channel
        mag_spectrum = np.abs(spectrum)  # Shape: (freq_bins, num_mics)
        
        # Compute pairwise coherence (vectorized)
        # Numerator: |cross-spectrum| averaged over all channel pairs
        # Denominator: geometric mean of power spectra
        coherence_sum = np.zeros(freq_bins, dtype=np.float64)
        pair_count = 0
        
        for i in range(num_mics):
            for j in range(i + 1, num_mics):
                # Cross-spectrum magnitude
                cross_spec = np.abs(
                    spectrum[:, i] * np.conj(spectrum[:, j])
                )
                
                # Power spectra (geometric mean)
                power_geom = np.sqrt(
                    (mag_spectrum[:, i] ** 2) * (mag_spectrum[:, j] ** 2) + self._eps
                )
                
                # Coherence for this pair
                coherence_pair = cross_spec / (power_geom + self._eps)
                coherence_sum += coherence_pair
                pair_count += 1
        
        # Average over all pairs
        avg_coherence = coherence_sum / max(pair_count, 1)
        
        # Clip to [0, 1] range
        return np.clip(avg_coherence, 0.0, 1.0)

    def _apply_backward_null_constraint(self, weights: np.ndarray, steering: np.ndarray, 
                                       theta_rad: float) -> np.ndarray:
        """
        Apply spatial null constraint at 180° (opposite direction) to suppress back diffraction.
        
        Implementation: Project weights orthogonal to the backward steering vector, forcing a null
        in the backward direction while maintaining distortionless response at the main steering angle.
        
        :param weights: MVDR weights of shape (freq_bins, num_mics)
        :param steering: Forward steering vector of shape (freq_bins, num_mics)
        :param theta_rad: Steering angle in radians
        :return: Backward null-constrained weights of shape (freq_bins, num_mics)
        """
        freq_bins = weights.shape[0]
        
        # Compute backward steering vector (180° opposite)
        backward_theta_rad = theta_rad + np.pi
        
        # Backward direction as a simple formula: negate the forward direction vector
        backward_steering = -steering  # Equivalent to steering at theta + 180°
        
        # For each frequency bin, project out the backward component
        # This forces the null constraint: minimize response at 180° while preserving response at theta
        weights_constrained = weights.copy()
        
        for f in range(freq_bins):
            w = weights[f, :]
            a_back = backward_steering[f, :]  # Backward steering vector at this frequency
            
            # Projection coefficient: how much does w point in the backward direction?
            numerator = np.abs(np.vdot(a_back, w)) ** 2
            denominator = np.abs(np.vdot(a_back, a_back)) + self._eps
            
            # Project out the backward component with strength control
            # strength=0: no projection (normal MVDR)
            # strength=1: full projection (maximum null)
            projection_coeff = (self.backward_null_strength * numerator) / denominator
            
            # Updated weights: remove backward component
            weights_constrained[f, :] = w - projection_coeff * a_back
        
        # Re-normalize to maintain distortionless response at steering angle
        # Recompute gain to preserve amplitude at steered direction
        denom = np.einsum("fi,fi->f", np.conj(steering), weights_constrained, optimize=True)
        valid = np.abs(denom) > self._eps
        weights_constrained[valid] = weights_constrained[valid] / denom[valid, None]
        weights_constrained[~valid] = 1.0 / self.channel_count
        
        return weights_constrained

    def process(self, block: np.ndarray, theta_deg: float) -> np.ndarray:
        selected = self._select_channels(np.asarray(block, dtype=np.float64))
        n_samples = selected.shape[0]
        
        if n_samples == 0:
            return np.array([], dtype=np.float64)

        # # Auto-detect DOA changes and reset weight AND covariance history to prevent lag-induced artifacts.
        # # When steering angle changes significantly (≥1°), reset both _prev_weights and _covariance
        # # to allow instant adaptation to new steering direction without lag mismatch.
        # if self._prev_steering_angle_deg is not None:
        #     doa_delta = abs(theta_deg - self._prev_steering_angle_deg)
        #     # Require strictly greater than 1.0° to avoid resets on exact 1.0° step transitions
        #     if doa_delta > 1.0:
        #         self.reset_on_doa_change()
        #         self.logger.debug(f"[MVDR] Auto-reset weights + covariance on DOA change: {self._prev_steering_angle_deg:.1f}° → {theta_deg:.1f}° (delta: {doa_delta:.2f}°)")
        # Store prev angle at reduced precision to avoid floating-point noise in comparisons
        self._prev_steering_angle_deg = float(round(theta_deg, 3))

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
        # Use adaptive smoothing factor to sharpen main lobe during steady-state high-SNR conditions.
        if self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            adaptive_alpha = self._compute_adaptive_weight_smooth_alpha(block_snr_ratio)
            weights = adaptive_alpha * self._prev_weights + (1.0 - adaptive_alpha) * weights

        # Re-enforce distortionless response after smoothing.
        smooth_denom = np.einsum("fi,fi->f", np.conj(steering), weights, optimize=True)
        valid_smooth = np.abs(smooth_denom) > self._eps
        weights[valid_smooth] = weights[valid_smooth] / smooth_denom[valid_smooth, None]
        weights[~valid_smooth] = 1.0 / self.channel_count
        self._prev_weights = weights.copy()

        # BACKWARD NULL CONSTRAINT: Suppress back diffraction at 180° (particularly effective at low frequencies).
        # This targets the physical problem of open-backed MEMS arrays where diffraction creates large back lobes.
        if self.backward_null_strength > self._eps:
            weights = self._apply_backward_null_constraint(weights, steering, theta_rad)

        # COHERENCE-BASED SIDELOBE SUPPRESSION: Suppress diffuse noise while preserving main source
        # Compute inter-channel coherence (0=diffuse, 1=coherent point-source)
        coherence = self._compute_coherence_strength(spectrum)  # Shape: (freq_bins,)
        
        # Apply user-controlled suppression strength to coherence gain
        # suppression_strength = 0.0: No suppression, pure MVDR
        # suppression_strength = 0.5 (default): Balanced suppression
        # suppression_strength = 1.0: Aggressive suppression
        # 
        # Coherence gain mapping with tunable strength:
        # - High coherence (0.95) -> gain ~0.975 (almost full MVDR)
        # - Medium coherence (0.5) -> gain ~0.75 (moderate suppression)
        # - Low coherence (0.1) -> gain ~0.55 at strength=0.5, or 0.1 at strength=1.0
        base_gain = 0.5 + 0.5 * coherence  # Base mapping: [0, 1] -> [0.5, 1.0]
        coherence_gain = base_gain ** (1.0 - 0.8 * self.coherence_suppression_strength)
        # ^ At strength=0.5: ^0.6 exponent, adds mid-range suppression
        # ^ At strength=1.0: ^0.2 exponent, very aggressive suppression
        
        # Apply coherence-based modulation to the output spectrum
        # This reduces energy from frequency bins with low inter-channel coherence (diffuse noise)
        output_spectrum = np.einsum("fi,fi->f", np.conj(weights), spectrum, optimize=True)
        output_spectrum *= coherence_gain  # Suppress low-coherence frequencies

        # Store coherence for external use (e.g., AGC coherence gating)
        self._last_coherence = coherence.copy()

        return np.fft.irfft(output_spectrum, n=n_samples).astype(np.float64, copy=False)

    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        beam_start = time.perf_counter()
        angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)
        result = self.process(block, angle)

        # Output-level boundary taper: apply a short fade near the block edges
        # without feeding previous output back into the next block.
        result_arr = np.asarray(result, dtype=np.float64)

        n_fade_samples = min(len(result_arr) // 8, 64)
        if n_fade_samples > 2:
            fade_in = np.linspace(0.0, 1.0, n_fade_samples, dtype=np.float64)
            fade_out = fade_in[::-1]

            output_arr = result_arr.copy()
            output_arr[:n_fade_samples] *= fade_in
            output_arr[-n_fade_samples:] *= fade_out
        else:
            output_arr = result_arr

        beam_time_ms = (time.perf_counter() - beam_start) * 1000.0
        self.last_process_time_ms = beam_time_ms
        return output_arr
    
    def apply_with_overlap_add_crossfade(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        """
        Alias for apply() - output-level crossfading is handled internally in apply().
        Kept for backwards compatibility with Array_RealTime.
        """
        return self.apply(block, theta_deg)
