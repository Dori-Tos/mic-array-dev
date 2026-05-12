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

    Optional stages can be enabled or disabled independently.
    
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
    :param enable_frequency_smoothing: Smooth covariance across nearby frequency bins before EMA update.
    :param frequency_smoothing_strength: Blend factor for frequency smoothing when enabled.
    :param enable_eigenvalue_suppression: Boost diagonal loading when the dominant eigenvalue dominates.
    :param enable_adaptive_loading: Enable SNR-dependent adaptive diagonal loading.
    :param enable_weight_smoothing: Enable temporal smoothing of MVDR weights.
    :param coherence_suppression_strength: Strength of diffuse noise suppression via coherence weighting (0–1).
        0.0 = no suppression (pure MVDR), 1.0 = maximum suppression (aggressive noise reduction).
        Default 0.5 balances directivity with speech preservation.
    :param enable_coherence_suppression: Enable coherence-based output modulation.
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
        covariance_alpha: float = 0.7,  # Increased smoothing (was 0.9): slower covariance adaptation = less block jitter
        diagonal_loading: float = 1e-3,
        spectral_whitening_factor: float = 0.3,
        weight_smooth_alpha: float = 0.88,  # Increased smoothing (was 0.82): more weight temporal stability
        max_adaptive_loading_scale: float = 8.0,
        enable_frequency_smoothing: bool = True,
        frequency_smoothing_strength: float = 0.3,
        enable_eigenvalue_suppression: bool = True,
        enable_adaptive_loading: bool = True,
        enable_weight_smoothing: bool = True,
        coherence_suppression_strength: float = 0.5,
        enable_coherence_suppression: bool = True,
        weight_smooth_alpha_min: float = 0.45,
        weight_smooth_alpha_max: float = 0.82,
        snr_threshold_for_sharpening: float = 2.0,
        enable_backward_null_constraint: bool = True,
        backward_null_strength: float = 0.5,
        enable_output_crossfade: bool = True,
        max_beamform_freq: float = 6000.0,
        # Crossfade tuning for handling frequent 1° step updates and DOA pauses
        crossfade_base_samples: int = 16,
        crossfade_min_samples: int = 8,
        crossfade_max_samples: int = 48,
        crossfade_ms: float = 10.0,
        crossfade_blocks: int = 4,
        crossfade_angle_threshold_deg: float = 1.0,
        crossfade_energy_threshold: float = 1e-5,
        crossfade_accumulation_ms: float = 200.0,
        crossfade_hold_ms: float = 150.0,
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
        self.weight_smooth_alpha = float(weight_smooth_alpha)  # Current smoothing alpha (can be adjusted per-block)
        self.weight_smooth_alpha_fixed = float(weight_smooth_alpha)  # Keep original for reference
        self.weight_smooth_alpha_min = float(weight_smooth_alpha_min)  # Sharp main lobe (steady state)
        self.weight_smooth_alpha_max = float(weight_smooth_alpha_max)  # Stable (noisy conditions)
        self.snr_threshold_for_sharpening = float(snr_threshold_for_sharpening)  # Transition point
        self.max_adaptive_loading_scale = float(max_adaptive_loading_scale)
        self.enable_frequency_smoothing = bool(enable_frequency_smoothing)
        self.frequency_smoothing_strength = float(frequency_smoothing_strength)
        self.enable_eigenvalue_suppression = bool(enable_eigenvalue_suppression)
        self.enable_adaptive_loading = bool(enable_adaptive_loading)
        self.enable_weight_smoothing = bool(enable_weight_smoothing)
        self.coherence_suppression_strength = float(coherence_suppression_strength)
        self.enable_coherence_suppression = bool(enable_coherence_suppression)
        self.backward_null_strength = float(backward_null_strength)  # Null suppression at 180°
        self.enable_backward_null_constraint = bool(enable_backward_null_constraint)
        self.enable_output_crossfade = bool(enable_output_crossfade)
        # If <= 0 or None, masked beamforming is disabled (process all bins)
        if max_beamform_freq is None:
            self.max_beamform_freq = None
        else:
            max_b = float(max_beamform_freq)
            if max_b <= 0.0:
                self.max_beamform_freq = None
            else:
                # Clip to Nyquist for safety
                self.max_beamform_freq = min(max_b, float(self.sample_rate) / 2.0)
        self._covariance: np.ndarray | None = None
        self._prev_weights: np.ndarray | None = None
        self._power_iteration_vec: np.ndarray | None = None
        self._last_coherence: np.ndarray | None = None  # Store last computed coherence for external use (e.g., AGC)
        self._prev_steering_angle_deg: float | None = None  # Track steering angle for auto-reset on DOA changes
        self._last_dominant_eigenvalue: float = 1.0  # Dominant eigenvalue from power iteration (for eigenvalue suppression)
        self._last_avg_eigenvalue: float = 1.0  # Average eigenvalue (for eigenvalue suppression)
        self._identity = np.eye(self.channel_count, dtype=np.complex128)
        self._eps = 1e-12
        
        # Output-level crossfading for steering angle transitions (blend block outputs, not beamformer states)
        self._prev_output: np.ndarray | None = None  # Previous block's beamformed output for blending
        self._prev_steering_angle_for_blend: float | None = None  # Steering angle of previous block
        
        # Output-domain IIR smoothing to catch block-boundary discontinuities
        self._iir_state: float = 0.0  # Previous sample for one-pole IIR filter
        self._output_iir_alpha: float = 0.7  # IIR filter pole; higher = more smoothing, lower = more responsiveness
        
        # Cross-block covariance temporal smoothing to reduce sudden weight jumps
        self._prev_covariance: np.ndarray | None = None  # Store covariance from previous block
        self._covariance_cross_block_alpha: float = 0.3  # Blend new covariance with previous; 0=full new, 1=full old
        # Crossfade tuning
        self.crossfade_base_samples = int(crossfade_base_samples)
        self.crossfade_min_samples = int(crossfade_min_samples)
        self.crossfade_max_samples = int(crossfade_max_samples)
        self.crossfade_ms = float(crossfade_ms)
        self.crossfade_blocks = int(crossfade_blocks)
        self.crossfade_angle_threshold_deg = float(crossfade_angle_threshold_deg)
        self.crossfade_energy_threshold = float(crossfade_energy_threshold)
        self.crossfade_accumulation_ms = float(crossfade_accumulation_ms)
        self.crossfade_hold_ms = float(crossfade_hold_ms)
        # Transition tuning for large-angle changes
        self.transition_blocks_per_degree: float = 1.5  # blocks per degree change (0.5 -> 10° -> 5 blocks)
        self.transition_blocks_max: int = 16
        self.transition_loading_scale: float = 4.0  # multiply diagonal loading during transition
        self.transition_weight_smooth_alpha: float = float(self.weight_smooth_alpha_max)
        # Internal crossfade state for accumulating rapid small steps
        self._blend_accumulated_angle: float = 0.0
        self._last_angle_update_time: float | None = None
        # Transition state: ramp steering angle itself over several blocks.
        self._transition_start_angle: float | None = None
        self._transition_target_angle: float | None = None
        self._transition_blocks_total: int = 0
        self._transition_blocks_left: int = 0
        # If True, delay starting the steering transition until we have a previous
        # weight history to smooth against. This prevents a first-block abrupt
        # weight jump when _prev_weights is None.
        self._defer_transition_until_prev_weights: bool = False
        self._transition_log_interval = 1.0
        self._last_transition_log_time = 0.0

    def reset(self):
        self._covariance = None
        self._prev_weights = None
        self._power_iteration_vec = None
        self._last_coherence = None
        self._prev_steering_angle_deg = None
        self._prev_output = None
        self._prev_steering_angle_for_blend = None
        self._last_dominant_eigenvalue = 1.0
        self._last_avg_eigenvalue = 1.0
        # Reset crossfade accumulation state
        self._blend_accumulated_angle = 0.0
        self._last_angle_update_time = None
        self._transition_start_angle = None
        self._transition_target_angle = None
        self._transition_blocks_total = 0
        self._transition_blocks_left = 0
        self._in_transition: bool = False
        self._in_transition = False
    
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
        
    def _smooth_covariance_across_frequencies(self, r_inst: np.ndarray, freq_bins: int) -> np.ndarray:
        """
        Smooth covariance matrix across nearby frequency bins to reduce noise jitter.
        
        Mechanism: Low frequencies are noisier (fewer samples per bin at low freq due to frequency resolution).
        Smooth aggressively at low frequencies, gently at high frequencies to preserve directivity.
        
        Kernel width is frequency-dependent:
        - Low freq (100 Hz): kernel_width ≈ 5, strong smoothing
        - Mid freq (1000 Hz): kernel_width ≈ 3
        - High freq (4000 Hz): kernel_width ≈ 1-2, minimal smoothing
        
        :param r_inst: Instantaneous covariance of shape (freq_bins, num_mics, num_mics)
        :param freq_bins: Number of frequency bins
        :return: Frequency-smoothed covariance (same shape)
        """
        # Estimate frequency in Hz for each bin (assuming 48 kHz, n_fft from bin count)
        # freq_bins is rfft output, so n_fft ≈ 2 * (freq_bins - 1)
        n_fft_est = 2 * (freq_bins - 1)
        freq_hz = np.fft.rfftfreq(n_fft_est, d=1.0 / self.sample_rate)
        
        # Adaptive kernel width: wider at low freq, narrower at high freq
        # kernel_width(f) = max(1, int(5000 / (f + 50)))
        # At f=100 Hz: 5000/150 ≈ 33 bins (clip to ~5)
        # At f=1000 Hz: 5000/1050 ≈ 5 bins
        # At f=4000 Hz: 5000/4050 ≈ 1 bin
        kernel_widths = np.maximum(1, (5000.0 / (freq_hz + 100.0)).astype(int))
        kernel_widths = np.minimum(kernel_widths, 5)  # Cap at 5 to avoid over-smoothing
        
        smoothed = r_inst.copy()
        smoothing_strength = 0.3  # Light smoothing to preserve directivity
        
        for f in range(freq_bins):
            kernel_width = kernel_widths[f]
            if kernel_width <= 1:
                continue  # No smoothing needed at high frequencies
            
            # Create triangular window for smoothing kernel
            half_width = kernel_width // 2
            kernel = np.array([kernel_width - abs(i - half_width) for i in range(kernel_width)], dtype=np.float64)
            kernel = kernel / np.sum(kernel)  # Normalize
            
            # Apply kernel across nearby frequency bins
            smooth_r = np.zeros_like(r_inst[f])
            weight_sum = 0.0
            
            for offset, k_weight in enumerate(kernel):
                bin_idx = f - half_width + offset
                if 0 <= bin_idx < freq_bins:
                    smooth_r += k_weight * r_inst[bin_idx]
                    weight_sum += k_weight
            
            if weight_sum > 1e-10:
                smooth_r = smooth_r / weight_sum
                # Blend: preserve original to maintain directivity, but incorporate smoothed version
                smoothed[f] = (1.0 - smoothing_strength) * r_inst[f] + smoothing_strength * smooth_r
        
        return smoothed

    def _compute_block_snr_estimate(self, spectrum: np.ndarray) -> float:
        """
        Compute SNR estimate for the block using power iteration on average covariance.
        Runs once per block (not per-bin) for efficient noise estimation.
        
        Also stores dominant_eigenvalue and avg_eigenvalue for eigenvalue suppression use.
        """
        freq_bins = spectrum.shape[0]
        avg_covariance = np.einsum("fi,fj->ij", spectrum, np.conj(spectrum), optimize=True)
        avg_covariance /= max(freq_bins, 1)
        
        try:
            dominant_eigenvalue, _ = self._power_iteration(avg_covariance, num_iterations=2)
        except (np.linalg.LinAlgError, ValueError):
            dominant_eigenvalue = 1.0
        
        trace_r = float(np.trace(avg_covariance).real)
        avg_eigenvalue = trace_r / self.channel_count if trace_r > self._eps else self._eps
        
        # Store for eigenvalue suppression in process()
        self._last_dominant_eigenvalue = max(dominant_eigenvalue, self._eps)
        self._last_avg_eigenvalue = max(avg_eigenvalue, self._eps)
        
        snr_ratio = self._last_dominant_eigenvalue / (self._last_avg_eigenvalue + self._eps)
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

        try:
            freq_hz = np.fft.rfftfreq(n_samples, d=1.0 / self.sample_rate)
        except Exception:
            freq_hz = np.fft.rfftfreq(2 * (freq_bins - 1), d=1.0 / self.sample_rate)

        if self.max_beamform_freq is not None and 0.0 < self.max_beamform_freq < (self.sample_rate / 2.0):
            process_mask = freq_hz <= float(self.max_beamform_freq)
            if process_mask.shape[0] != freq_bins:
                process_mask = np.resize(process_mask, freq_bins)
        else:
            process_mask = np.ones(freq_bins, dtype=bool)

        # Start with a zero-cost pass-through for skipped bins. This avoids any beamforming
        # work above the cutoff and lets downstream bandpass filtering remove those bins.
        output_spectrum = spectrum[:, 0].copy()

        if not np.any(process_mask):
            return np.fft.irfft(output_spectrum, n=n_samples).astype(np.float64, copy=False)

        # Compute SNR estimate only when a stage actually needs it.
        need_snr_estimate = (
            self.enable_adaptive_loading
            or self.enable_weight_smoothing
            or self.enable_eigenvalue_suppression
        )
        spectrum_proc = spectrum[process_mask]
        steering_proc = steering[process_mask]
        block_snr_ratio = self._compute_block_snr_estimate(spectrum_proc) if need_snr_estimate else 1.0

        # Update covariance only for processed bins.
        r_inst_proc = spectrum_proc[:, :, None] * np.conj(spectrum_proc[:, None, :])
        
        # FREQUENCY-SMOOTHING: Reduce noise jitter by smoothing covariance across nearby bins.
        # Low frequencies benefit from aggressive smoothing; high frequencies use minimal smoothing.
        # Keep this optional because it can smear transient detail in some rooms.
        if self.enable_frequency_smoothing:
            r_inst_proc = self._smooth_covariance_across_frequencies(r_inst_proc, int(r_inst_proc.shape[0]))
        
        self._covariance[process_mask] = (
            self.covariance_alpha * self._covariance[process_mask]
            + (1.0 - self.covariance_alpha) * r_inst_proc
        )
        
        # CROSS-BLOCK COVARIANCE SMOOTHING: Reduce sudden weight jumps at block boundaries.
        # Blend current covariance with previous block's covariance to smooth out discontinuities
        # caused by per-block FFT independence. This adds latency (~1 block) but stabilizes MVDR weights.
        if self._prev_covariance is not None and self._prev_covariance.shape == self._covariance.shape:
            self._covariance = (
                (1.0 - self._covariance_cross_block_alpha) * self._covariance
                + self._covariance_cross_block_alpha * self._prev_covariance
            )
        # Store current covariance for next block
        self._prev_covariance = self._covariance.copy()

        # Robust, frequency-dependent loading with angle-dependent stabilization.
        # At edge angles (±25°), increase diagonal loading to handle DOA errors and instability.
        angle_loading_multiplier = self._compute_angle_dependent_diagonal_loading(theta_deg)
        adjusted_diagonal_loading = self.diagonal_loading * angle_loading_multiplier
        
        trace_r = np.real(np.trace(self._covariance[process_mask], axis1=1, axis2=2))
        base_loading = adjusted_diagonal_loading * (trace_r / self.channel_count + self._eps)
        if self.enable_adaptive_loading:
            inv_snr = np.clip(1.0 / (block_snr_ratio + self._eps), 0.0, self.max_adaptive_loading_scale)
            adaptive_loading = adjusted_diagonal_loading * (1.0 + self.spectral_whitening_factor * inv_snr)
        else:
            adaptive_loading = 0.0
        total_loading = base_loading + adaptive_loading
        # If a steering transition is in progress, soften weights by increasing loading
        if getattr(self, "_in_transition", False):
            total_loading = total_loading * float(self.transition_loading_scale)
        
        # EIGENVALUE-BASED NOISE SUBSPACE SUPPRESSION: Boost loading when noise eigenvalues dominate.
        # When dominant_eigenvalue >> avg_eigenvalue, signal is strong but noise floor is raised.
        # Suppress noise eigenvalues by boosting diagonal loading.
        # Suppression ratio ranges from 1.0 (no suppression) to 3.0 (aggressive).
        if self.enable_eigenvalue_suppression:
            eigenvalue_suppression_ratio = np.clip(
                self._last_dominant_eigenvalue / (self._last_avg_eigenvalue + self._eps),
                1.0, 3.0
            )
            total_loading = total_loading * eigenvalue_suppression_ratio
        
        r_loaded_proc = self._covariance[process_mask] + total_loading[:, None, None] * self._identity[None, :, :]

        try:
            r_inv_a_sub = np.linalg.solve(r_loaded_proc, steering_proc[:, :, None])[..., 0]
        except (np.linalg.LinAlgError, ValueError):
            r_inv_a_sub = np.einsum("fij,fj->fi", np.linalg.pinv(r_loaded_proc), steering_proc, optimize=True)

        denom_sub = np.einsum("fi,fi->f", np.conj(steering_proc), r_inv_a_sub, optimize=True)
        weights_sub = np.empty_like(r_inv_a_sub)
        valid_sub = np.abs(denom_sub) > self._eps
        weights_sub[valid_sub] = r_inv_a_sub[valid_sub] / denom_sub[valid_sub, None]
        weights_sub[~valid_sub] = 1.0 / self.channel_count

        weights = np.zeros_like(steering)
        weights[process_mask] = weights_sub

        # Temporal smoothing suppresses rapid weight swings that cause popping with side/rear interferers.
        # Use adaptive smoothing factor to sharpen main lobe during steady-state high-SNR conditions.
        if self.enable_weight_smoothing and self._prev_weights is not None and self._prev_weights.shape == weights.shape:
            adaptive_alpha = self._compute_adaptive_weight_smooth_alpha(block_snr_ratio)
            # During transitions, ensure stronger temporal smoothing to avoid abrupt weight jumps
            if getattr(self, "_in_transition", False):
                adaptive_alpha = max(adaptive_alpha, float(self.transition_weight_smooth_alpha))
            weights = adaptive_alpha * self._prev_weights + (1.0 - adaptive_alpha) * weights

        # Re-enforce distortionless response after smoothing.
        smooth_denom = np.einsum("fi,fi->f", np.conj(steering_proc), weights[process_mask], optimize=True)
        valid_smooth = np.abs(smooth_denom) > self._eps
        weights_proc = weights[process_mask]
        weights_proc[valid_smooth] = weights_proc[valid_smooth] / smooth_denom[valid_smooth, None]
        weights_proc[~valid_smooth] = 1.0 / self.channel_count
        weights[process_mask] = weights_proc
        if self.enable_weight_smoothing:
            self._prev_weights = weights.copy()
        else:
            self._prev_weights = None

        # BACKWARD NULL CONSTRAINT: Suppress back diffraction at 180° (particularly effective at low frequencies).
        # This targets the physical problem of open-backed MEMS arrays where diffraction creates large back lobes.
        if self.enable_backward_null_constraint and self.backward_null_strength > self._eps:
            weights_proc = self._apply_backward_null_constraint(weights[process_mask], steering_proc, theta_rad)
            weights[process_mask] = weights_proc

        # COHERENCE-BASED SIDELOBE SUPPRESSION: Suppress diffuse noise while preserving main source
        # Compute inter-channel coherence (0=diffuse, 1=coherent point-source)
        output_spectrum[process_mask] = np.einsum("fi,fi->f", np.conj(weights[process_mask]), spectrum_proc, optimize=True)
        if self.enable_coherence_suppression:
            coherence = self._compute_coherence_strength(spectrum_proc)  # Shape: (processed_bins,)
            
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
            output_spectrum[process_mask] *= coherence_gain  # Suppress low-coherence frequencies
            self._last_coherence = np.ones(freq_bins, dtype=np.float64)
            self._last_coherence[process_mask] = coherence.copy()
        else:
            self._last_coherence = np.ones(freq_bins, dtype=np.float64)

        # Compute final time-domain output
        output = np.fft.irfft(output_spectrum, n=n_samples).astype(np.float64, copy=False)
        
        # OUTPUT-DOMAIN IIR SMOOTHING: One-pole low-pass filter on final output to catch
        # block-to-block discontinuities that weight smoothing misses. This is phase-neutral
        # and won't cause roboticism like windowed FFT overlap-add.
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        # where _iir_state holds y[n-1] from the previous sample
        filtered_output = np.empty_like(output)
        for i in range(n_samples):
            filtered_output[i] = self._output_iir_alpha * output[i] + (1.0 - self._output_iir_alpha) * self._iir_state
            self._iir_state = filtered_output[i]
        
        return filtered_output

    def _compute_angle_dependent_alphas(self, theta_deg: float) -> tuple[float, float, float, float]:
        """
        Compute angle-dependent smoothing parameters to reduce side-angle artifacts.
        
        At front (0°): keep the current tuned smoothing
        At edges (±25°): increase smoothing to stabilize block-to-block weight changes
        Linearly interpolate between based on absolute steering angle.
        
        :param theta_deg: Steering angle in degrees
        :return: Tuple (cov_alpha, alpha_min, alpha_max, alpha_center)
        """
        abs_angle = np.abs(float(theta_deg))
        
        # Covariance alpha parameters
        front_cov_alpha = 0.75
        edge_cov_alpha = 0.6
        
        # Weight smoothing center (higher at edges for stronger temporal smoothing)
        front_alpha_center = self.weight_smooth_alpha_fixed
        edge_alpha_center = 0.68
        
        # Normalize angle to [0, 1]: t=0 at 0°, t=1 at ±25°
        t = np.clip(abs_angle / 25.0, 0.0, 1.0)
        
        # Linear interpolation for covariance alpha
        adj_cov_alpha = front_cov_alpha + (edge_cov_alpha - front_cov_alpha) * t
        
        # Linear interpolation for weight smoothing center
        adj_alpha_center = front_alpha_center + (edge_alpha_center - front_alpha_center) * t
        
        # Adjust min/max range around the center
        range_width = self.weight_smooth_alpha_max - self.weight_smooth_alpha_min
        adj_alpha_min = np.clip(adj_alpha_center - range_width * 0.5, 0.0, 1.0)
        adj_alpha_max = np.clip(adj_alpha_center + range_width * 0.5, 0.0, 1.0)
        
        return float(adj_cov_alpha), float(adj_alpha_min), float(adj_alpha_max), float(adj_alpha_center)

    def _compute_angle_dependent_diagonal_loading(self, theta_deg: float) -> float:
        """
        Compute angle-dependent diagonal loading multiplier to stabilize MVDR at edge angles.
        
        Higher loading at edge angles (±25°) makes the covariance matrix inversion more robust
        when steering is uncertain. At front (0°), use normal loading. At edges (±25°),
        apply strong regularization to dampen oscillations from DOA estimation errors.
        
        :param theta_deg: Steering angle in degrees
        :return: Multiplier to scale diagonal_loading (1.0 at front, 5-8x at edges)
        """
        abs_angle = np.abs(float(theta_deg))
        
        # Front (0°): normal loading multiplier
        front_multiplier = 1.0
        # Edges (±25°): strong regularization to handle DOA errors and instability
        edge_multiplier = 12.0
        
        # Normalize angle to [0, 1]: t=0 at 0°, t=1 at ±25°
        t = np.clip(abs_angle / 25.0, 0.0, 1.0)
        
        # Linear interpolation for smooth transition
        multiplier = front_multiplier + (edge_multiplier - front_multiplier) * t
        
        return float(multiplier)

    def apply(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        current_log_time = time.monotonic()
        beam_start = time.perf_counter()
        requested_angle = self.get_steering_angle() if theta_deg is None else float(theta_deg)

        # Smooth the beamforming change by ramping the steering angle itself across
        # a small number of blocks. This avoids attenuation during fixed-angle moments.
        effective_angle = requested_angle
        previous_angle = self._prev_steering_angle_for_blend

        # Require a meaningful angle delta before starting a transition to avoid
        # float-noise-triggered transitions. Use configured hysteresis threshold.
        angle_changed = False
        if previous_angle is not None:
            delta = abs(float(requested_angle) - float(previous_angle))
            angle_changed = delta >= float(self.crossfade_angle_threshold_deg)
        else:
            # No previous angle recorded -> treat as no change (startup)
            angle_changed = False

        if self.enable_output_crossfade:
            if angle_changed:
                # If we don't have previous weight history yet, defer starting the
                # transition until after process() has had a chance to set
                # _prev_weights, otherwise the very first transition block can
                # introduce an abrupt, unsmoothed weight jump (audible pop).
                if self._prev_weights is None:
                    self._defer_transition_until_prev_weights = True
                    if hasattr(self, "logger"):
                        self.logger.debug(
                            f"[MVDR] Deferring transition (no prev weights) {previous_angle!r}→{requested_angle!r}"
                        )
                else:
                    self._transition_start_angle = float(previous_angle)
                    self._transition_target_angle = float(requested_angle)
                    # Compute transition length proportional to angle delta
                    delta = abs(float(requested_angle) - float(previous_angle))
                    calc_blocks = max(1, int(np.ceil(delta * self.transition_blocks_per_degree)))
                    total_blocks = min(self.transition_blocks_max, max(calc_blocks, int(self.crossfade_blocks)))
                    self._transition_blocks_total = int(total_blocks)
                    self._transition_blocks_left = int(self._transition_blocks_total)
                    self._in_transition = True
                    if current_log_time - self._last_transition_log_time > self._transition_log_interval:
                        self._last_transition_log_time = current_log_time
                        if hasattr(self, "logger"):
                            self.logger.debug(f"[MVDR] Starting transition: delta={delta:.2f}°, blocks={self._transition_blocks_total}")

            if (
                self._transition_blocks_left > 0
                and self._transition_start_angle is not None
                and self._transition_target_angle is not None
            ):
                # Use blocks_done starting at 0 so the first transition block starts
                # at fraction==0 and eases towards 1. This avoids an immediate
                # jump at the first transition block.
                blocks_done = self._transition_blocks_total - self._transition_blocks_left
                fraction = min(max(blocks_done / float(self._transition_blocks_total), 0.0), 1.0)
                # Cosine easing keeps the steering move gentle without muting the whole block.
                eased = 0.5 * (1.0 - np.cos(np.pi * fraction))
                effective_angle = self._transition_start_angle + (
                    self._transition_target_angle - self._transition_start_angle
                ) * eased
                # Advance the transition after computing this block's effective angle
                self._transition_blocks_left -= 1

                if self._transition_blocks_left <= 0:
                    self._transition_start_angle = None
                    self._transition_target_angle = None
                    self._transition_blocks_total = 0
                    self._in_transition = False

        # Apply angle-dependent smoothing to reduce side-angle artifacts
        adj_cov_alpha, adj_alpha_min, adj_alpha_max, adj_alpha_center = self._compute_angle_dependent_alphas(effective_angle)
        
        # Save original parameters
        orig_cov_alpha = self.covariance_alpha
        orig_alpha_min = self.weight_smooth_alpha_min
        orig_alpha_max = self.weight_smooth_alpha_max
        
        # Temporarily apply adjusted parameters for this block
        self.covariance_alpha = adj_cov_alpha
        self.weight_smooth_alpha_min = adj_alpha_min
        self.weight_smooth_alpha_max = adj_alpha_max
        
        # Process with adjusted smoothing
        result = self.process(block, effective_angle)
        
        # Restore original parameters for next block
        self.covariance_alpha = orig_cov_alpha
        self.weight_smooth_alpha_min = orig_alpha_min
        self.weight_smooth_alpha_max = orig_alpha_max
        output_arr = np.asarray(result, dtype=np.float64)
        # Keep the last requested angle so the next block can detect changes.
        self._blend_accumulated_angle = 0.0
        self._last_angle_update_time = None

        try:
            self._prev_output = output_arr.copy()
            # Store the effective angle actually used for the block. Using
            # `effective_angle` (not `requested_angle`) ensures future
            # transition detection compares against the last-steered angle
            # rather than the desired target, preventing spurious transitions
            # and mismatches between steering and blend bookkeeping.
            self._prev_steering_angle_for_blend = float(effective_angle)
        except Exception:
            self._prev_output = None
            self._prev_steering_angle_for_blend = None

        # If we deferred the transition until we had previous weights, and
        # process() has now set _prev_weights, start the transition for the
        # next block.
        if getattr(self, "_defer_transition_until_prev_weights", False) and self._prev_weights is not None:
            self._defer_transition_until_prev_weights = False
            # Initialize transition using the stored previous steering angle
            if previous_angle is not None:
                delta = abs(float(requested_angle) - float(previous_angle))
                calc_blocks = max(1, int(np.ceil(delta * self.transition_blocks_per_degree)))
                total_blocks = min(self.transition_blocks_max, max(calc_blocks, int(self.crossfade_blocks)))
                self._transition_start_angle = float(previous_angle)
                self._transition_target_angle = float(requested_angle)
                self._transition_blocks_total = int(total_blocks)
                self._transition_blocks_left = int(self._transition_blocks_total)
                self._in_transition = True
                if hasattr(self, "logger"):
                    self.logger.debug(f"[MVDR] Deferred transition now starting: delta={delta:.2f}°, blocks={self._transition_blocks_total}")

        # Update timing metric
        beam_time_ms = (time.perf_counter() - beam_start) * 1000.0
        self.last_process_time_ms = beam_time_ms
        return output_arr
    
    def apply_with_overlap_add_crossfade(self, block: np.ndarray, theta_deg: float | None = None) -> np.ndarray:
        """
        Alias for apply(). Output continuity should be handled by the caller
        (for example Array_RealTime's chunk join path) when enabled.
        Kept for backwards compatibility with Array_RealTime.
        """
        return self.apply(block, theta_deg)
