from .Beamformer import Beamformer
import numpy as np
import time
import logging

class DOAEstimator:
    """
    Base class for Direction of Arrival (DOA) estimation. Subclasses implement specific DOA estimation algorithms.
    
    :param logger: logging.Logger instance for logging messages.
    :param update_rate: Maximum rate (in Hz) at which the DOA estimate is updated (default: 3.0). 
        Higher values allow more frequent updates but increase computational load.
    :param angle_range: Tuple specifying the minimum and maximum angles (in degrees) to consider
        for DOA estimation (default: (-25, 25)). The estimator will only search for sources within this angular range.
        
    """
    
    def __init__(self, logger: logging.Logger,
                 update_rate: float = 3.0, angle_range: tuple = (-25, 25)):
        
        if update_rate <= 0:
            raise ValueError("update_rate must be > 0")
        if angle_range[0] > angle_range[1]:
            raise ValueError("angle_range must be (min_angle, max_angle)")
        
        self.logger: logging.Logger = logger
        self.frozen: bool = False
        self.latest_doa: float | None = None
        self.update_rate = update_rate
        self.angle_range = angle_range
    
    def freeze(self, angle_deg: float | None = None):
        """
        Freeze the DOA estimator to prevent further updates.
        """
        
        self.frozen = True
        if angle_deg is not None:
            self.latest_doa = angle_deg
    
    def unfreeze(self):
        """
        Unfreeze the DOA estimator to allow updates.
        """
        self.frozen = False
        
    def estimate_doa(self, audio_block):
        """
        Estimate the direction of arrival (DOA) from the given audio block.
        This is a placeholder method and should be overridden by subclasses with actual DOA estimation logic.
        """
        raise NotImplementedError("estimate_doa method must be implemented by subclasses")
    
    @property
    def is_frozen(self):
        return self.frozen
    
    
class IterativeDOAEstimator(DOAEstimator):
    """
    Iterative DOA estimator that performs a local search around the current DOA estimate to find the angle with the highest beamformed gain.
    
    :param logger: logging.Logger instance for logging messages.
    :param update_rate: Maximum rate (in Hz) at which the DOA estimate is updated (default: 3.0).
    :param angle_range: Tuple specifying the minimum and maximum angles (in degrees) to consider for DOA estimation (default: (-25, 25)).
    :param doa_beamformer: Beamformer instance to use for computing the beamformed output at different angles. It is different from the main beamformer used for processing, allowing for faster DOA scanning (e.g., using a simple DAS beamformer).
    :param beamformer: Main beamformer instance used for processing. This is needed to reset its weight history when the DOA changes significantly, preventing lag-induced artifacts.
    :param scan_step_deg: Step size in degrees for scanning neighboring angles during the local search
        (default: 1.0). Smaller steps may yield more accurate DOA estimates but increase computational load.
    :param normalize_channels: If True, normalize each channel's RMS before DOA estimation to reduce bias from mic gain mismatch (default: True).
    :param bootstrap_full_scan: If True, perform a full scan of the angle range on the first update to initialize the DOA estimate (default: True). This ensures the algorithm starts at the global best angle instead of 0°.
    :param periodic_full_scan_blocks: Number of update iterations between full rescans (default: 60). Set to 0 to disable periodic scans. Periodic full scans help escape local maxima and explore the full angle range.
    :param local_search_radius_deg: Radius in degrees around current DOA to search during local hill-climbing (default: 3.0). Smaller values (3-5°) help avoid wild jumps; larger values cover more range per iteration.
    :param min_update_rms: Minimum RMS energy required in the audio block to perform a DOA update (default: 0.0005). This prevents updates during silence or very low energy, which can cause noisy DOA estimates.
    :param min_confidence_db: Minimum confidence in dB required to accept a new DOA estimate (default: 1.5). Confidence is computed as the gain difference between the best and second-best angles. This prevents updates when the DOA estimate is ambiguous, reducing jitter.
    :param min_gain_improvement_db: Minimum improvement in dB required to accept a new angle during local search (default: 0.5 dB). Prevents oscillation between equally-good neighbors; requires meaningful gain change to prevent thrashing.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        update_rate: float = 3.0,
        angle_range: tuple = (-25, 25),
        doa_beamformer: Beamformer | None = None,
        beamformer: Beamformer | None = None,
        scan_step_deg: float = 1.0,
        smooth_step_deg: float = 1.5,
        normalize_channels: bool = True,
        bootstrap_full_scan: bool = True,
        periodic_full_scan_blocks: int = 60,
        local_search_radius_deg: float = 3.0,
        min_update_rms: float = 0.0005,
        min_confidence_db: float = 1.5,
        min_gain_improvement_db: float = 0.5,
    ):
        super().__init__(logger=logger, update_rate=update_rate, angle_range=angle_range)

        if doa_beamformer is None:
            raise ValueError("IterativeDOAEstimator requires a Beamformer instance")
        if beamformer is None:
            raise ValueError("IterativeDOAEstimator requires a main Beamformer instance")
        if scan_step_deg <= 0:
            raise ValueError("scan_step_deg must be > 0")
        if local_search_radius_deg <= 0:
            raise ValueError("local_search_radius_deg must be > 0")
        if min_update_rms <= 0:
            raise ValueError("min_update_rms must be > 0")
        if min_confidence_db < 0:
            raise ValueError("min_confidence_db must be >= 0")

        self.doa_beamformer = doa_beamformer
        self.beamformer = beamformer
        self.scan_step_deg = float(scan_step_deg)
        self.smooth_step_deg = float(smooth_step_deg)
        self.normalize_channels = bool(normalize_channels)
        self.bootstrap_full_scan = bool(bootstrap_full_scan)
        self.periodic_full_scan_blocks = int(periodic_full_scan_blocks)
        self.local_search_radius_deg = float(local_search_radius_deg)
        self.min_update_rms = float(min_update_rms)
        self.min_confidence_db = float(min_confidence_db)
        self.min_gain_improvement_db = float(min_gain_improvement_db)
        self._last_update_time = 0.0
        self._latest_gain = None
        self.latest_confidence_db: float | None = None
        self._update_count = 0  # Track updates for periodic full scan
        self.last_process_time_ms: float = 0.0  # Actual computation time in ms (not including early returns)
        # Low-energy threshold for skipping DOA updates to avoid noise during pauses
        self._consecutive_low_energy_frames = 0  # Track how many consecutive low-energy frames
        # DOA smoothing: smooth DOA transitions to prevent sharp steering vector changes
        # Target-based interpolation: move towards new DOA by at most 1° per block
        self._smoothed_doa = None  # Current smoothed DOA sent to beamformer (updates every block)
        self._target_doa = None    # Target DOA from latest estimation
        self._doa_step_per_block = 1.0  # Max 1° per block to reach new estimates smoothly
        # Logging rate-limiting to avoid log flooding from per-block stepping
        self._last_stepping_log_time = 0.0
        self._stepping_log_interval = 0.25  # seconds between stepping logs
        # Adaptive stepping: scale per-block step based on confidence (0.5x to 2.0x)
        self._last_confidence_db = None  # Track confidence for adaptive stepping logic

    def reset(self):
        self._last_update_time = 0.0
        self._latest_gain = None
        self.latest_doa = None
        self.latest_confidence_db = None
        self._update_count = 0
        self.last_process_time_ms = 0.0
        self._consecutive_low_energy_frames = 0
        self._smoothed_doa = None
        self._target_doa = None
        self._last_confidence_db = None

    def _compute_gain(self, block: np.ndarray, angle_deg: float) -> float:
        beamformed = self.doa_beamformer.apply(block, theta_deg=float(angle_deg))
        beamformed_arr = np.asarray(beamformed, dtype=np.float64)
        return float(np.mean(beamformed_arr * beamformed_arr))

    def _compute_gains_vectorized(self, block: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        """
        Compute beamformer gains for multiple angles in frequency domain (vectorized).
        
        This is ~10-20x faster than calling apply() for each angle because:
        1. FFT computed once for entire block (not per-angle)
        2. All steering matrices computed vectorized
        3. Gains computed directly in frequency domain (no IFFT needed)
        4. Takes advantage of numpy broadcasting
        
        :param block: Input audio block (samples, channels)
        :param angles_deg: Array of angles to evaluate in degrees
        :return: Array of gains corresponding to each angle
        """
        # Prepare block
        selected = self.doa_beamformer._select_channels(np.asarray(block, dtype=np.float64))
        n_samples = selected.shape[0]
        
        if n_samples == 0 or len(angles_deg) == 0:
            return np.array([])
        
        # Single FFT for the entire block (not per-angle!)
        spectrum = np.fft.rfft(selected, axis=0)  # Shape: (freq_bins, channels)
        freq_bins = spectrum.shape[0]
        
        gains = np.zeros(len(angles_deg), dtype=np.float64)
        
        # Compute steering and gains for all angles
        for angle_idx, angle_deg in enumerate(angles_deg):
            theta_rad = np.deg2rad(angle_deg)
            # Reuse beamformer's steering matrix computation
            steering = self.doa_beamformer._steering_matrix(theta_rad=theta_rad, n_fft=n_samples)  # (freq_bins, channels)
            
            # Apply steering (complex conjugate) and sum across channels: shape (freq_bins,)
            weights = np.conj(steering) / self.doa_beamformer.channel_count
            output_spectrum = np.sum(spectrum * weights, axis=1)  # (freq_bins,)
            
            # Compute gain directly from frequency domain (no IFFT needed!)
            # Power = mean of |output|^2 across frequency bins
            gain = float(np.mean(np.abs(output_spectrum) ** 2))
            gains[angle_idx] = gain
        
        return gains

    def _initial_full_scan(self, block: np.ndarray) -> tuple[float, float] | None:
        min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
        scan_angles = np.arange(min_angle, max_angle + 0.5 * self.scan_step_deg, self.scan_step_deg)
        if scan_angles.size == 0:
            return None

        if min_angle <= 0.0 <= max_angle and not np.any(np.isclose(scan_angles, 0.0)):
            scan_angles = np.append(scan_angles, 0.0)

        angles = np.array(sorted(scan_angles.tolist(), key=lambda a: (abs(a), a)), dtype=np.float64)

        # Use vectorized computation for all angles at once (10-20x faster than per-angle calls)
        gains = self._compute_gains_vectorized(block, angles)
        
        if gains.size == 0:
            return None
        
        best_idx = int(np.argmax(gains))
        best_angle = float(angles[best_idx])
        best_gain = float(gains[best_idx])
        
        # Find second-best for confidence
        sorted_idx = np.argsort(gains)[::-1]
        second_best_gain = float(gains[sorted_idx[1]]) if len(sorted_idx) > 1 else -np.inf
        
        for idx, gain in enumerate(gains):
            self.logger.debug(f"Angle {angles[idx]:6.1f}° -> gain {gain:.6e}")

        if not np.isfinite(best_gain):
            return None

        # Confidence: peakiness of the best-vs-second-best beamformed power.
        eps = 1e-20
        if np.isfinite(second_best_gain) and second_best_gain > 0.0:
            self.latest_confidence_db = float(10.0 * np.log10((best_gain + eps) / (second_best_gain + eps)))
        else:
            self.latest_confidence_db = None
        return best_angle, best_gain

    def _normalize_block_channels(self, block: np.ndarray) -> np.ndarray:
        """Normalize each channel RMS to reduce DOA bias from mic gain mismatch."""
        arr = np.asarray(block, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return arr

        eps = 1e-12
        rms = np.sqrt(np.mean(arr * arr, axis=0) + eps)
        return arr / rms[None, :]

    def _compute_confidence_db(self, gains: np.ndarray) -> float | None:
        finite_gains = np.asarray(gains, dtype=np.float64)
        finite_gains = finite_gains[np.isfinite(finite_gains)]
        if finite_gains.size < 2:
            return None

        sorted_gains = np.sort(finite_gains)[::-1]
        best = float(sorted_gains[0])
        second = float(sorted_gains[1])
        if second <= 0.0:
            return None
        eps = 1e-20
        return float(10.0 * np.log10((best + eps) / (second + eps)))
    
    def _get_adaptive_step_rate(self) -> float:
        """
        Scale the per-block stepping rate based on confidence.
        
        - Low confidence (< 1 dB): 0.75x (slower, more cautious)
        - Medium confidence (1-3 dB): 1.0x (normal)
        - High confidence (> 3 dB): 1.5x (faster tracking)
        
        This allows quick responses when DOA is clear, but slower when ambiguous.
        """
        if self._last_confidence_db is None:
            return 1.0  # Normal rate
        
        conf = self._last_confidence_db
        if conf < 1.0:
            return 0.75  # Low confidence: step slower
        elif conf > 3.0:
            return 1.5   # High confidence: step faster
        else:
            # Linear interpolation between 0.75 and 1.5 for confidence 1-3 dB
            return 0.75 + (conf - 1.0) * (1.5 - 0.75) / 2.0
    
    
    def _reset_beamformer_on_doa_change(self, new_doa: float):
        """
        Disabled: Aggressive resets on DOA changes cause artifacts through discontinuities.
        
        Let the beamformer handle DOA tracking changes through natural alpha smoothing
        of weights and covariance. The smoothing already adapts gradually to steering changes.
        """
        pass

    def _update_smoothed_doa(self):
        """
        Update target DOA from latest estimation.
        
        Calls _step_smoothed_doa() to gradually approach this target by at most 1° per block.
        This decouples DOA estimation (3 Hz) from steering angle updates (50 Hz per block),
        ensuring smooth beamformer transitions.
        """
        if self.latest_doa is None:
            return
        
        # Initialize or update target
        old_target = self._target_doa
        self._target_doa = float(self.latest_doa)
        
        # First estimate: initialize smoothed to target
        if self._smoothed_doa is None:
            self._smoothed_doa = self._target_doa
            self.logger.debug(f"[DOA] Smoothing initialized at {self._smoothed_doa:.1f}°")
            return
        
        # Log new target if it changed significantly
        if old_target is not None and abs(self._target_doa - old_target) >= 1.0:
            self.logger.debug(f"[DOA] New target: {self._target_doa:.1f}° (was {old_target:.1f}°)")
    
    def _step_smoothed_doa(self):
        """
        Advance smoothed DOA towards target by 0.5–2.25° per block (adaptive stepping).
        Called once per block (every ~20ms) regardless of DOA estimation frequency.
        
        Stepping rate adapts based on confidence:
        - Low confidence: slower (0.5°/block)
        - Medium confidence: normal (1.5°/block)  
        - High confidence: faster (2.25°/block)
        
        This interpolates from current smoothed angle towards the target,
        ensuring beamformer steering changes gradually but responsively.
        """
        if self._smoothed_doa is None or self._target_doa is None:
            return
        
        # Calculate distance to target
        distance = self._target_doa - self._smoothed_doa
        
        # Compute adaptive step based on confidence
        adaptive_multiplier = self._get_adaptive_step_rate()
        adaptive_step = self.smooth_step_deg * adaptive_multiplier
        
        # Move at most adaptive_step towards target
        now = time.monotonic()
        if abs(distance) > adaptive_step:
            # Not at target yet: step towards it
            step_sign = 1.0 if distance > 0 else -1.0
            self._smoothed_doa += step_sign * adaptive_step
            # Throttle stepping logs to avoid flooding
            if now - self._last_stepping_log_time >= self._stepping_log_interval:
                conf_str = f"{self._last_confidence_db:.1f}" if self._last_confidence_db is not None else "??"
                self.logger.debug(
                    f"[DOA] → Stepping: {self._smoothed_doa:.1f}° (target {self._target_doa:.1f}°, "
                    f"dist {distance:.1f}°, conf={conf_str}dB, rate={adaptive_multiplier:.2f}x)"
                )
                self._last_stepping_log_time = now
        else:
            # Close enough: snap to target
            if abs(self._smoothed_doa - self._target_doa) > 1e-9:
                self._smoothed_doa = self._target_doa
                # Log reached-target at most once per interval
                if now - self._last_stepping_log_time >= self._stepping_log_interval:
                    self.logger.debug(f"[DOA] ✓ Reached target: {self._smoothed_doa:.1f}°")
                    self._last_stepping_log_time = now
    
    def get_steering_angle(self) -> float | None:
        """
        Get the angle to use for beamformer steering.
        
        Returns the smoothed DOA to allow gradual transitions instead of sharp jumps.
        This prevents beamformer weight mismatches that cause audio artifacts.
        
        If smoothed DOA is not initialized, returns latest raw estimate.
        """
        if self._smoothed_doa is not None:
            return float(self._smoothed_doa)
        return self.latest_doa

    def estimate_doa(self, audio_block):
        """
        Smooth DOA tracking using local bidirectional hill-climbing.
        
        Behavior:
        - First update performs one full scan to bootstrap initial DOA
        - Subsequent updates evaluate only 3 angles: current, left step, right step
        - Moves to neighbor only if its gain is higher than current gain
        - Updates at most self.update_rate times per second
        - Returns last valid DOA when frozen or before next update interval
        - Works with any beamformer (DAS, MVDR, etc.)
        """
        doa_start = time.perf_counter()
        
        if self.frozen:
            return self.latest_doa

        if audio_block is None:
            return self.latest_doa

        block = np.asarray(audio_block)
        if block.ndim != 2 or block.shape[0] == 0:
            return self.latest_doa

        # Step smoothed DOA towards target on EVERY block (before rate-limiting check)
        # This ensures smooth steering changes at ~50 Hz regardless of DOA update rate (3 Hz)
        self._step_smoothed_doa()

        now = time.monotonic()
        min_interval = 1.0 / self.update_rate
        if (now - self._last_update_time) < min_interval:
            steering_angle = self.get_steering_angle()
            if steering_angle is not None:
                # Throttle this per-block informational log to avoid spamming
                now_log = time.monotonic()
                if now_log - self._last_stepping_log_time >= self._stepping_log_interval:
                    self.logger.debug(f"[DOA] → Stepping towards target: {steering_angle:.1f}°")
                    self._last_stepping_log_time = now_log
            return self.latest_doa

        block_float = np.asarray(block, dtype=np.float64)
        block_rms = float(np.sqrt(np.mean(block_float * block_float) + 1e-12))
        if block_rms < self.min_update_rms:
            doa_time_ms = (time.perf_counter() - doa_start) * 1000.0
            self.last_process_time_ms = doa_time_ms
            self._consecutive_low_energy_frames += 1
            # During low energy (pauses), skip DOA update but keep last DOA value via rate limiting
            return self.latest_doa
        
        # Energy is above threshold - reset low-energy counter
        self._consecutive_low_energy_frames = 0
        
        block_for_doa = self._normalize_block_channels(block) if self.normalize_channels else np.asarray(block, dtype=np.float64)

        # Periodic full scan to escape local maxima
        should_do_periodic_full_scan = (
            self._update_count > 0 
            and self.periodic_full_scan_blocks > 0 
            and self._update_count % self.periodic_full_scan_blocks == 0
        )

        # Bootstrap strategy when DOA is not initialized yet.
        if self.latest_doa is None:
            if self.bootstrap_full_scan:
                init_result = self._initial_full_scan(block_for_doa)
                if init_result is not None:
                    init_angle, init_gain = init_result
                    # Initialize to best angle found
                    self.latest_doa = init_angle
                    self._latest_gain = init_gain
                    self._last_update_time = now
                    self._update_count += 1
                    self._update_smoothed_doa()  # Set target, stepping will begin next block
                    self.logger.debug(
                        f"[DOA] ✓ Bootstrap: {init_angle:.1f}° (rms={block_rms:.5f})"
                    )
                return self.latest_doa

            # Default: initialize at 0° (or nearest bound) and use local search immediately.
            min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
            if min_angle <= 0.0 <= max_angle:
                self.latest_doa = 0.0
            else:
                self.latest_doa = float(np.clip(0.0, min_angle, max_angle))

        # Do periodic full scan or local search
        if should_do_periodic_full_scan:
            self.logger.debug(f"[DOA] Performing periodic full scan at update #{self._update_count}")
            scan_result = self._initial_full_scan(block_for_doa)
            if scan_result is not None:
                scan_angle, scan_gain = scan_result
                # Direct update with target-based smoothing
                self.latest_doa = scan_angle
                self._latest_gain = scan_gain
                self._update_smoothed_doa()  # Set target, stepping will reach it gradually
                doa_str = f"{self.latest_doa:.1f}" if self.latest_doa is not None else "???"
                self.logger.debug(f"[DOA] ↻ Periodic rescan: {doa_str}°")
            self._last_update_time = now
            self._update_count += 1
            doa_time_ms = (time.perf_counter() - doa_start) * 1000.0
            self.last_process_time_ms = doa_time_ms  # Track actual computation time
            doa_str = f"{self.latest_doa:.1f}" if self.latest_doa is not None else "???"
            self.logger.debug(f"[DOA] Estimated: {doa_str}° (took {doa_time_ms:.2f}ms)")
            return self.latest_doa

        # Local hill-climbing search
        min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
        current_angle = float(np.clip(self.latest_doa, min_angle, max_angle))
        
        # Generate search angles around current position within local_search_radius_deg
        search_angles = np.arange(
            current_angle - self.local_search_radius_deg,
            current_angle + self.local_search_radius_deg + 0.5 * self.scan_step_deg,
            self.scan_step_deg
        )
        # Clip to angle_range and ensure current is always included
        candidate_angles = []
        for angle in search_angles:
            clipped = float(np.clip(angle, min_angle, max_angle))
            if not any(np.isclose(clipped, seen) for seen in candidate_angles):
                candidate_angles.append(clipped)
        
        # Ensure current angle is always checked first
        if current_angle not in candidate_angles:
            candidate_angles.insert(0, current_angle)

        # Use vectorized computation for all candidate angles at once
        candidate_angles_arr = np.array(candidate_angles, dtype=np.float64)
        gains = self._compute_gains_vectorized(block_for_doa, candidate_angles_arr)
        
        if gains.size > 0:
            current_gain = gains[candidate_angles_arr.tolist().index(current_angle)]
            best_idx = int(np.argmax(gains))
            best_angle = candidate_angles[best_idx]
            best_gain = gains[best_idx]
            local_confidence_db = self._compute_confidence_db(gains)
            self.latest_confidence_db = local_confidence_db
            self._last_confidence_db = local_confidence_db  # Store for adaptive stepping

            for idx, gain in enumerate(gains):
                self.logger.debug(f"Angle {candidate_angles_arr[idx]:6.1f}° -> gain {gain:.6e}")

            # HYSTERESIS: Move to best angle only if gain improves significantly (min_gain_improvement_db)
            # This prevents oscillation between equally-good neighbors
            if current_gain > 0.0:
                gain_improvement_db = 10.0 * np.log10((best_gain + 1e-10) / (current_gain + 1e-10))
            else:
                gain_improvement_db = 0.0 if best_gain <= current_gain else float('inf')
            
            if best_gain > current_gain and gain_improvement_db >= self.min_gain_improvement_db:
                self.latest_doa = float(best_angle)
                self._latest_gain = float(best_gain)
                self._update_smoothed_doa()  # Apply DOA smoothing
                conf_str = f"{local_confidence_db:.2f}" if local_confidence_db is not None else "??"
                self.logger.debug(
                    f"[DOA] ↗ Moved to {best_angle:.1f}° (+{gain_improvement_db:.2f}dB, conf={conf_str}dB)"
                )
            else:
                self._latest_gain = float(current_gain)
                # Log when hysteresis prevents a move (for debugging)
                if best_gain > current_gain and gain_improvement_db < self.min_gain_improvement_db:
                    self.logger.debug(
                        f"[DOA] ⊗ Hysteresis: didn't move to {best_angle:.1f}° "
                        f"({gain_improvement_db:.2f}dB < {self.min_gain_improvement_db:.2f}dB threshold)"
                    )

            self._last_update_time = now
            self._update_count += 1

        # Log timing information
        doa_time_ms = (time.perf_counter() - doa_start) * 1000.0
        self.last_process_time_ms = doa_time_ms  # Track actual computation time
        doa_str = f"{self.latest_doa:.1f}" if self.latest_doa is not None else "???"
        self.logger.debug(f"[DOA] Estimated: {doa_str}° (took {doa_time_ms:.2f}ms)")
        
        return self.latest_doa
        