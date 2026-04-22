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
    :param beamformer: Beamformer instance to use for computing the beamformed output at different angles. This allows the DOA estimator to be flexible and work with any beamforming algorithm (e.g., DAS, MVDR).
    :param scan_step_deg: Step size in degrees for scanning neighboring angles during the local search
        (default: 1.0). Smaller steps may yield more accurate DOA estimates but increase computational load.
    :param normalize_channels: If True, normalize each channel's RMS before DOA estimation to reduce bias from mic gain mismatch (default: True).
    :param bootstrap_full_scan: If True, perform a full scan of the angle range on the first update to initialize the DOA estimate (default: True). This ensures the algorithm starts at the global best angle instead of 0°.
    :param periodic_full_scan_blocks: Number of update iterations between full rescans (default: 60). Set to 0 to disable periodic scans. Periodic full scans help escape local maxima and explore the full angle range.
    :param local_search_radius_deg: Radius in degrees around current DOA to search during local hill-climbing (default: 5.0). Larger values cover more range per iteration (e.g., 5.0 checks from -5° to +5°). This allows escaping plateaus faster but costs more computation.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        update_rate: float = 3.0,
        angle_range: tuple = (-25, 25),
        beamformer: Beamformer | None = None,
        scan_step_deg: float = 1.0,
        normalize_channels: bool = True,
        bootstrap_full_scan: bool = True,
        periodic_full_scan_blocks: int = 60,
        local_search_radius_deg: float = 5.0,
    ):
        super().__init__(logger=logger, update_rate=update_rate, angle_range=angle_range)

        if beamformer is None:
            raise ValueError("IterativeDOAEstimator requires a Beamformer instance")
        if scan_step_deg <= 0:
            raise ValueError("scan_step_deg must be > 0")
        if local_search_radius_deg <= 0:
            raise ValueError("local_search_radius_deg must be > 0")

        self.beamformer = beamformer
        self.scan_step_deg = float(scan_step_deg)
        self.normalize_channels = bool(normalize_channels)
        self.bootstrap_full_scan = bool(bootstrap_full_scan)
        self.periodic_full_scan_blocks = int(periodic_full_scan_blocks)
        self.local_search_radius_deg = float(local_search_radius_deg)
        self._last_update_time = 0.0
        self._latest_gain = None
        self.latest_confidence_db: float | None = None
        self._update_count = 0  # Track updates for periodic full scan
        self.last_process_time_ms: float = 0.0  # Actual computation time in ms (not including early returns)

    def _compute_gain(self, block: np.ndarray, angle_deg: float) -> float:
        beamformed = self.beamformer.apply(block, theta_deg=float(angle_deg))
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
        selected = self.beamformer._select_channels(np.asarray(block, dtype=np.float64))
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
            steering = self.beamformer._steering_matrix(theta_rad=theta_rad, n_fft=n_samples)  # (freq_bins, channels)
            
            # Apply steering (complex conjugate) and sum across channels: shape (freq_bins,)
            weights = np.conj(steering) / self.beamformer.channel_count
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

        now = time.monotonic()
        min_interval = 1.0 / self.update_rate
        if (now - self._last_update_time) < min_interval:
            return self.latest_doa

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
                    self.latest_doa, self._latest_gain = init_result
                    self._last_update_time = now
                    self._update_count += 1
                return self.latest_doa

            # Default: initialize at 0° (or nearest bound) and use local search immediately.
            min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
            if min_angle <= 0.0 <= max_angle:
                self.latest_doa = 0.0
            else:
                self.latest_doa = float(np.clip(0.0, min_angle, max_angle))

        # Do periodic full scan or local search
        if should_do_periodic_full_scan:
            self.logger.debug(f"[DOA] Performing full scan at update #{self._update_count}")
            scan_result = self._initial_full_scan(block_for_doa)
            if scan_result is not None:
                self.latest_doa, self._latest_gain = scan_result
                self.logger.debug(f"[DOA] Full scan result: {self.latest_doa:.1f}°")
            self._last_update_time = now
            self._update_count += 1
            doa_time_ms = (time.perf_counter() - doa_start) * 1000.0
            self.last_process_time_ms = doa_time_ms  # Track actual computation time
            self.logger.debug(f"[DOA] Estimated: {self.latest_doa:.1f}° (took {doa_time_ms:.2f}ms)")
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

            for idx, gain in enumerate(gains):
                self.logger.debug(f"Angle {candidate_angles_arr[idx]:6.1f}° -> gain {gain:.6e}")

            # Simple logic: move to best angle found
            if best_gain > current_gain:
                self.latest_doa = float(best_angle)
                self._latest_gain = float(best_gain)
                gain_diff_db = 10.0 * np.log10((best_gain + 1e-10) / (current_gain + 1e-10)) if current_gain > 0 else 0
                self.logger.debug(f"[DOA] Moving to {best_angle:.1f}° (gain diff: {gain_diff_db:.2f}dB)")
            else:
                self._latest_gain = float(current_gain)

            self._last_update_time = now
            self._update_count += 1

        # Log timing information
        doa_time_ms = (time.perf_counter() - doa_start) * 1000.0
        self.last_process_time_ms = doa_time_ms  # Track actual computation time
        self.logger.debug(f"[DOA] Estimated: {self.latest_doa:.1f}° (took {doa_time_ms:.2f}ms)")
        
        return self.latest_doa
        