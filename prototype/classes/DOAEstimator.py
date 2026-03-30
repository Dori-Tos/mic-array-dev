from .Beamformer import Beamformer
import numpy as np
import time
import logging

class DOAEstimator:
    def __init__(self, logger: logging.Logger,
                 update_rate: float = 3.0, angle_range: tuple = (-25, 25)):
        
        if update_rate <= 0:
            raise ValueError("update_rate must be > 0")
        if angle_range[0] > angle_range[1]:
            raise ValueError("angle_range must be (min_angle, max_angle)")
        
        self.logger: logging.Logger = logger
        self.frozen: bool = False
        self.latest_doa = None
        self.update_rate = update_rate
        self.angle_range = angle_range
    
    def freeze(self):
        """
        Freeze the DOA estimator to prevent further updates.
        """
        
        self.frozen = True
    
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
    def __init__(
        self,
        logger: logging.Logger,
        update_rate: float = 3.0,
        angle_range: tuple = (-25, 25),
        beamformer: Beamformer | None = None,
        scan_step_deg: float = 1.0,
        normalize_channels: bool = True,
        bootstrap_full_scan: bool = False,
    ):
        super().__init__(logger=logger, update_rate=update_rate, angle_range=angle_range)

        if beamformer is None:
            raise ValueError("IterativeDOAEstimator requires a Beamformer instance")
        if scan_step_deg <= 0:
            raise ValueError("scan_step_deg must be > 0")

        self.beamformer = beamformer
        self.scan_step_deg = float(scan_step_deg)
        self.normalize_channels = bool(normalize_channels)
        self.bootstrap_full_scan = bool(bootstrap_full_scan)
        self._last_update_time = 0.0
        self._latest_gain = None

    def _compute_gain(self, block: np.ndarray, angle_deg: float) -> float:
        beamformed = self.beamformer.apply(block, theta_deg=float(angle_deg))
        beamformed_arr = np.asarray(beamformed, dtype=np.float64)
        return float(np.mean(beamformed_arr * beamformed_arr))

    def _initial_full_scan(self, block: np.ndarray) -> tuple[float, float] | None:
        min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
        scan_angles = np.arange(min_angle, max_angle + 0.5 * self.scan_step_deg, self.scan_step_deg)
        if scan_angles.size == 0:
            return None

        if min_angle <= 0.0 <= max_angle and not np.any(np.isclose(scan_angles, 0.0)):
            scan_angles = np.append(scan_angles, 0.0)

        angles = np.array(sorted(scan_angles.tolist(), key=lambda a: (abs(a), a)), dtype=np.float64)

        best_angle = float(angles[0])
        best_gain = -np.inf
        for angle in angles:
            try:
                gain = self._compute_gain(block, float(angle))
                self.logger.debug(f"Angle {angle:6.1f}° -> gain {gain:.6e}")
                if gain > best_gain:
                    best_gain = gain
                    best_angle = float(angle)
            except Exception as e:
                self.logger.error(f"Error computing beamform for angle {angle}: {e}", exc_info=True)
                continue

        if not np.isfinite(best_gain):
            return None
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

        # Bootstrap strategy when DOA is not initialized yet.
        if self.latest_doa is None:
            if self.bootstrap_full_scan:
                init_result = self._initial_full_scan(block_for_doa)
                if init_result is not None:
                    self.latest_doa, self._latest_gain = init_result
                    self._last_update_time = now
                return self.latest_doa

            # Default: initialize at 0° (or nearest bound) and use local search immediately.
            min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
            if min_angle <= 0.0 <= max_angle:
                self.latest_doa = 0.0
            else:
                self.latest_doa = float(np.clip(0.0, min_angle, max_angle))

        min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
        current_angle = float(np.clip(self.latest_doa, min_angle, max_angle))
        left_angle = float(np.clip(current_angle - self.scan_step_deg, min_angle, max_angle))
        right_angle = float(np.clip(current_angle + self.scan_step_deg, min_angle, max_angle))

        # Keep order deterministic and remove duplicates (at boundaries).
        candidate_angles = []
        for a in (current_angle, left_angle, right_angle):
            if not any(np.isclose(a, seen) for seen in candidate_angles):
                candidate_angles.append(a)

        gain_by_angle: dict[float, float] = {}
        for angle in candidate_angles:
            try:
                gain = self._compute_gain(block_for_doa, angle)
                gain_by_angle[angle] = gain
                self.logger.debug(f"Angle {angle:6.1f}° -> gain {gain:.6e}")
            except Exception as e:
                self.logger.error(f"Error computing beamform for angle {angle}: {e}", exc_info=True)

        if gain_by_angle:
            current_gain = gain_by_angle.get(current_angle, -np.inf)
            best_angle = max(gain_by_angle, key=gain_by_angle.get)
            best_gain = gain_by_angle[best_angle]

            # Move only if improved; otherwise keep current angle.
            if best_gain > current_gain:
                self.latest_doa = float(best_angle)
                self._latest_gain = float(best_gain)
            else:
                self.latest_doa = current_angle
                self._latest_gain = float(current_gain)

            self._last_update_time = now

        return self.latest_doa
        