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
    ):
        super().__init__(logger=logger, update_rate=update_rate, angle_range=angle_range)

        if beamformer is None:
            raise ValueError("IterativeDOAEstimator requires a Beamformer instance")
        if scan_step_deg <= 0:
            raise ValueError("scan_step_deg must be > 0")

        self.beamformer = beamformer
        self.scan_step_deg = float(scan_step_deg)
        self._last_update_time = 0.0
        self._latest_gain = None

    def estimate_doa(self, audio_block):
        """
        Iteratively scan beamformer steering angles and pick the angle with max output gain.
        
        OPTIMIZED: Caches FFT of input block to avoid recomputing for each angle scan.
        This reduces computation from O(6*N*log(N)) to O(N*log(N) + 6*N).

        Behavior:
        - Scans only within self.angle_range
        - Updates at most self.update_rate times per second
        - Returns last valid DOA when frozen or before next update interval
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

        min_angle, max_angle = float(self.angle_range[0]), float(self.angle_range[1])
        angles = np.arange(min_angle, max_angle + 0.5 * self.scan_step_deg, self.scan_step_deg)
        if angles.size == 0:
            return self.latest_doa

        best_angle = self.latest_doa if self.latest_doa is not None else min_angle
        best_gain = -np.inf

        # OPTIMIZATION: Pre-compute FFT once and reuse for all angles
        # This is the key bottleneck - FFT is O(N*log(N)) and was being computed 6 times
        selected = self.beamformer._select_channels(np.asarray(block, dtype=np.float64))
        n_samples = selected.shape[0]
        
        # Compute FFT once
        spectrum = np.fft.rfft(selected, axis=0)
        channel_count = self.beamformer.channel_count
        sample_rate = self.beamformer.sample_rate
        
        for angle in angles:
            try:
                # FAST PATH: Compute steering vector and apply to cached FFT
                theta_rad = np.deg2rad(float(angle))
                
                # Compute steering matrix for this angle
                freq_bins = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
                direction = np.array(
                    [np.sin(theta_rad), 0.0, np.cos(theta_rad)],
                    dtype=np.float64,
                )
                relative_positions = self.beamformer.mic_positions_m - self.beamformer.mic_positions_m[0]
                delays = (relative_positions @ direction) / self.beamformer.sound_speed_m_s
                steering = np.exp(-1j * 2.0 * np.pi * freq_bins[:, None] * delays[None, :])
                
                # Apply steering to cached spectrum (MVDR beamforming)
                output_spectrum = np.zeros(spectrum.shape[0], dtype=np.complex128)
                identity = np.eye(channel_count, dtype=np.complex128)
                
                for i in range(spectrum.shape[0]):
                    x = spectrum[i, :].reshape(-1, 1)
                    r_inst = x @ x.conj().T
                    
                    # Use 0.9 smoothing factor for stability
                    covariance = 0.9 * np.eye(channel_count, dtype=np.complex128) + 0.1 * r_inst
                    
                    trace_r = np.trace(covariance).real
                    loading = 1e-3 * (trace_r / channel_count + 1e-12)
                    r_loaded = covariance + loading * identity
                    
                    a = steering[i, :].reshape(-1, 1)
                    r_inv_a = np.linalg.pinv(r_loaded) @ a
                    denom = (a.conj().T @ r_inv_a).item()
                    
                    if np.abs(denom) > 1e-12:
                        w = r_inv_a / denom
                    else:
                        w = np.conj(steering[i, :]).reshape(-1, 1) / channel_count
                    
                    output_spectrum[i] = (w.conj().T @ x).item()
                
                # Compute output power
                y_out = np.fft.irfft(output_spectrum, n=n_samples)
                gain = float(np.mean(y_out * y_out))
                
                if gain > best_gain:
                    best_gain = gain
                    best_angle = float(angle)
                    
            except Exception as e:
                self.logger.debug(f"Error computing beamform for angle {angle}: {e}")
                continue

        if np.isfinite(best_gain):
            self.latest_doa = best_angle
            self._latest_gain = best_gain
            self._last_update_time = now

        return self.latest_doa
        