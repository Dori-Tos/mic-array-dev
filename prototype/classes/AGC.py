
import numpy as np
import logging
import time


class AGC:
    """
    Automatic Gain Control (AGC) class that applies gain to audio samples to maintain a target RMS level.
    
    Parameters:
    - target_rms: Desired RMS level for the output audio (default: 0.12).
    - min_gain: Minimum gain factor to prevent excessive attenuation (default: 0.2).
    - max_gain: Maximum gain factor to prevent excessive amplification (default: 20.0).
    - attack_ms: Time in milliseconds for the AGC to reduce gain when the signal is too hot (default: 50 ms).
    - release_ms: Time in milliseconds for the AGC to increase gain when the signal is too quiet (default: 300 ms).
    - noise_floor_rms: RMS threshold below which the AGC will apply a gate to prevent boosting background noise (default: 0.01).
    - gate_gain: Gain factor applied when the input signal is below the noise floor (default: 0.15).    
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        target_rms: float = 0.12,
        min_gain: float = 0.2,
        max_gain: float = 20.0,
        attack_ms: float = 50.0,
        release_ms: float = 300.0,
        noise_floor_rms: float = 0.002,
        gate_gain: float = 1.0,
    ):
        if target_rms <= 0:
            raise ValueError("target_rms must be > 0")
        if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
            raise ValueError("invalid min_gain/max_gain")
        if attack_ms <= 0:
            raise ValueError("attack_ms must be > 0")
        if release_ms <= 0:
            raise ValueError("release_ms must be > 0")
        if noise_floor_rms < 0:
            raise ValueError("noise_floor_rms must be >= 0")
        if not 0.0 <= gate_gain <= 1.0:
            raise ValueError("gate_gain must be in [0, 1]")

        self.logger: logging.Logger = logger
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.noise_floor_rms = float(noise_floor_rms)
        self.gate_gain = float(gate_gain)

        self.current_gain = 1.0
        self._eps = 1e-9
        
        # Debug logging: track time for 1-second log intervals
        self._last_log_time = time.time()
        self._last_desired_gain = 1.0

    def reset(self):
        self.current_gain = 1.0
        self._last_log_time = time.time()
        self._last_desired_gain = 1.0

    def _time_to_alpha(self, delay_ms: float, frame_duration_s: float) -> float:
        delay_s = max(delay_ms * 1e-3, 1e-6)
        # alpha = 1 - exp(-T/delay)
        alpha = 1.0 - float(np.exp(-frame_duration_s / delay_s))
        return float(np.clip(alpha, 1e-4, 1.0))

    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply AGC to mono float audio in [-1, 1], using attack/release time constants."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        if sample_rate <= 0:
            return np.clip(x, -1.0, 1.0)

        rms = float(np.sqrt(np.mean(x * x) + self._eps))
        desired_unclamped = self.target_rms / max(rms, self._eps)
        desired_gain = float(np.clip(desired_unclamped, self.min_gain, self.max_gain))
        below_noise_floor = rms < self.noise_floor_rms
        
        # Debug logging: log every second (regardless of noise floor)
        current_time = time.time()
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(
                f"[AGC] Input RMS: {rms:.6f} | Target: {self.target_rms:.6f} | "
                f"Desired: {desired_gain:.2f} (raw {desired_unclamped:.2f}) | Current: {self.current_gain:.2f} | "
                f"Clamped: {desired_gain != desired_unclamped} | BelowFloor: {below_noise_floor}"
            )
            self._last_log_time = current_time
            self._last_desired_gain = desired_gain
        
        # Prevent AGC from boosting low-level background noise.
        if below_noise_floor:
            # Hold current gain (no upward gain chasing in near-noise frames)
            # and optionally apply a mild gate attenuation.
            y = x * self.current_gain * self.gate_gain
            return np.clip(y, -1.0, 1.0)

        frame_duration_s = float(x.size / float(sample_rate))

        # Attack: reduce gain quickly when signal is above the threshold.
        # Release: increase gain more slowly to avoid pumping.
        if desired_gain < self.current_gain:
            coeff = self._time_to_alpha(self.attack_ms, frame_duration_s)
        else:
            coeff = self._time_to_alpha(self.release_ms, frame_duration_s)

        self.current_gain += coeff * (desired_gain - self.current_gain)
        y = x * self.current_gain
        return np.clip(y, -1.0, 1.0)


class TwoStageAGC:
    """Chain two AGC stages: fast leveler first, slow makeup second."""

    def __init__(self, logger: logging.Logger, stage1: AGC, stage2: AGC):
        self.logger: logging.Logger = logger
        self.stage1 = stage1
        self.stage2 = stage2

    def reset(self):
        if hasattr(self.stage1, "reset"):
            self.stage1.reset()
        if hasattr(self.stage2, "reset"):
            self.stage2.reset()

    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        y = self.stage1.process(samples, sample_rate)
        y = self.stage2.process(y, sample_rate)
        return np.asarray(y, dtype=np.float32).reshape(-1)