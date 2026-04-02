
import numpy as np
import logging
import time


class Amplifier:
    """
    Simple fixed-gain amplifier stage. Applies constant gain multiplication.
    Can be used before limiters or other AGC for output level control without modulation.
    
    :param gain: Linear gain factor (default: 4.0). E.g., gain=4.0 means 12dB amplification.
    :param max_output: Clamp output to this level to prevent hard clipping (default: 1.0).
    """
    
    def __init__(self, logger: logging.Logger, gain: float = 4.0, max_output: float = 1.0):
        if gain <= 0:
            raise ValueError("gain must be > 0")
        if max_output <= 0 or max_output > 1.0:
            raise ValueError("max_output must be in (0, 1]")
        
        self.logger = logger
        self.gain = float(gain)
        self.max_output = float(max_output)
        self._last_log_time = time.time()
    
    def reset(self):
        self._last_log_time = time.time()
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply fixed gain amplification."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        
        y = x * self.gain
        peak_before = float(np.max(np.abs(x)))
        peak_after = float(np.max(np.abs(y)))
        
        # Debug logging
        current_time = time.time()
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(f"[Amplifier] Gain: {self.gain:.2f}x | Peak before: {peak_before:.4f} → after: {peak_after:.4f}")
            self._last_log_time = current_time
        
        return np.asarray(y, dtype=np.float32)


class AdaptiveAmplifier:
    """
    Adaptive amplifier that estimates an appropriate baseline gain based on input statistics.
    Unlike AGC, this adapts SLOWLY to find the right operating level, not to track a target output.
    
    **Design Philosophy**:
    - Measures input RMS/peak to estimate what gain is needed
    - Updates gain gradually (exponential smoothing) to avoid pumping
    - Designed as a PRE-stage before AGC/Limiter (which handle fast dynamics)
    
    **Typical Usage**:
    ```python
    agc_chain = AGCChain(logger, stages=[
        AdaptiveAmplifier(logger, target_rms=0.1, min_gain=1.0, max_gain=16.0),
        PedalboardAGC(logger, ...)  # Handles limiter + compressor
    ])
    ```
    
    :param target_rms: Desired RMS level at amplifier output (default: 0.1 = -20dB).
                       This is the BASELINE level aimed for (PedalboardAGC handles fine tuning).
    :param min_gain: Minimum gain to prevent over-attenuation (default: 1.0).
    :param max_gain: Maximum gain to prevent excessive amplification (default: 16.0 = 24dB).
    :param adapt_alpha: Exponential smoothing factor for gain updates (default: 0.05 = slow).
                        Smaller = slower adaptation (less pumping). Range: [0.01, 0.5].
    :param rms_floor: Minimum RMS to avoid dividing by zero or boosting silence (default: 1e-4).
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        target_rms: float = 0.1,
        min_gain: float = 1.0,
        max_gain: float = 16.0,
        adapt_alpha: float = 0.05,
        rms_floor: float = 1e-4,
    ):
        if target_rms <= 0:
            raise ValueError("target_rms must be > 0")
        if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
            raise ValueError("invalid min_gain/max_gain")
        if not 0.01 <= adapt_alpha <= 0.5:
            raise ValueError("adapt_alpha should be in [0.01, 0.5] for stable adaptation")
        if rms_floor <= 0:
            raise ValueError("rms_floor must be > 0")
        
        self.logger = logger
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.adapt_alpha = float(adapt_alpha)
        self.rms_floor = float(rms_floor)
        self._eps = 1e-9
        
        self.current_gain = 1.0  # Start at unity gain
        self._last_log_time = time.time()
    
    def reset(self):
        self.current_gain = 1.0
        self._last_log_time = time.time()
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply adaptive amplification based on input RMS.
        Gain adapts slowly to reach target_rms without pumping artifacts.
        """
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        
        # Measure input RMS and peak
        rms = float(np.sqrt(np.mean(x * x) + self._eps))
        peak = float(np.max(np.abs(x)))
        current_time = time.time()
        
        # Calculate desired gain for this frame
        rms_to_use = max(rms, self.rms_floor)  # Avoid boosting silence
        desired_unclamped = self.target_rms / rms_to_use
        desired_gain = float(np.clip(desired_unclamped, self.min_gain, self.max_gain))
        
        # Apply exponential smoothing to adapt gain slowly (avoid pumping)
        self.current_gain += self.adapt_alpha * (desired_gain - self.current_gain)
        self.current_gain = float(np.clip(self.current_gain, self.min_gain, self.max_gain))
        
        # Apply amplification
        y = x * self.current_gain
        peak_after = peak * self.current_gain
        
        # Debug logging (every 1 second)
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(
                f"[AdaptiveAmplifier] Input RMS: {rms:.5f} | Target: {self.target_rms:.5f} | "
                f"Desired gain: {desired_gain:.2f}x | Current gain: {self.current_gain:.2f}x | "
                f"Peak before: {peak:.4f} → after: {peak_after:.4f}"
            )
            self._last_log_time = current_time
        
        return np.asarray(y, dtype=np.float32)


class AGCChain:
    """
    Chain multiple AGC/processing stages (Amplifier, Limiter, AGC, PeakHoldAGC, etc.).
    Similar to how filters are chained—processes signal through each stage sequentially.
    """
    
    def __init__(self, logger: logging.Logger, stages: list):
        if not stages or not isinstance(stages, list):
            raise ValueError("stages must be a non-empty list of AGC/Amplifier/Limiter objects")
        
        self.logger = logger
        self.stages = stages
    
    def reset(self):
        for stage in self.stages:
            if hasattr(stage, "reset"):
                stage.reset()
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Process through all stages sequentially."""
        y = samples
        for stage in self.stages:
            y = stage.process(y, sample_rate)
        return np.asarray(y, dtype=np.float32).reshape(-1)


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
        gate_open_ms: float = 12.0,
        gate_close_ms: float = 220.0,
        gate_hold_ms: float = 160.0,
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
        if gate_open_ms <= 0 or gate_close_ms <= 0 or gate_hold_ms < 0:
            raise ValueError("invalid gate timing values")

        self.logger: logging.Logger = logger
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.noise_floor_rms = float(noise_floor_rms)
        self.gate_gain = float(gate_gain)
        self.gate_open_ms = float(gate_open_ms)
        self.gate_close_ms = float(gate_close_ms)
        self.gate_hold_ms = float(gate_hold_ms)

        self.current_gain = 1.0
        self._gate_value = 1.0
        self._last_above_floor_time = time.time()
        self._eps = 1e-9
        
        # Debug logging: track time for 1-second log intervals
        self._last_log_time = time.time()
        self._last_desired_gain = 1.0

    def reset(self):
        self.current_gain = 1.0
        self._gate_value = 1.0
        self._last_above_floor_time = time.time()
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
        frame_duration_s = float(x.size / float(sample_rate))
        current_time = time.time()

        # Gate hangover: keep gate open briefly after speech drops below threshold.
        if not below_noise_floor:
            self._last_above_floor_time = current_time

        in_hold = (current_time - self._last_above_floor_time) < (self.gate_hold_ms * 1e-3)
        gate_target = 1.0 if (not below_noise_floor or in_hold) else self.gate_gain

        gate_ms = self.gate_open_ms if gate_target > self._gate_value else self.gate_close_ms
        gate_coeff = self._time_to_alpha(gate_ms, frame_duration_s)
        self._gate_value += gate_coeff * (gate_target - self._gate_value)
        gate_applied = float(np.clip(self._gate_value, self.gate_gain, 1.0))
        
        # Debug logging: log every second (regardless of noise floor)
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(
                f"[AGC] Input RMS: {rms:.6f} | Target: {self.target_rms:.6f} | "
                f"Desired: {desired_gain:.2f} (raw {desired_unclamped:.2f}) | Current: {self.current_gain:.2f} | "
                f"Clamped: {desired_gain != desired_unclamped} | BelowFloor: {below_noise_floor} | Gate: {gate_applied:.2f}"
            )
            self._last_log_time = current_time
            self._last_desired_gain = desired_gain
        
        # Prevent AGC from boosting low-level background noise.
        if below_noise_floor and not in_hold:
            # Hold current gain (no upward gain chasing in near-noise frames)
            # and optionally apply a mild gate attenuation.
            y = x * self.current_gain * gate_applied
            return np.asarray(y, dtype=np.float32)

        # Attack: reduce gain quickly when signal is above the threshold.
        # Release: increase gain more slowly to avoid pumping.
        if desired_gain < self.current_gain:
            coeff = self._time_to_alpha(self.attack_ms, frame_duration_s)
        else:
            coeff = self._time_to_alpha(self.release_ms, frame_duration_s)

        self.current_gain += coeff * (desired_gain - self.current_gain)
        y = x * self.current_gain * gate_applied
        return np.asarray(y, dtype=np.float32)


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


class Limiter:
    """
    Simple hard limiter: Prevents output from exceeding a threshold.
    No gain control—just clamps peaks above threshold.
    More predictable than AGC: no artifacts from gain ramping.
    
    :param threshold: Maximum output level (default: 0.95). Values above this are clipped.
    :param soft_knee_db: Optional soft-knee region in dB (default: 0). 0 = hard knee (hard clipping).
    :param release_ms: Release time after peak (default: 50ms). Longer = smoother recovery.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        threshold: float = 0.95,
        soft_knee_db: float = 0.0,
        release_ms: float = 50.0,
    ):
        if threshold <= 0 or threshold > 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if soft_knee_db < 0:
            raise ValueError("soft_knee_db must be >= 0")
        if release_ms <= 0:
            raise ValueError("release_ms must be > 0")
        
        self.logger = logger
        self.threshold = float(threshold)
        self.soft_knee_db = float(soft_knee_db)
        self.release_ms = float(release_ms)
        self._eps = 1e-9
        
        # Convert soft knee to linear: threshold ± (soft_knee_db/2)
        self.soft_knee_factor = 10.0 ** (soft_knee_db / 20.0) if soft_knee_db > 0 else 1.0
        self.soft_knee_start = threshold / self.soft_knee_factor
        
        self._last_log_time = time.time()
    
    def reset(self):
        self._last_log_time = time.time()
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply hard limiting to prevent clipping. No gain control."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        
        peak = float(np.max(np.abs(x)))
        
        if self.soft_knee_db > 0:
            # Soft knee: smooth transition between start and threshold
            if peak > self.soft_knee_start:
                # Gain reduction in soft region
                overshoot_ratio = (peak - self.soft_knee_start) / (self.threshold - self.soft_knee_start)
                overshoot_ratio = float(np.clip(overshoot_ratio, 0.0, 1.0))
                # Apply smooth gain reduction curve
                gain_reduction = 1.0 - (overshoot_ratio ** 2.0)
                y = x * gain_reduction
            else:
                y = x
        else:
            # Hard knee: clip anything above threshold
            y = np.clip(x, -self.threshold, self.threshold)
        
        # Debug logging
        current_time = time.time()
        if current_time - self._last_log_time >= 1.0 and peak > self.threshold * 0.8:
            self.logger.debug(f"[Limiter] Peak: {peak:.4f} | Threshold: {self.threshold:.4f}")
            self._last_log_time = current_time
        
        return np.asarray(y, dtype=np.float32)


class PeakHoldAGC:
    """
    Peak-tracking AGC: More stable than RMS-based AGC.
    Tracks peak amplitude instead of RMS for more predictable gain control.
    Slower release to prevent pumping on transients.
    
    :param target_peak: Desired peak level (default: 0.8). AGC aims to keep peaks at this level.
    :param min_gain: Minimum gain to avoid over-attenuation (default: 0.5).
    :param max_gain: Maximum gain to avoid excessive amplification (default: 6.0).
    :param attack_ms: Fast gain reduction when peak exceeds target (default: 10ms).
    :param release_ms: Slow gain increase when peak is below target (default: 500ms). Prevents pumping.
    :param peak_hold_ms: How long to hold the last peak measurement (default: 50ms).
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        target_peak: float = 0.8,
        min_gain: float = 0.5,
        max_gain: float = 6.0,
        attack_ms: float = 10.0,
        release_ms: float = 500.0,
        peak_hold_ms: float = 50.0,
    ):
        if target_peak <= 0 or target_peak > 1.0:
            raise ValueError("target_peak must be in (0, 1]")
        if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
            raise ValueError("invalid min_gain/max_gain")
        if attack_ms <= 0 or release_ms <= 0:
            raise ValueError("attack_ms and release_ms must be > 0")
        if peak_hold_ms <= 0:
            raise ValueError("peak_hold_ms must be > 0")
        
        self.logger = logger
        self.target_peak = float(target_peak)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.peak_hold_ms = float(peak_hold_ms)
        self._eps = 1e-9
        
        self.current_gain = 1.0
        self._last_peak = 0.0
        self._peak_hold_start_time = time.time()
        self._last_log_time = time.time()
    
    def reset(self):
        self.current_gain = 1.0
        self._last_peak = 0.0
        self._peak_hold_start_time = time.time()
        self._last_log_time = time.time()
    
    def _time_to_alpha(self, delay_ms: float, frame_duration_s: float) -> float:
        delay_s = max(delay_ms * 1e-3, 1e-6)
        alpha = 1.0 - float(np.exp(-frame_duration_s / delay_s))
        return float(np.clip(alpha, 1e-4, 1.0))
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply peak-tracking AGC: More stable than RMS-based."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        if sample_rate <= 0:
            return x
        
        frame_duration_s = float(x.size / float(sample_rate))
        current_time = time.time()
        
        # Measure current peak
        peak = float(np.max(np.abs(x)))
        
        # Peak hold: keep the highest peak for peak_hold_ms
        if peak > self._last_peak:
            self._last_peak = peak
            self._peak_hold_start_time = current_time
        else:
            # Release the held peak after peak_hold_ms
            hold_elapsed = (current_time - self._peak_hold_start_time) * 1000.0
            if hold_elapsed > self.peak_hold_ms:
                self._last_peak = peak
        
        # Calculate desired gain based on held peak
        held_peak = self._last_peak
        if held_peak > self._eps:
            desired_gain = self.target_peak / held_peak
        else:
            desired_gain = self.max_gain
        
        desired_gain = float(np.clip(desired_gain, self.min_gain, self.max_gain))
        
        # Attack: fast reduction when gain must drop
        # Release: slow increase when gain can rise (prevents pumping)
        if desired_gain < self.current_gain:
            coeff = self._time_to_alpha(self.attack_ms, frame_duration_s)
        else:
            coeff = self._time_to_alpha(self.release_ms, frame_duration_s)
        
        self.current_gain += coeff * (desired_gain - self.current_gain)
        y = x * self.current_gain
        
        # Debug logging
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(
                f"[PeakHoldAGC] Peak: {peak:.4f} | Held: {held_peak:.4f} | "
                f"Target: {self.target_peak:.4f} | Desired: {desired_gain:.2f} | Current: {self.current_gain:.2f}"
            )
            self._last_log_time = current_time
        
        return np.asarray(y, dtype=np.float32)


class PedalboardAGC:
    """
    Professional-grade dynamics processing using Spotify's Pedalboard library.
    Combines Compressor and Limiter in a single stage for transparent, artifact-free
    volume control without spectral distortion.
    
    Uses time-domain adaptive processing with automatic windowing and overlap-add
    handling (internal to Pedalboard). Zero artifacts from FFT block processing.
    
    **Architecture**:
    - Compressor: Gentle adaptive gain reduction (threshold-based)
    - Limiter: Hard ceiling protection to prevent clipping at output
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of audio (typically 48000 Hz).
    :param threshold_db: Compressor threshold in dB below reference level (default: -30 dB).
    :param ratio: Compressor ratio (default: 4.0 = 4:1 compression).
    :param attack_ms: Compressor attack time in milliseconds (default: 10 ms).
    :param release_ms: Compressor release time in milliseconds (default: 100 ms).
    :param limiter_threshold_db: Limiter threshold in dB (default: -0.1 dB, just below clipping).
    :param limiter_release_ms: Limiter release time in milliseconds (default: 50 ms).
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        sample_rate: int,
        threshold_db: float = -30.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        limiter_threshold_db: float = -0.1,
        limiter_release_ms: float = 50.0,
    ):
        """
        Initialize PedalboardAGC with compressor and limiter stages.
        
        The compressor reduces level above threshold (gentle adaptive gain).
        The limiter prevents clipping at output (hard ceiling).
        """
        try:
            import pedalboard
        except ImportError:
            raise RuntimeError(
                "Pedalboard library not found. Install with: pip install pedalboard"
            )
        
        self.logger = logger
        self.sample_rate = sample_rate
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.limiter_threshold_db = float(limiter_threshold_db)
        self.limiter_release_ms = float(limiter_release_ms)
        self.pedalboard = pedalboard
        
        # Create pedalboard chain: Compressor → Limiter
        self.board = pedalboard.Pedalboard([
            pedalboard.Compressor(
                threshold_db=self.threshold_db,
                ratio=self.ratio,
                attack_ms=self.attack_ms,
                release_ms=self.release_ms,
            ),
            pedalboard.Limiter(
                threshold_db=self.limiter_threshold_db,
                release_ms=self.limiter_release_ms,
            ),
        ])
        
        self.logger.info(
            f"PedalboardAGC initialized at {sample_rate} Hz. "
            f"Compressor: threshold={threshold_db}dB, ratio={ratio}:1, "
            f"attack={attack_ms}ms, release={release_ms}ms. "
            f"Limiter: threshold={limiter_threshold_db}dB, release={limiter_release_ms}ms"
        )
    
    def reset(self):
        """Reset Pedalboard state (typically not needed, but provided for API compatibility)."""
        pass
    
    def process(self, samples: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply Pedalboard processing (compressor + limiter) to audio samples.
        
        Handles both 1D and 2D arrays. Automatically processes with proper windowing
        and overlap-add handling (internal to Pedalboard).
        
        :param samples: Audio array, shape (samples,) for mono or (samples, channels) for multi-channel.
        :param sample_rate: Sampling rate (used for compatibility, actual rate is from __init__).
        :return: Processed audio array with same shape as input.
        """
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        
        try:
            processed = self.board(x, self.sample_rate)
            return np.asarray(processed, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"PedalboardAGC.process() failed: {e}")
            raise