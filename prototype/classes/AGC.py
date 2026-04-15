
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
    :param speech_activity_rms: RMS threshold below which the stage will avoid upward adaptation
                                and decay gain toward min_gain (default: 0.001).
    :param silence_decay_alpha: Per-frame decay factor used when rms < speech_activity_rms
                                (default: 0.03). Smaller = slower return to min_gain.
    :param activity_hold_ms: Hold time after recent speech activity before allowing silence decay
                             (default: 450 ms). Prevents word/syllable dropouts.
    :param peak_protect_threshold: Peak level above which desired gain is damped to protect
                                   downstream compressor/limiter (default: 0.35).
    :param peak_protect_strength: Strength of peak-based damping in [0,1] (default: 0.85).
    :param max_gain_warn_rms_min: Minimum RMS required before emitting max-gain clamp warnings.
                                  Prevents warning spam in near-silence (default: 0.001).
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        target_rms: float = 0.1,
        min_gain: float = 1.0,
        max_gain: float = 16.0,
        adapt_alpha: float = 0.05,
        rms_floor: float = 1e-4,
        speech_activity_rms: float = 0.001,
        silence_decay_alpha: float = 0.03,
        activity_hold_ms: float = 450.0,
        peak_protect_threshold: float = 0.35,
        peak_protect_strength: float = 0.85,
        max_gain_warn_rms_min: float = 0.001,
    ):
        if target_rms <= 0:
            raise ValueError("target_rms must be > 0")
        if min_gain <= 0 or max_gain <= 0 or min_gain > max_gain:
            raise ValueError("invalid min_gain/max_gain")
        if not 0.01 <= adapt_alpha <= 0.5:
            raise ValueError("adapt_alpha should be in [0.01, 0.5] for stable adaptation")
        if rms_floor <= 0:
            raise ValueError("rms_floor must be > 0")
        if speech_activity_rms <= 0:
            raise ValueError("speech_activity_rms must be > 0")
        if not 0.0 < silence_decay_alpha <= 0.5:
            raise ValueError("silence_decay_alpha should be in (0.0, 0.5]")
        if activity_hold_ms < 0:
            raise ValueError("activity_hold_ms must be >= 0")
        if not 0.0 < peak_protect_threshold < 1.0:
            raise ValueError("peak_protect_threshold must be in (0, 1)")
        if not 0.0 <= peak_protect_strength <= 1.0:
            raise ValueError("peak_protect_strength must be in [0, 1]")
        if max_gain_warn_rms_min < 0:
            raise ValueError("max_gain_warn_rms_min must be >= 0")
        
        self.logger = logger
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.adapt_alpha = float(adapt_alpha)
        self.rms_floor = float(rms_floor)
        self.speech_activity_rms = float(speech_activity_rms)
        self.silence_decay_alpha = float(silence_decay_alpha)
        self.activity_hold_ms = float(activity_hold_ms)
        self.peak_protect_threshold = float(peak_protect_threshold)
        self.peak_protect_strength = float(peak_protect_strength)
        self.max_gain_warn_rms_min = float(max_gain_warn_rms_min)
        self._eps = 1e-9
        
        self.current_gain = 1.0  # Start at unity gain
        self._last_active_time = time.time()
        self._last_log_time = time.time()
        self._last_max_gain_warn_time = 0.0
        self._last_peak_warn_time = 0.0
    
    def reset(self):
        self.current_gain = 1.0
        self._last_active_time = time.time()
        self._last_log_time = time.time()
        self._last_max_gain_warn_time = 0.0
        self._last_peak_warn_time = 0.0
    
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
        
        # In low-activity frames, do not chase noise/silence upward. Instead, decay gain to min_gain.
        is_active_frame = rms >= self.speech_activity_rms
        if is_active_frame:
            self._last_active_time = current_time
        in_activity_hold = (current_time - self._last_active_time) <= (self.activity_hold_ms * 1e-3)

        desired_unclamped = self.current_gain
        if is_active_frame:
            rms_to_use = max(rms, self.rms_floor)
            desired_unclamped = self.target_rms / rms_to_use
            desired_gain = float(np.clip(desired_unclamped, self.min_gain, self.max_gain))
        elif in_activity_hold:
            # Preserve gain briefly between speech segments to avoid dropping quiet syllables.
            desired_gain = self.current_gain
        else:
            desired_gain = self.current_gain + self.silence_decay_alpha * (self.min_gain - self.current_gain)
            desired_gain = float(np.clip(desired_gain, self.min_gain, self.max_gain))

        # Peak-aware damping: when transient peaks are high, reduce allowed gain headroom.
        if peak > self.peak_protect_threshold:
            denom = max(1.0 - self.peak_protect_threshold, self._eps)
            over_norm = float(np.clip((peak - self.peak_protect_threshold) / denom, 0.0, 1.0))
            damping = 1.0 - (self.peak_protect_strength * over_norm)
            damped_max_gain = self.min_gain + (self.max_gain - self.min_gain) * damping
            desired_gain = min(desired_gain, float(np.clip(damped_max_gain, self.min_gain, self.max_gain)))

        # Warn if desired gain is being clamped to max_gain.
        # if (
        #     is_active_frame
        #     and rms >= self.max_gain_warn_rms_min
        #     and desired_unclamped > self.max_gain
        #     and (current_time - self._last_max_gain_warn_time) >= 1.0
        # ):
        #     self.logger.warning(
        #         f"[AdaptiveAmplifier] Desired gain {desired_unclamped:.2f}x exceeds max_gain {self.max_gain:.2f}x; "
        #         f"clamping to max. Input RMS={rms:.5f}, target_rms={self.target_rms:.5f}"
        #     )
        #     self._last_max_gain_warn_time = current_time
        
        # Apply exponential smoothing to adapt gain slowly (avoid pumping)
        self.current_gain += self.adapt_alpha * (desired_gain - self.current_gain)
        self.current_gain = float(np.clip(self.current_gain, self.min_gain, self.max_gain))
        
        # Apply amplification
        y = x * self.current_gain
        peak_after = peak * self.current_gain

        if peak_after > 1.0 and (current_time - self._last_peak_warn_time) >= 1.0:
            self.logger.warning(
                f"[AdaptiveAmplifier] Output peak {peak_after:.3f} exceeds 1.0 before downstream limiting/compression"
            )
            self._last_peak_warn_time = current_time
        
        # Debug logging (every 1 second)
        if current_time - self._last_log_time >= 1.0:
            self.logger.debug(
                f"[AdaptiveAmplifier] Input RMS: {rms:.5f} | Target: {self.target_rms:.5f} | "
                f"Desired gain: {desired_gain:.2f}x | Current gain: {self.current_gain:.2f}x | "
                f"Peak before: {peak:.4f} → after: {peak_after:.4f} | "
                f"ActiveFrame: {is_active_frame} | Hold: {in_activity_hold}"
            )
            self._last_log_time = current_time
        
        return np.asarray(y, dtype=np.float32)


class NoiseAwareAdaptiveAmplifier:
    """
    Noise-aware adaptive amplifier that prevents background noise poisoning by:
    1. Continuously estimating the noise floor (via minimum energy in recent history)
    2. Only increasing gain when SNR > threshold (typical: 6-10 dB)
    3. Using asymmetric dynamics: fast gain-down, very slow gain-up (prevents chasing artifacts)
    4. Capping max_gain inversely with noise floor so residual noise cannot bloom
    
    This solves the "ghost noise" problem where AGC amplifies post-event residual noise
    into audible artifacts that persist and worsen over time.
    
    :param target_rms: Desired RMS at output during speech (default: 0.08)
    :param min_gain: Minimum gain (typically 0.7-1.0 to allow attenuation of silence/noise)
    :param max_gain_baseline: Maximum gain baseline (default: 6.0); actual max will be capped by noise floor
    :param gain_up_alpha: Exponential smoothing for gain increase (SLOW; default: 0.008 = very smooth)
    :param gain_down_alpha: Exponential smoothing for gain decrease (FAST; default: 0.1 = fast response)
    :param snr_threshold_db: Only increase gain when SNR > this threshold (default: 8.0 dB)
    :param noise_floor_alpha: Exponential smoothing for noise floor estimate (default: 0.997 = very slow)
    :param activity_hold_ms: Hold gain after speech ends before allowing decay (default: 200 ms)
    :param peak_protect_threshold: Reduce gain when peaks exceed this (default: 0.35)
    :param peak_protect_strength: Strength of peak damping (default: 1.0)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        target_rms: float = 0.08,
        min_gain: float = 0.7,
        max_gain_baseline: float = 6.0,
        gain_up_alpha: float = 0.008,
        gain_down_alpha: float = 0.1,
        snr_threshold_db: float = 8.0,
        noise_floor_alpha: float = 0.997,
        activity_hold_ms: float = 200.0,
        peak_protect_threshold: float = 0.35,
        peak_protect_strength: float = 1.0,
    ):
        if target_rms <= 0:
            raise ValueError("target_rms must be > 0")
        if min_gain <= 0 or max_gain_baseline <= 0 or min_gain > max_gain_baseline:
            raise ValueError("min_gain and max_gain_baseline must be > 0 and min <= max")
        if not (0.001 <= gain_up_alpha <= 0.5):
            raise ValueError("gain_up_alpha must be in [0.001, 0.5]")
        if not (0.01 <= gain_down_alpha <= 1.0):
            raise ValueError("gain_down_alpha must be in [0.01, 1.0]")
        if snr_threshold_db < 0:
            raise ValueError("snr_threshold_db must be >= 0")
        if not (0.9 <= noise_floor_alpha < 1.0):
            raise ValueError("noise_floor_alpha must be in [0.9, 1.0)")
        
        self.logger = logger
        self.target_rms = float(target_rms)
        self.min_gain = float(min_gain)
        self.max_gain_baseline = float(max_gain_baseline)
        self.gain_up_alpha = float(gain_up_alpha)
        self.gain_down_alpha = float(gain_down_alpha)
        self.snr_threshold_db = float(snr_threshold_db)
        self.noise_floor_alpha = float(noise_floor_alpha)
        self.activity_hold_ms = float(activity_hold_ms)
        self.peak_protect_threshold = float(peak_protect_threshold)
        self.peak_protect_strength = float(peak_protect_strength)
        
        self.current_gain = 1.0
        self.noise_floor_rms = 0.0001  # Start with conservative floor estimate
        self._last_active_time = time.time()
        self._last_log_time = time.time()
        self._eps = 1e-9
    
    def reset(self):
        self.current_gain = 1.0
        self.noise_floor_rms = 0.0001
        self._last_active_time = time.time()
        self._last_log_time = time.time()
    
    def process(self, samples: np.ndarray, sample_rate: float, coherence_signal: np.ndarray | None = None) -> np.ndarray:
        """
        Apply noise-aware adaptive amplification.
        Prevents AGC from chasing and amplifying residual noise into audible artifacts.
        
        :param samples: Input audio samples
        :param sample_rate: Sample rate in Hz
        :param coherence_signal: Optional inter-channel coherence per frequency bin (shape: freq_bins,).
                                 If provided, gates AGC boost when coherence is low (diffuse noise, back diffraction).
                                 Range: [0, 1] where 1=coherent point-source, 0=incoherent diffuse noise.
        """
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        
        rms = float(np.sqrt(np.mean(x * x) + self._eps))
        peak = float(np.max(np.abs(x)))
        current_time = time.time()
        
        # Update noise floor estimate: smoothly track the minimum energy
        # This represents the true background noise floor
        self.noise_floor_rms = (
            self.noise_floor_alpha * self.noise_floor_rms +
            (1.0 - self.noise_floor_alpha) * min(rms, self.noise_floor_rms * 1.5)
        )
        self.noise_floor_rms = float(np.clip(self.noise_floor_rms, 1e-5, 0.01))
        
        # Compute SNR in dB
        snr_db = 20.0 * float(np.log10(max(rms, self._eps) / max(self.noise_floor_rms, self._eps)))
        
        # COHERENCE-BASED GATING: Suppress AGC boost when coherence is low (diffuse/back noise)
        # Coherence is an indicator of signal quality:
        # - High coherence (>0.7) = coherent point-source (speech) -> allow normal AGC boost
        # - Low coherence (<0.5) = diffuse noise (back/room diffraction) -> suppress boost
        coherence_gate = 1.0  # Default: full boost
        if coherence_signal is not None:
            avg_coherence = float(np.mean(np.clip(coherence_signal, 0.0, 1.0)))
            # Linear gate: coherence of 0.7 = full boost (1.0), coherence of 0.4 = half boost (0.5), coherence of 0.0 = no boost
            coherence_threshold = 0.6  # Below this, start suppressing boost
            if avg_coherence < coherence_threshold:
                # Suppress boost strength when coherence is low
                coherence_gate = np.clip((avg_coherence - 0.2) / (coherence_threshold - 0.2), 0.0, 1.0)
        
        # Only allow gain to increase if SNR is above threshold (indicating speech, not noise)
        # Otherwise, allow only gain decrease to prevent noise amplification
        is_high_snr = snr_db >= self.snr_threshold_db
        
        if is_high_snr:
            self._last_active_time = current_time
        
        in_hold = (current_time - self._last_active_time) <= (self.activity_hold_ms * 1e-3)
        can_increase_gain = is_high_snr or in_hold
        
        # Desired gain based on RMS
        desired_unclamped = self.target_rms / max(rms, self._eps)
        
        # Dynamic max gain cap based on noise floor:
        # If noise floor is high, reduce max_gain so it doesn't get amplified
        # Formula: max_gain_effective = min(baseline, target / (k * noise_floor))
        # where k=2-4 provides safety margin
        k = 3.0
        max_gain_noise_limited = self.target_rms / max(k * self.noise_floor_rms, self._eps)
        max_gain_effective = min(self.max_gain_baseline, max_gain_noise_limited)
        
        desired_gain = float(np.clip(desired_unclamped, self.min_gain, max_gain_effective))
        
        # Peak protect: reduce gain when peaks are high
        if peak > self.peak_protect_threshold:
            peak_damping = 1.0 - self.peak_protect_strength * min(1.0, (peak - self.peak_protect_threshold) / (1.0 - self.peak_protect_threshold))
            desired_gain *= peak_damping
        
        # Apply coherence gate: suppress upward adaptation when coherence is low (back noise)
        # If coherence is low, desired_gain won't increase as much
        if coherence_gate < 1.0:
            # Blend toward current gain (no increase) when coherence is poor
            desired_gain = self.current_gain + (desired_gain - self.current_gain) * coherence_gate
        
        # Asymmetric gain adaptation: fast down, slow up
        if desired_gain > self.current_gain:
            if can_increase_gain:
                # Slow gain increase only when SNR is high
                self.current_gain += self.gain_up_alpha * (desired_gain - self.current_gain)
            # else: hold gain, do not increase
        else:
            # Fast gain decrease to immediately suppress loud events
            self.current_gain += self.gain_down_alpha * (desired_gain - self.current_gain)
        
        self.current_gain = float(np.clip(self.current_gain, self.min_gain, max_gain_effective))
        
        # Apply amplification
        y = x * self.current_gain
        peak_after = peak * self.current_gain
        
        # Debug logging (every 1 second)
        if current_time - self._last_log_time >= 1.0:
            coherence_str = f"Coherence: {float(np.mean(coherence_signal if coherence_signal is not None else [0.0])):.3f}" if coherence_signal is not None else "NoCoherence"
            self.logger.debug(
                f"[NoiseAwareAdaptiveAmplifier] RMS: {rms:.5f} | Noise floor: {self.noise_floor_rms:.5f} | "
                f"SNR: {snr_db:.1f} dB | {coherence_str} (gate: {coherence_gate:.2f}) | Current gain: {self.current_gain:.2f}x (max eff: {max_gain_effective:.2f}x) | "
                f"Peak before: {peak:.4f} → after: {peak_after:.4f} | HighSNR: {is_high_snr}"
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
    
    def process(self, samples: np.ndarray, sample_rate: float, coherence_signal: np.ndarray | None = None) -> np.ndarray:
        """
        Process through all stages sequentially.
        
        :param samples: Input audio samples
        :param sample_rate: Sample rate in Hz
        :param coherence_signal: Optional inter-channel coherence signal. Passed to stages that support coherence gating.
        """
        y = samples
        for stage in self.stages:
            # Pass coherence to stages that support it; ignore for stages that don't
            if hasattr(stage, 'process'):
                import inspect
                sig = inspect.signature(stage.process)
                if 'coherence_signal' in sig.parameters:
                    y = stage.process(y, sample_rate, coherence_signal=coherence_signal)
                else:
                    y = stage.process(y, sample_rate)
            else:
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