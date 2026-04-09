import scipy.signal as signal
import numpy as np
import logging
from collections import deque

class Filter:
    """
    Common base class for audio filters. Subclasses implement specific filter types (e.g., high-pass, low-pass).
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio to be processed (e.g., 16000 or 48000 Hz).
    :param order: Filter order (default: 4). Higher order means steeper rolloff but more computational load.    
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, order: int = 4):
        self.logger = logger
        self.sample_rate = sample_rate
        self.order = order
        self.sos = None
        self._zi = None

    def _validate_cutoff(self, cutoff_freq: float):
        nyquist = self.sample_rate / 2
        if cutoff_freq <= 0 or cutoff_freq >= nyquist:
            raise ValueError(
                f"cutoff_freq must be between 0 and {nyquist} Hz, got {cutoff_freq}"
            )

    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim != 2:
            raise ValueError("Filter input must be 1D or 2D with shape (samples, channels)")
        return arr

    def _ensure_state(self, num_channels: int):
        expected_sections = self.sos.shape[0]
        if self._zi is None or self._zi.shape != (expected_sections, 2, num_channels):
            self._zi = np.zeros((expected_sections, 2, num_channels), dtype=np.float64)

    def reset(self):
        self._zi = None

    def apply(self, data: np.ndarray) -> np.ndarray:
        if self.sos is None:
            raise RuntimeError("Filter coefficients are not initialized")

        arr = self._prepare_data(data)
        num_channels = arr.shape[1]
        self._ensure_state(num_channels)

        filtered, self._zi = signal.sosfilt(self.sos, arr, axis=0, zi=self._zi)

        if np.asarray(data).ndim == 1:
            return filtered[:, 0]
        return filtered


class HighPassFilter(Filter):
    """
    A high-pass filter that allows frequencies above the cutoff frequency to pass through.
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio to be processed (e.g., 16000 or 48000 Hz).
    :param cutoff_freq: Cutoff frequency in Hz (must be > 0 and < Nyquist frequency).
    :param order: Filter order (default: 4). Higher order means steeper rolloff but more computational load.  
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, cutoff_freq: float, order: int = 4):
        super().__init__(logger, sample_rate, order=order)
        self.cutoff_freq = cutoff_freq
        self._validate_cutoff(cutoff_freq)
        self.sos = signal.butter(
            self.order,
            cutoff_freq / (sample_rate / 2),
            btype='highpass',
            output='sos'
        )
    
    
class LowPassFilter(Filter):
    """
    A low-pass filter that allows frequencies below the cutoff frequency to pass through.
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio to be processed (e.g., 16000 or 48000 Hz).
    :param cutoff_freq: Cutoff frequency in Hz (must be > 0 and < Nyquist frequency).
    :param order: Filter order (default: 4). Higher order means steeper rolloff but more computational load.  
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, cutoff_freq: float, order: int = 4):
        super().__init__(logger, sample_rate, order=order)
        self.cutoff_freq = cutoff_freq
        self._validate_cutoff(cutoff_freq)
        self.sos = signal.butter(
            self.order,
            cutoff_freq / (sample_rate / 2),
            btype='lowpass',
            output='sos'
        )
        
    
class BandPassFilter(Filter):
    """
    A band-pass filter that allows frequencies between low_cutoff and high_cutoff to pass through.
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio to be processed (e.g., 16000 or 48000 Hz).
    :param low_cutoff: Lower cutoff frequency in Hz (must be > 0 and < high_cutoff).
    :param high_cutoff: Upper cutoff frequency in Hz (must be > low_cutoff and < Nyquist frequency).
    :param order: Filter order (default: 4). Higher order means steeper rolloff but more computational load.
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, low_cutoff: float, high_cutoff: float, order: int = 4):
        super().__init__(logger, sample_rate, order=order)
        self._validate_cutoff(low_cutoff)
        self._validate_cutoff(high_cutoff)
        if low_cutoff >= high_cutoff:
            raise ValueError("low_cutoff must be less than high_cutoff")
        self.sos = signal.butter(
            self.order,
            [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)],
            btype='bandpass',
            output='sos'
        )
        
        
class BandStopFilter(Filter):
    """
    A band-stop (notch) filter that attenuates frequencies between low_cutoff and high_cutoff.
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio to be processed (e.g., 16000 or 48000 Hz).
    :param low_cutoff: Lower cutoff frequency in Hz (must be > 0 and < high_cutoff).
    :param high_cutoff: Upper cutoff frequency in Hz (must be > low_cutoff and < Nyquist frequency).
    :param order: Filter order (default: 4). Higher order means steeper rolloff but more computational load.
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, low_cutoff: float, high_cutoff: float, order: int = 4):
        super().__init__(logger, sample_rate, order=order)
        self._validate_cutoff(low_cutoff)
        self._validate_cutoff(high_cutoff)
        if low_cutoff >= high_cutoff:
            raise ValueError("low_cutoff must be less than high_cutoff")
        self.sos = signal.butter(
            self.order,
            [low_cutoff / (sample_rate / 2), high_cutoff / (sample_rate / 2)],
            btype='bandstop',
            output='sos'
        )
        


class WienerFilter(Filter):
    """
    Adaptive spectral Wiener denoiser with speech-aware enhancements for streaming 1D/2D audio blocks.

    Speech-aware features:
    - Pre-emphasis boost (2–4 kHz) for consonant preservation
    - Formant preservation (protects vowel resonances at F1/F2/F3)
    - Spectral continuity detection (smooth regions = speech; spiky = noise)
    - Frequency-dependent gain floor (gentler on speech bands, aggressive on rumble/hiss)

    :param noise_alpha: EMA factor for noise PSD tracking (higher = slower updates).
    :param gain_floor: Minimum spectral gain to avoid musical artifacts.
    :param gain_smooth_alpha: Temporal smoothing factor for per-bin gain.
    :param noise_update_snr_db: Update noise model when frame SNR is below this threshold.
    :param noise_update_rms: Force noise update for very quiet frames.
    :param pre_emphasis_db: Boost amount (dB) for consonant region 2–4 kHz (default: 3.0 dB).
    :param formant_preservation_db: Boost amount (dB) for formant regions (default: 2.0 dB).
    :param spectral_continuity_factor: Smoothing bias toward speech-like regions (0–1, default: 0.4).
    """

    def __init__(
        self,
        logger: logging.Logger,
        sample_rate: int,
        noise_alpha: float = 0.985,
        gain_floor: float = 0.05,
        gain_smooth_alpha: float = 0.3,
        noise_update_snr_db: float = 6.0,
        noise_update_rms: float = 8e-4,
        pre_emphasis_db: float = 3.0,
        formant_preservation_db: float = 2.0,
        spectral_continuity_factor: float = 0.4,
    ):
        super().__init__(logger, sample_rate, order=1)
        if not 0.0 < noise_alpha < 1.0:
            raise ValueError("noise_alpha must be in (0, 1)")
        if not 0.0 <= gain_floor <= 1.0:
            raise ValueError("gain_floor must be in [0, 1]")
        if not 0.0 <= gain_smooth_alpha < 1.0:
            raise ValueError("gain_smooth_alpha must be in [0, 1)")
        if noise_update_rms < 0.0:
            raise ValueError("noise_update_rms must be >= 0")
        if pre_emphasis_db < 0:
            raise ValueError("pre_emphasis_db must be >= 0")
        if formant_preservation_db < 0:
            raise ValueError("formant_preservation_db must be >= 0")
        if not 0.0 <= spectral_continuity_factor <= 1.0:
            raise ValueError("spectral_continuity_factor must be in [0, 1]")

        self.noise_alpha = float(noise_alpha)
        self.gain_floor = float(gain_floor)
        self.gain_smooth_alpha = float(gain_smooth_alpha)
        self.noise_update_snr_db = float(noise_update_snr_db)
        self.noise_update_rms = float(noise_update_rms)
        self.pre_emphasis_db = float(pre_emphasis_db)
        self.formant_preservation_db = float(formant_preservation_db)
        self.spectral_continuity_factor = float(spectral_continuity_factor)
        self._eps = 1e-12

        self._noise_psd_1d: np.ndarray | None = None
        self._prev_gain_1d: np.ndarray | None = None
        self._noise_psd_2d: list[np.ndarray] | None = None
        self._prev_gain_2d: list[np.ndarray] | None = None
        self._prev_power_1d: np.ndarray | None = None
        self._prev_power_2d: list[np.ndarray] | None = None

    def reset(self):
        self._noise_psd_1d = None
        self._prev_gain_1d = None
        self._noise_psd_2d = None
        self._prev_gain_2d = None
        self._prev_power_1d = None
        self._prev_power_2d = None

    def _speech_aware_gain_floor(self, freq_hz: np.ndarray) -> np.ndarray:
        """
        Compute frequency-dependent gain floor.
        - Gentle on speech band (300–3000 Hz): gain_floor
        - Aggressive on rumble (<100 Hz): gain_floor * 0.3
        - Aggressive on hiss (>7000 Hz): gain_floor * 0.2
        """
        floor = np.ones_like(freq_hz, dtype=np.float64) * self.gain_floor
        floor[freq_hz < 100] *= 0.1
        floor[freq_hz > 7000] *= 0.2
        return floor

    def _formant_preservation_boost(self, freq_hz: np.ndarray) -> np.ndarray:
        """
        Boost gain in typical formant regions (vowel resonances):
        - F1: 500–1000 Hz (vowel identity)
        - F2: 1500–2500 Hz (consonant-vowel transition)
        - F3: 2500–3500 Hz (voice quality)
        - Pre-emphasis: 2000–4000 Hz (consonant sharpness)
        """
        pre_emphasis_linear = 10.0 ** (self.pre_emphasis_db / 20.0)
        formant_linear = 10.0 ** (self.formant_preservation_db / 20.0)

        boost = np.ones_like(freq_hz, dtype=np.float64)
        boost[(freq_hz >= 500) & (freq_hz <= 1000)] *= formant_linear  # F1
        boost[(freq_hz >= 1500) & (freq_hz <= 2500)] *= formant_linear  # F2
        boost[(freq_hz >= 2500) & (freq_hz <= 3500)] *= formant_linear  # F3
        boost[(freq_hz >= 2000) & (freq_hz <= 4000)] *= pre_emphasis_linear  # Pre-emphasis
        return boost

    def _spectral_continuity(self, power: np.ndarray, prev_power: np.ndarray | None) -> np.ndarray:
        """
        Detect spectral continuity (smooth regions likely speech, spiky regions likely noise).
        Returns a smoothness mask [0, 1]: high = smooth speech-like, low = spiky noise-like.
        """
        if prev_power is None or prev_power.shape != power.shape:
            return np.ones_like(power, dtype=np.float64)

        # Normalized difference between consecutive frames
        diff = np.abs(power - prev_power) / (np.abs(prev_power) + self._eps)
        # Smooth regions have low diff; spiky regions have high diff
        # Invert: continuity = 1 / (1 + diff) => high continuity for smooth, low for spiky
        continuity = 1.0 / (1.0 + np.clip(diff, 0, 10.0))
        return continuity

    def _denoise_channel(
        self,
        x: np.ndarray,
        noise_psd: np.ndarray | None,
        prev_gain: np.ndarray | None,
        prev_power: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            empty = np.array([], dtype=np.float64)
            return empty, np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        x_fft = np.fft.rfft(x)
        power = (np.abs(x_fft) ** 2).astype(np.float64, copy=False)

        if noise_psd is None or noise_psd.shape != power.shape:
            noise_psd = power.copy()
        else:
            frame_rms = float(np.sqrt(np.mean(x * x) + self._eps))
            frame_snr_db = 10.0 * np.log10((float(np.mean(power)) + self._eps) / (float(np.mean(noise_psd)) + self._eps))
            if frame_rms <= self.noise_update_rms or frame_snr_db <= self.noise_update_snr_db:
                noise_psd = self.noise_alpha * noise_psd + (1.0 - self.noise_alpha) * power

        # Compute speech-aware gains
        freq_bins = np.fft.rfftfreq(x.size, d=1.0 / self.sample_rate)
        speech_floor = self._speech_aware_gain_floor(freq_bins)
        formant_boost = self._formant_preservation_boost(freq_bins)
        continuity_mask = self._spectral_continuity(power, prev_power)

        # Base Wiener gain
        post_snr = power / (noise_psd + self._eps)
        wiener_gain = np.maximum(speech_floor, (post_snr - 1.0) / np.maximum(post_snr, self._eps))

        # Apply formant preservation: reduce suppression on speech-like frequencies
        gain = wiener_gain * formant_boost

        # Apply spectral continuity bias: smooth regions get more gain (less suppression)
        # Add continuity-aware boost to smooth regions (likely speech)
        continuity_boost = 1.0 + (continuity_mask - 0.5) * self.spectral_continuity_factor
        gain = gain * np.clip(continuity_boost, 0.7, 1.3)

        if prev_gain is None or prev_gain.shape != gain.shape:
            smoothed_gain = gain
        else:
            smoothed_gain = self.gain_smooth_alpha * prev_gain + (1.0 - self.gain_smooth_alpha) * gain

        y_fft = x_fft * smoothed_gain
        y = np.fft.irfft(y_fft, n=x.size)
        return y.astype(np.float64, copy=False), noise_psd, smoothed_gain, power

    def apply(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            y, self._noise_psd_1d, self._prev_gain_1d, self._prev_power_1d = self._denoise_channel(
                arr, self._noise_psd_1d, self._prev_gain_1d, self._prev_power_1d
            )
            return y

        if arr.ndim == 2:
            n_ch = arr.shape[1]
            if self._noise_psd_2d is None or len(self._noise_psd_2d) != n_ch:
                self._noise_psd_2d = [None] * n_ch
                self._prev_gain_2d = [None] * n_ch
                self._prev_power_2d = [None] * n_ch

            out = np.empty_like(arr, dtype=np.float64)
            for ch in range(n_ch):
                y, self._noise_psd_2d[ch], self._prev_gain_2d[ch], self._prev_power_2d[ch] = self._denoise_channel(
                    arr[:, ch], self._noise_psd_2d[ch], self._prev_gain_2d[ch], self._prev_power_2d[ch]
                )
                out[:, ch] = y
            return out

        raise ValueError("WienerFilter input must be 1D or 2D with shape (samples, channels)")
    
    
class KalmanFilter(Filter):
    """
    
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, order: int = 4):
        super().__init__(logger, sample_rate, order=order)

    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("KalmanFilter is not implemented yet. Use WienerFilter for denoising.")


class SpectralSubtractionFilter(Filter):
    """
    Spectral subtraction denoiser with 50% overlap-add (COLA) processing.
    
    Uses professional overlap-add windowing to smooth transitions while preserving
    voice naturalness. Single-window approach (analysis side only) for transparent processing.
    
    **How it works**:
    1. Accumulates input in a buffer (50% overlap)
    2. Applies Hann window to frame (COLA-compliant)
    3. Computes FFT magnitude spectrum
    4. Performs spectral subtraction (gentle, configurable)
    5. Inverse FFT (no additional windowing on output)
    6. Overlaps with previous frame's second half for smooth blending
    
    Single-window approach ensures:
    - Natural voice quality (less robotic artifacts)
    - Proper COLA property (perfect reconstruction at 50% overlap)
    - Smooth frame-to-frame transitions
    
    :param logger: logging.Logger instance for logging messages.
    :param sample_rate: Sample rate of the audio (e.g., 48000 Hz).
    :param noise_factor: Subtraction aggressiveness (default: 0.8). Lower = gentler, more natural voice.
    :param gain_floor: Minimum spectral gain (default: 0.3 = -10dB). Higher = more signal preservation.
    :param noise_alpha: EMA factor for noise PSD estimation (default: 0.98). Higher = slower tracking.
    :param noise_update_snr_db: Update noise when SNR below threshold (default: 3.0 dB).
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        sample_rate: int,
        order: int = 4,
        noise_factor: float = 0.8,
        gain_floor: float = 0.3,
        noise_alpha: float = 0.98,
        noise_update_snr_db: float = 3.0,
    ):
        super().__init__(logger, sample_rate, order=order)
        
        if noise_factor <= 0:
            raise ValueError("noise_factor must be > 0")
        if not 0.0 <= gain_floor <= 1.0:
            raise ValueError("gain_floor must be in [0, 1]")
        if not 0.0 < noise_alpha < 1.0:
            raise ValueError("noise_alpha must be in (0, 1)")
        
        self.noise_factor = float(noise_factor)
        self.gain_floor = float(gain_floor)
        self.noise_alpha = float(noise_alpha)
        self.noise_update_snr_db = float(noise_update_snr_db)
        self._eps = 1e-12
        
        # 50% overlap-add buffers: store half-frame (hop_size)
        self.hop_size = None  # Will be determined from first input
        self._input_buffer_1d = None
        self._output_buffer_1d = None
        self._noise_psd_1d = None
        
        self._input_buffer_2d = None
        self._output_buffer_2d = None
        self._noise_psd_2d = None
        
        self._last_log_time = None
        
        # State tracking: prevent corruption during silence-to-speech transition
        self._silence_to_speech_transition_frames = 0
        self._max_transition_frames = 60  # ~1.25 seconds at 48kHz with 960-sample blocks
        self._last_frame_rms = 0.0
        self._silence_threshold_rms = 0.001  # < 1 mV = silence

    
    def reset(self):
        """Reset all state."""
        self._input_buffer_1d = None
        self._output_buffer_1d = None
        self._noise_psd_1d = None
        self._input_buffer_2d = None
        self._output_buffer_2d = None
        self._noise_psd_2d = None
        self._last_log_time = None
        self._silence_to_speech_transition_frames = 0
        self._last_frame_rms = 0.0
    
    def _process_frame(self, frame: np.ndarray, noise_psd: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame with spectral subtraction.
        
        Single-window approach: apply Hann window on analysis side only (before FFT).
        This preserves naturalness while ensuring COLA property at 50% overlap.
        
        Includes transition detection to prevent noise model corruption during silence->speech onset.
        """
        frame_len = len(frame)
        frame_rms = float(np.sqrt(np.mean(frame * frame) + self._eps))
        
        # Detect silence-to-speech transition: RMS jumps significantly
        is_silence = frame_rms < self._silence_threshold_rms
        was_silence = self._last_frame_rms < self._silence_threshold_rms
        
        if was_silence and not is_silence:
            # Transition from silence to speech detected
            self._silence_to_speech_transition_frames = self._max_transition_frames
        
        # Decrement transition counter each frame
        if self._silence_to_speech_transition_frames > 0:
            self._silence_to_speech_transition_frames -= 1
        
        self._last_frame_rms = frame_rms
        
        # Analysis window: only applied before FFT, not after IFFT
        window = np.hanning(frame_len)
        x_windowed = frame * window
        
        # FFT
        x_fft = np.fft.rfft(x_windowed)
        power = (np.abs(x_fft) ** 2).astype(np.float64, copy=False)
        
        # Initialize or update noise PSD (with transition protection)
        if noise_psd is None or noise_psd.shape != power.shape:
            noise_psd = power.copy()
        else:
            snr_db = 10.0 * np.log10((np.mean(power) + self._eps) / (np.mean(noise_psd) + self._eps))
            
            # Only update noise model if:
            # 1. SNR is low (actual noise state)
            # 2. AND we're NOT in speech-onset transition (prevents corruption)
            if snr_db < self.noise_update_snr_db and self._silence_to_speech_transition_frames == 0:
                noise_psd = self.noise_alpha * noise_psd + (1.0 - self.noise_alpha) * power
        
        # Spectral subtraction: gentle and configurable
        gain = np.maximum(self.gain_floor, 1.0 - (self.noise_factor * noise_psd) / (power + self._eps))
        y_fft = x_fft * gain
        
        # IFFT: no additional windowing on output
        # (Single-window approach maintains COLA property at 50% overlap)
        y = np.fft.irfft(y_fft, n=frame_len)
        
        return y, noise_psd
    
    def _denoise_channel_with_overlap_add(
        self,
        x: np.ndarray,
        input_buffer: np.ndarray | None,
        output_buffer: np.ndarray | None,
        noise_psd: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process channel with 50% overlap-add buffering.
        
        Returns: (output, new_input_buffer, new_output_buffer, noise_psd)
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        
        if x.size == 0:
            return np.array([], dtype=np.float64), input_buffer, output_buffer, noise_psd
        
        # On first call, determine frame size from input
        if self.hop_size is None:
            self.hop_size = x.size
        
        frame_len = 2 * self.hop_size  # Full frame = 2 * hop_size (50% overlap)
        
        # Initialize buffers on first use
        if input_buffer is None:
            input_buffer = np.zeros(self.hop_size, dtype=np.float64)
            output_buffer = np.zeros(2 * self.hop_size, dtype=np.float64)
        
        
        # Create output array
        output = np.zeros_like(x)
        
        # Process in hop_size chunks
        for i in range(0, x.size, self.hop_size):
            # Get next chunk (may be smaller on last iteration)
            chunk = x[i:i + self.hop_size]
            chunk_len = len(chunk)
            
            # Concatenate with buffered samples: [buffered_half | new_chunk]
            if chunk_len == self.hop_size:
                frame = np.concatenate([input_buffer, chunk])
            else:
                # Last chunk: pad with zeros
                chunk_padded = np.concatenate([chunk, np.zeros(self.hop_size - chunk_len)])
                frame = np.concatenate([input_buffer, chunk_padded])
            
            # Process frame
            y_windowed, noise_psd = self._process_frame(frame, noise_psd)
            
            # Overlap-add: add previous output buffer's second half
            y_windowed[:self.hop_size] += output_buffer[self.hop_size:]
            
            # Store new output buffer for next iteration
            output_buffer = y_windowed.copy()
            
            # Extract and store output (first hop_size samples)
            out_chunk = y_windowed[:chunk_len]
            output[i:i + chunk_len] = out_chunk
            
            # Update input buffer with new chunk for next iteration
            input_buffer = chunk if chunk_len == self.hop_size else np.zeros(self.hop_size)
        
        return output, input_buffer, output_buffer, noise_psd
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction with 50% overlap-add to audio data.
        
        :param data: Audio array, shape (samples,) for mono or (samples, channels) for multi-channel.
        :return: Denoised audio array with same shape as input.
        """
        arr = np.asarray(data, dtype=np.float64)
        
        if arr.ndim == 1:
            # Mono processing
            y, self._input_buffer_1d, self._output_buffer_1d, self._noise_psd_1d = \
                self._denoise_channel_with_overlap_add(
                    arr, self._input_buffer_1d, self._output_buffer_1d, self._noise_psd_1d
                )
            return y
        
        elif arr.ndim == 2:
            # Multi-channel processing
            n_ch = arr.shape[1]
            
            if self._input_buffer_2d is None:
                self._input_buffer_2d = [None] * n_ch
                self._output_buffer_2d = [None] * n_ch
                self._noise_psd_2d = [None] * n_ch
            
            out = np.empty_like(arr, dtype=np.float64)
            for ch in range(n_ch):
                y, self._input_buffer_2d[ch], self._output_buffer_2d[ch], self._noise_psd_2d[ch] = \
                    self._denoise_channel_with_overlap_add(
                        arr[:, ch], self._input_buffer_2d[ch], self._output_buffer_2d[ch], self._noise_psd_2d[ch]
                    )
                out[:, ch] = y
            
            return out
        
        else:
            raise ValueError("SpectralSubtractionFilter input must be 1D or 2D with shape (samples, channels)")