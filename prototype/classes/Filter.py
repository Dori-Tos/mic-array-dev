import scipy.signal as signal
import numpy as np
import logging

class Filter:
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
        
        
class KalmanFilter(Filter):
    """
    
    """
    
    def __init__(self, logger: logging.Logger, sample_rate: int, order: int = 4):
        super().__init__(logger, sample_rate, order=order)

    def apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError("KalmanFilter is not implemented yet. Use WienerFilter for denoising.")


class WienerFilter(Filter):
    """
    Adaptive spectral Wiener denoiser for streaming 1D/2D audio blocks.

    Parameters:
    - noise_alpha: EMA factor for noise PSD tracking (higher = slower updates).
    - gain_floor: Minimum spectral gain to avoid musical artifacts.
    - gain_smooth_alpha: Temporal smoothing factor for per-bin gain.
    - noise_update_snr_db: Update noise model when frame SNR is below this threshold.
    - noise_update_rms: Force noise update for very quiet frames.
    """

    def __init__(
        self,
        logger: logging.Logger,
        sample_rate: int,
        noise_alpha: float = 0.98,
        gain_floor: float = 0.08,
        gain_smooth_alpha: float = 0.75,
        noise_update_snr_db: float = 3.0,
        noise_update_rms: float = 4e-4,
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

        self.noise_alpha = float(noise_alpha)
        self.gain_floor = float(gain_floor)
        self.gain_smooth_alpha = float(gain_smooth_alpha)
        self.noise_update_snr_db = float(noise_update_snr_db)
        self.noise_update_rms = float(noise_update_rms)
        self._eps = 1e-12

        self._noise_psd_1d: np.ndarray | None = None
        self._prev_gain_1d: np.ndarray | None = None
        self._noise_psd_2d: list[np.ndarray] | None = None
        self._prev_gain_2d: list[np.ndarray] | None = None

    def reset(self):
        self._noise_psd_1d = None
        self._prev_gain_1d = None
        self._noise_psd_2d = None
        self._prev_gain_2d = None

    def _denoise_channel(
        self,
        x: np.ndarray,
        noise_psd: np.ndarray | None,
        prev_gain: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            empty = np.array([], dtype=np.float64)
            return empty, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        x_fft = np.fft.rfft(x)
        power = (np.abs(x_fft) ** 2).astype(np.float64, copy=False)

        if noise_psd is None or noise_psd.shape != power.shape:
            noise_psd = power.copy()
        else:
            frame_rms = float(np.sqrt(np.mean(x * x) + self._eps))
            frame_snr_db = 10.0 * np.log10((float(np.mean(power)) + self._eps) / (float(np.mean(noise_psd)) + self._eps))
            if frame_rms <= self.noise_update_rms or frame_snr_db <= self.noise_update_snr_db:
                noise_psd = self.noise_alpha * noise_psd + (1.0 - self.noise_alpha) * power

        post_snr = power / (noise_psd + self._eps)
        gain = np.maximum(self.gain_floor, (post_snr - 1.0) / np.maximum(post_snr, self._eps))

        if prev_gain is None or prev_gain.shape != gain.shape:
            smoothed_gain = gain
        else:
            smoothed_gain = self.gain_smooth_alpha * prev_gain + (1.0 - self.gain_smooth_alpha) * gain

        y_fft = x_fft * smoothed_gain
        y = np.fft.irfft(y_fft, n=x.size)
        return y.astype(np.float64, copy=False), noise_psd, smoothed_gain

    def apply(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            y, self._noise_psd_1d, self._prev_gain_1d = self._denoise_channel(
                arr, self._noise_psd_1d, self._prev_gain_1d
            )
            return y

        if arr.ndim == 2:
            n_ch = arr.shape[1]
            if self._noise_psd_2d is None or len(self._noise_psd_2d) != n_ch:
                self._noise_psd_2d = [None] * n_ch
                self._prev_gain_2d = [None] * n_ch

            out = np.empty_like(arr, dtype=np.float64)
            for ch in range(n_ch):
                y, self._noise_psd_2d[ch], self._prev_gain_2d[ch] = self._denoise_channel(
                    arr[:, ch], self._noise_psd_2d[ch], self._prev_gain_2d[ch]
                )
                out[:, ch] = y
            return out

        raise ValueError("WienerFilter input must be 1D or 2D with shape (samples, channels)")