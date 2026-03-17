import scipy.signal as signal
import numpy as np


class Filter:
    def __init__(self, sample_rate: int, order: int = 4):
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
    
    def __init__(self, sample_rate: int, cutoff_freq: float, order: int = 4):
        super().__init__(sample_rate, order=order)
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
    
    def __init__(self, sample_rate: int, cutoff_freq: float, order: int = 4):
        super().__init__(sample_rate, order=order)
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
    
    def __init__(self, sample_rate: int, low_cutoff: float, high_cutoff: float, order: int = 4):
        super().__init__(sample_rate, order=order)
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
    
    def __init__(self, sample_rate: int, low_cutoff: float, high_cutoff: float, order: int = 4):
        super().__init__(sample_rate, order=order)
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