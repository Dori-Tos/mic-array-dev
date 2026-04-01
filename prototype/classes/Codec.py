import numpy as np
import logging
import zlib


class Codec:
    """Base class for one-way (encode-only) codec demonstrations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _to_mono_float(self, samples: np.ndarray) -> np.ndarray:
        """Return mono float32 samples in [-1, 1] for encoding."""
        x = np.asarray(samples)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        if x.ndim == 2:
            # Collapse multichannel to mono for simple demo transport.
            x = np.mean(x, axis=1)
        elif x.ndim != 1:
            raise ValueError("samples must be 1D or 2D")

        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(x, -1.0, 1.0)

    def encode(self, samples: np.ndarray, sample_rate: int) -> bytes:
        raise NotImplementedError("Codec subclasses must implement encode()")

class G711Codec(Codec):
    """G.711 mu-law encoder (8-bit PCM payload)."""

    def __init__(self, logger: logging.Logger, mu: int = 255):
        super().__init__(logger)
        if mu <= 0:
            raise ValueError("mu must be > 0")
        self.mu = int(mu)

    def encode(self, samples: np.ndarray, sample_rate: int) -> bytes:
        # sample_rate is kept for API symmetry and future packet metadata.
        _ = sample_rate
        x = self._to_mono_float(samples)
        if x.size == 0:
            return b""

        # mu-law companding into 8-bit transport values.
        mu = float(self.mu)
        companded = np.sign(x) * (np.log1p(mu * np.abs(x)) / np.log1p(mu))
        encoded_u8 = np.floor((companded + 1.0) * 127.5 + 0.5).astype(np.uint8)
        return encoded_u8.tobytes()


class OpusCodec(Codec):
    """
    Opus encoder wrapper.

    If `opuslib` is available, this uses a real Opus encoder.
    Otherwise it uses a demo fallback (zlib-compressed PCM) for local testing.
    """

    def __init__(
        self,
        logger: logging.Logger,
        bitrate: int = 16000,
        frame_duration_ms: int = 20,
        application: str = "voip",
    ):
        super().__init__(logger)
        if bitrate <= 0:
            raise ValueError("bitrate must be > 0")
        if frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be > 0")

        self.bitrate = int(bitrate)
        self.frame_duration_ms = int(frame_duration_ms)
        self.application = str(application).lower()
        self._encoder_cache: dict[tuple[int, int], object] = {}
        self._opuslib = None
        self._warned_demo_fallback = False

        try:
            import opuslib  # type: ignore

            self._opuslib = opuslib
        except Exception:
            self._opuslib = None

    def _get_or_create_encoder(self, sample_rate: int, channels: int):
        if self._opuslib is None:
            return None

        key = (int(sample_rate), int(channels))
        enc = self._encoder_cache.get(key)
        if enc is not None:
            return enc

        opuslib = self._opuslib
        app_map = {
            "voip": getattr(opuslib, "APPLICATION_VOIP", 2048),
            "audio": getattr(opuslib, "APPLICATION_AUDIO", 2049),
            "restricted_lowdelay": getattr(opuslib, "APPLICATION_RESTRICTED_LOWDELAY", 2051),
        }
        app = app_map.get(self.application, app_map["voip"])
        enc = opuslib.Encoder(int(sample_rate), int(channels), app)

        # Set bitrate if supported by installed opuslib flavor.
        try:
            enc.bitrate = self.bitrate
        except Exception:
            pass

        self._encoder_cache[key] = enc
        return enc

    def _demo_encode(self, pcm16_bytes: bytes) -> bytes:
        if not self._warned_demo_fallback:
            self.logger.warning("[Codec] opuslib not found, using demo zlib fallback for OpusCodec")
            self._warned_demo_fallback = True

        # Not real Opus, but useful for encode-only transport demonstration.
        payload = zlib.compress(pcm16_bytes, level=3)
        return b"OPUSDEMO" + payload

    def encode(self, samples: np.ndarray, sample_rate: int) -> bytes:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        x = self._to_mono_float(samples)
        if x.size == 0:
            return b""

        channels = 1
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        if frame_size <= 0:
            raise ValueError("invalid frame_size from sample_rate and frame_duration_ms")

        # Pad to whole Opus frames.
        rem = x.size % frame_size
        if rem != 0:
            x = np.pad(x, (0, frame_size - rem), mode="constant")

        pcm16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        pcm16_bytes = pcm16.tobytes()

        enc = self._get_or_create_encoder(sample_rate, channels)
        if enc is None:
            return self._demo_encode(pcm16_bytes)

        # Encode frame-by-frame; prefix each packet with uint16 length for demo transport.
        out = bytearray()
        frame_stride = frame_size * channels
        for i in range(0, pcm16.size, frame_stride):
            frame = pcm16[i:i + frame_stride]
            packet = enc.encode(frame.tobytes(), frame_size)
            plen = len(packet)
            if plen > 65535:
                raise ValueError("Opus packet too large for demo framing")
            out.extend(plen.to_bytes(2, byteorder="little", signed=False))
            out.extend(packet)

        return bytes(out)