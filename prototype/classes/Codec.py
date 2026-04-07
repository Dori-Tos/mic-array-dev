import numpy as np
import logging
import zlib
import socket
import struct


class Codec:
    """Base class for one-way (encode-only) codec demonstrations."""

    def __init__(
        self,
        logger: logging.Logger,
        remote_host: str | None = None,
        remote_port: int | None = None,
    ):
        self.logger = logger
        self.remote_host = remote_host
        self.remote_port = int(remote_port) if remote_port is not None else None
        self._udp_socket: socket.socket | None = None

    def configure_transport(self, remote_host: str, remote_port: int):
        if not remote_host:
            raise ValueError("remote_host must be non-empty")
        if int(remote_port) <= 0 or int(remote_port) > 65535:
            raise ValueError("remote_port must be in range 1..65535")
        self.remote_host = str(remote_host)
        self.remote_port = int(remote_port)

    def _ensure_udp_socket(self):
        if self._udp_socket is None:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_payload(self, payload: bytes):
        if not payload:
            return
        if not self.remote_host or self.remote_port is None:
            raise RuntimeError("Codec transport destination is not configured")
        self._ensure_udp_socket()
        self._udp_socket.sendto(payload, (self.remote_host, self.remote_port))

    def send_packets(self, packets: list[bytes]):
        for packet in packets:
            self.send_payload(packet)

    def close_transport(self):
        if self._udp_socket is not None:
            try:
                self._udp_socket.close()
            except Exception:
                pass
            self._udp_socket = None

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

    def __init__(
        self,
        logger: logging.Logger,
        mu: int = 255,
        remote_host: str | None = None,
        remote_port: int | None = None,
    ):
        super().__init__(logger, remote_host=remote_host, remote_port=remote_port)
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
        remote_host: str = "172.98.1.61",
        remote_port: int = 5004,
    ):
        super().__init__(logger, remote_host=remote_host, remote_port=remote_port)
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
        self._packet_seq = 0

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

    def _wrap_packet(self, opus_payload: bytes, sample_rate: int, frame_size: int) -> bytes:
        # 12-byte header: magic(4) + seq(uint32) + sample_rate(uint16) + frame_size(uint16)
        header = struct.pack("<4sIHH", b"OPUS", self._packet_seq, int(sample_rate), int(frame_size))
        self._packet_seq = (self._packet_seq + 1) & 0xFFFFFFFF
        return header + opus_payload

    def encode_packets(self, samples: np.ndarray, sample_rate: int) -> list[bytes]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        x = self._to_mono_float(samples)
        if x.size == 0:
            return []

        channels = 1
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        if frame_size <= 0:
            raise ValueError("invalid frame_size from sample_rate and frame_duration_ms")

        rem = x.size % frame_size
        if rem != 0:
            x = np.pad(x, (0, frame_size - rem), mode="constant")

        pcm16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
        enc = self._get_or_create_encoder(sample_rate, channels)

        if enc is None:
            demo_payload = self._demo_encode(pcm16.tobytes())
            return [self._wrap_packet(demo_payload, sample_rate, frame_size)]

        packets: list[bytes] = []
        frame_stride = frame_size * channels
        for i in range(0, pcm16.size, frame_stride):
            frame = pcm16[i:i + frame_stride]
            packet = enc.encode(frame.tobytes(), frame_size)
            packets.append(self._wrap_packet(packet, sample_rate, frame_size))
        return packets

    def encode(self, samples: np.ndarray, sample_rate: int) -> bytes:
        packets = self.encode_packets(samples, sample_rate)
        return b"".join(packets)