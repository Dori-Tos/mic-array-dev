import argparse
import socket
import struct
import threading
import zlib
from collections import deque

import numpy as np
import sounddevice as sd


def main():
    parser = argparse.ArgumentParser(description="Receive and play Opus UDP stream from Array_RealTime")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="Local IP to bind UDP socket")
    parser.add_argument("--port", type=int, default=5004, help="UDP port to listen on")
    parser.add_argument("--playback-rate", type=int, default=48000, help="Playback sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Playback channels")
    args = parser.parse_args()

    opuslib = None
    opus_available = True
    warned_missing_opus = False
    try:
        import opuslib as _opuslib  # type: ignore
        opuslib = _opuslib
    except Exception:
        opus_available = False
        print(
            "[Receiver] opuslib/native Opus library not available. "
            "Will only decode OPUSDEMO fallback packets."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    print(f"Listening on UDP {args.bind_ip}:{args.port}")

    decoder_cache: dict[int, object] = {}
    audio_fifo = deque()
    fifo_lock = threading.Lock()
    running = True

    def get_decoder(sample_rate: int):
        if opuslib is None:
            raise RuntimeError("Opus decoder is not available on this machine")
        dec = decoder_cache.get(sample_rate)
        if dec is None:
            dec = opuslib.Decoder(sample_rate, args.channels)
            decoder_cache[sample_rate] = dec
        return dec

    def receiver_loop():
        nonlocal running, warned_missing_opus
        while running:
            try:
                data, _addr = sock.recvfrom(4096)
                if len(data) < 12:
                    continue
                magic, _seq, sample_rate, frame_size = struct.unpack("<4sIHH", data[:12])
                if magic != b"OPUS":
                    continue

                payload = data[12:]
                if not payload:
                    continue

                if payload.startswith(b"OPUSDEMO"):
                    # Fallback payload if sender has no opuslib.
                    try:
                        raw_pcm = zlib.decompress(payload[len(b"OPUSDEMO"):])
                        pcm = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    except Exception:
                        continue
                else:
                    if not opus_available:
                        if not warned_missing_opus:
                            print(
                                "[Receiver] Received real Opus packets but decoder is unavailable. "
                                "Install native Opus runtime on Windows (libopus) or run sender in demo mode."
                            )
                            warned_missing_opus = True
                        continue
                    try:
                        dec = get_decoder(int(sample_rate))
                        decoded = dec.decode(payload, int(frame_size), False)
                        pcm = np.frombuffer(decoded, dtype=np.int16).astype(np.float32) / 32768.0
                    except Exception:
                        continue

                with fifo_lock:
                    audio_fifo.append(pcm)
                    # Keep bounded to avoid unlimited latency growth.
                    while len(audio_fifo) > 12:
                        audio_fifo.popleft()
            except Exception:
                continue

    recv_thread = threading.Thread(target=receiver_loop, daemon=True)
    recv_thread.start()

    def callback(outdata, frames, time_info, status):
        if status:
            print(f"[Output callback] {status}")

        chunk = np.zeros(frames, dtype=np.float32)
        write_idx = 0

        with fifo_lock:
            while write_idx < frames and len(audio_fifo) > 0:
                head = audio_fifo[0]
                take = min(frames - write_idx, head.size)
                if take > 0:
                    chunk[write_idx:write_idx + take] = head[:take]
                    write_idx += take
                    if take == head.size:
                        audio_fifo.popleft()
                    else:
                        audio_fifo[0] = head[take:]

        outdata[:, 0] = np.clip(chunk, -1.0, 1.0)

    stream = sd.OutputStream(
        samplerate=args.playback_rate,
        channels=args.channels,
        dtype="float32",
        callback=callback,
        latency="low",
    )

    print("Receiver started. Press Ctrl+C to stop.")
    try:
        with stream:
            while True:
                sd.sleep(200)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        sock.close()


if __name__ == "__main__":
    main()
    
    # Run via command line:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\tools\opus_receiver.py --port 5004
