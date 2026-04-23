from .Array import Array

import time
import numpy as np
import sounddevice as sd
import threading
import queue
import logging

from scipy import signal
from collections import deque
from math import gcd


_UNSET = object()


def _dbfs(value: float, floor: float = 1e-10) -> float:
    return 20.0 * np.log10(max(float(value), floor))


def apply_realtime_processing_chain(
    block: np.ndarray,
    beamformer,
    filters,
    agc,
    sample_rate: int,
    monitor_gain: float = 1.0,
    theta_deg: float | None = None,
    freeze_beamformer: bool = False,
    freeze_angle_deg: float | None = None,
    return_timing: bool = False,
) -> np.ndarray | tuple:
    """
    Apply the same beamformer/filter/AGC chain used by Array_RealTime.

    This helper allows offline protocols to reuse the exact processing order
    without duplicating implementation details.
    
    :param return_timing: If True, returns (processed_audio, timing_dict) with per-component timing in ms
    """
    import time
    
    timing = {
        'beamformer_ms': 0.0,
        'per_filter_ms': [],
        'agc_ms': 0.0,
        'total_ms': 0.0,
    }
    total_start = time.perf_counter()
    
    processed = np.asarray(block, dtype=np.float32)

    if beamformer is not None:
        # Optional freeze mode: lock steering to a fixed angle for all processed blocks.
        beam_start = time.perf_counter()
        if freeze_beamformer:
            if freeze_angle_deg is not None and hasattr(beamformer, "set_steering_angle"):
                beamformer.set_steering_angle(float(freeze_angle_deg))
            processed = beamformer.apply(processed)
        else:
            if theta_deg is None:
                processed = beamformer.apply(processed)
            else:
                processed = beamformer.apply(processed, theta_deg=theta_deg)
        timing['beamformer_ms'] = (time.perf_counter() - beam_start) * 1000.0

    if processed.ndim > 1:
        processed = np.squeeze(processed)
    processed = np.asarray(processed, dtype=np.float32).reshape(-1)

    # Filters - collect per-filter timing
    if filters:
        for i, filt in enumerate(filters or []):
            processed = np.asarray(filt.apply(processed), dtype=np.float32).reshape(-1)
            filter_name = filt.__class__.__name__
            filter_time = filt.last_process_time_ms if hasattr(filt, 'last_process_time_ms') else 0.0
            timing['per_filter_ms'].append({
                'index': i,
                'name': filter_name,
                'time_ms': filter_time
            })

    # AGC - Collect overall timing and per-stage if available
    if agc is not None:
        processed = np.asarray(agc.process(processed, sample_rate=sample_rate), dtype=np.float32).reshape(-1)
        agc_name = agc.__class__.__name__
        if hasattr(agc, 'last_process_time_ms'):
            timing['agc_ms'] = agc.last_process_time_ms
        if hasattr(agc, 'stage_process_times_ms') and agc.stage_process_times_ms:
            timing['agc_stages'] = agc.stage_process_times_ms

    if monitor_gain != 1.0:
        processed = processed * float(monitor_gain)

    timing['total_ms'] = (time.perf_counter() - total_start) * 1000.0
    
    if return_timing:
        return processed, timing
    return processed

class Array_RealTime(Array):    
    """
    Real-time audio processing array class that extends the base Array with real-time capabilities.
    
    :param logger: logging.Logger instance for logging messages.
    :param monitor_gain: Gain factor applied to the beamformed output for monitoring through speakers (default: 0.35).
    :param downsample_rate: If set, downsample the input audio to this rate for processing
        (e.g., 16000 Hz) to reduce computational load. The original sample rate is restored after processing.
    :param initial_silence_duration: Duration in seconds to silence output at startup while filters adapt (default: 2.0).
        Set to 0 to disable. Filters often need 1-3 seconds to stabilize before sound quality is good.
    """ 
    
    def __init__(self, *args, 
            logger: logging.Logger, 
            monitor_gain: float = 0.35, 
            output_mode: str = "local",
            output_boundary_fade_ms: float = 0.0,
            downsample_rate: int | None = None, 
            initial_silence_duration: float = 2.0, **kwargs
        ):
        
        super().__init__(*args, logger=logger, **kwargs)
        self._latest_block = None
        self._latest_per_mic = {}
        self._latest_doa = None
        self._latest_beamformed = None
        
        self.monitor_gain = monitor_gain
        self.output_mode = str(output_mode).strip().lower()
        if self.output_mode not in ("local", "codec"):
            raise ValueError("output_mode must be 'local' or 'codec'")
        self.downsample_rate = downsample_rate  # Downsample to this rate (e.g., 16000 Hz)
        self.initial_silence_duration = float(initial_silence_duration)  # Silence duration (sec) at startup
        self._stream_start_time = None  # Track when audio stream started
        self._codec_stream_active = False
        self._output_playback_rate = self.sampling_rate
        self._output_stream = None
        self._output_status_state = {"last_print": 0.0}
        self._adapt_log_state = {
            "started_logged": False,
            "last_print": 0.0,
            "setup_logged": False,
        }
        self._alert_state = {
            "beamformer_peak_last": 0.0,
            "post_agc_peak_last": 0.0,
            "final_peak_last": 0.0,
        }
        self._warn_interval_s = 1.0
        self._post_agc_warn_threshold = 0.98
        self._final_warn_threshold = 0.98
        self._output_fifo = deque()
        self._output_fifo_lock = threading.Lock()
        self._output_current_chunk = np.zeros(0, dtype=np.float32)
        self._output_buffered_samples = 0
        self._output_max_buffer_samples = int(self._output_playback_rate * 1.0)
        self._output_prev_sample = 0.0
        fade_ms = max(0.0, float(output_boundary_fade_ms))
        self._output_fade_samples = int(self._output_playback_rate * (fade_ms / 1000.0))

        # Cache for downsampling internals so output-size probing is not repeated every block.
        self._downsample_cache_rate = None
        self._downsample_cache_in_len = None
        self._downsample_cache_channels = None
        self._downsample_scratch = None
        
        # Audio processing queue and thread
        # Larger queue to handle slow processing without dropping blocks
        self._audio_queue = queue.Queue(maxsize=20)
        self._queue_catchup_trigger = 6
        self._queue_keep_depth = 2
        self._skip_doa_queue_threshold = 4
        self._last_catchup_log = 0.0
        self._processing_thread = None
        self._processing_stop_event = threading.Event()
        self._processing_input_channel = 0
        self._last_input_blocksize = 0

        # Measurement tracking for side-door data extraction (no buffering).
        self._measurement_active = False
        self._measurement_started_at = None
        self._measurement_input_sum_squares = 0.0
        self._measurement_input_samples = 0
        self._measurement_input_peak = 0.0
        self._measurement_input_allch_sum_squares = 0.0
        self._measurement_input_allch_samples = 0
        self._measurement_output_sum_squares = 0.0
        self._measurement_output_samples = 0
        self._measurement_output_peak = 0.0
        self._measurement_block_count = 0
        self._measurement_doa_latest = None
        self._measurement_doa_conf_latest = None
        self._measurement_lock = threading.Lock()

    def start_side_door_measurement(self):
        """Start accumulating block-wise RMS/DOA statistics without buffering audio."""
        with self._measurement_lock:
            self._measurement_active = True
            self._measurement_started_at = time.monotonic()
            self._measurement_input_sum_squares = 0.0
            self._measurement_input_samples = 0
            self._measurement_input_peak = 0.0
            self._measurement_input_allch_sum_squares = 0.0
            self._measurement_input_allch_samples = 0
            self._measurement_output_sum_squares = 0.0
            self._measurement_output_samples = 0
            self._measurement_output_peak = 0.0
            self._measurement_block_count = 0
            self._measurement_doa_latest = None
            self._measurement_doa_conf_latest = None

    def stop_side_door_measurement(self):
        """Stop the live measurement accumulator without clearing the last snapshot."""
        with self._measurement_lock:
            self._measurement_active = False

    def get_side_door_measurement_snapshot(self, reset: bool = False) -> dict:
        """Return the current accumulated measurement values as a non-buffered snapshot."""
        with self._measurement_lock:
            input_rms = float(np.sqrt(self._measurement_input_sum_squares / self._measurement_input_samples)) if self._measurement_input_samples > 0 else 0.0
            input_allch_rms = float(np.sqrt(self._measurement_input_allch_sum_squares / self._measurement_input_allch_samples)) if self._measurement_input_allch_samples > 0 else 0.0
            output_rms = float(np.sqrt(self._measurement_output_sum_squares / self._measurement_output_samples)) if self._measurement_output_samples > 0 else 0.0
            snapshot = {
                "active": bool(self._measurement_active),
                "started_at": self._measurement_started_at,
                "elapsed_s": (time.monotonic() - self._measurement_started_at) if self._measurement_started_at is not None else 0.0,
                "block_count": int(self._measurement_block_count),
                "input_samples": int(self._measurement_input_samples),
                "input_allch_samples": int(self._measurement_input_allch_samples),
                "output_samples": int(self._measurement_output_samples),
                "input_rms": input_rms,
                "input_rms_dbfs": _dbfs(input_rms),
                "input_allch_rms": input_allch_rms,
                "input_allch_rms_dbfs": _dbfs(input_allch_rms),
                "input_peak": float(self._measurement_input_peak),
                "input_peak_dbfs": _dbfs(self._measurement_input_peak),
                "output_rms": output_rms,
                "output_rms_dbfs": _dbfs(output_rms),
                "output_peak": float(self._measurement_output_peak),
                "output_peak_dbfs": _dbfs(self._measurement_output_peak),
                "doa_deg": self._measurement_doa_latest,
                "doa_conf_db": self._measurement_doa_conf_latest,
            }
            if reset:
                self._measurement_started_at = time.monotonic() if self._measurement_active else None
                self._measurement_input_sum_squares = 0.0
                self._measurement_input_samples = 0
                self._measurement_input_peak = 0.0
                self._measurement_input_allch_sum_squares = 0.0
                self._measurement_input_allch_samples = 0
                self._measurement_output_sum_squares = 0.0
                self._measurement_output_samples = 0
                self._measurement_output_peak = 0.0
                self._measurement_block_count = 0
                self._measurement_doa_latest = None
                self._measurement_doa_conf_latest = None
            return snapshot

    def _accumulate_side_door_measurement(self, input_mono: np.ndarray, input_allch: np.ndarray, output_mono: np.ndarray, doa_value, doa_conf_db):
        if not self._measurement_active:
            return

        with self._measurement_lock:
            if not self._measurement_active:
                return

            input_mono = np.asarray(input_mono, dtype=np.float32).reshape(-1)
            output_mono = np.asarray(output_mono, dtype=np.float32).reshape(-1)

            if input_mono.size > 0:
                self._measurement_input_sum_squares += float(np.sum(input_mono * input_mono))
                self._measurement_input_samples += int(input_mono.size)
                self._measurement_input_peak = max(self._measurement_input_peak, float(np.max(np.abs(input_mono))))

            input_allch = np.asarray(input_allch, dtype=np.float32)
            if input_allch.ndim == 2 and input_allch.size > 0:
                self._measurement_input_allch_sum_squares += float(np.sum(input_allch * input_allch))
                self._measurement_input_allch_samples += int(input_allch.size)
            elif input_allch.ndim == 1 and input_allch.size > 0:
                self._measurement_input_allch_sum_squares += float(np.sum(input_allch * input_allch))
                self._measurement_input_allch_samples += int(input_allch.size)

            if output_mono.size > 0:
                self._measurement_output_sum_squares += float(np.sum(output_mono * output_mono))
                self._measurement_output_samples += int(output_mono.size)
                self._measurement_output_peak = max(self._measurement_output_peak, float(np.max(np.abs(output_mono))))

            self._measurement_block_count += 1
            self._measurement_doa_latest = doa_value
            self._measurement_doa_conf_latest = doa_conf_db

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start_realtime(self, blocksize: int = 0):
        """ 
        Start real-time audio processing. This will open the audio stream and begin calling the callback function.
        The callback will store the latest audio block, per-microphone samples, and DOA estimates for retrieval.
        """
        if self._is_running:
            self.logger.warning("Already running, ignoring start_realtime() call")
            return
        if not self.mic_list:
            raise ValueError("Cannot start stream without microphones")

        self.logger.info("Starting realtime audio processing")
        self._last_input_blocksize = int(blocksize)
        self._processing_stop_event.clear()
        self._processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self._processing_thread.start()
        self.logger.debug("Processing thread started")

        self._stream = sd.InputStream(
            samplerate=self.sampling_rate,
            channels=len(self.mic_list),
            dtype='int16',
            device=self.device_index,
            callback=self._audio_callback,
            latency='low',
            blocksize=blocksize
        )
        self._stream.start()
        self._is_running = True
        self.logger.info(f"Audio stream started: {self.sampling_rate}Hz, {len(self.mic_list)} channels, blocksize={blocksize}")

    def reconfigure_runtime(
        self,
        *,
        mic_list=_UNSET,
        doa_estimator=_UNSET,
        beamformer=_UNSET,
        echo_canceller=_UNSET,
        filters=_UNSET,
        agc=_UNSET,
        processing_input_channel: int | object = _UNSET,
        restart_if_running: bool = True,
        blocksize: int | None = None,
        restore_output_monitoring: bool = True,
    ):
        """
        Reconfigure processing topology while preserving instance state.

        This supports switching between full array beamforming and single-mic
        processing chains at runtime. If the stream is running and topology
        changes are requested, the stream is safely restarted.
        """
        was_running = self._is_running
        output_was_running = self._output_stream is not None or self._codec_stream_active

        requested_blocksize = self._last_input_blocksize if blocksize is None else int(blocksize)

        if mic_list is not _UNSET:
            self.mic_list = mic_list
        if doa_estimator is not _UNSET:
            self.doa_estimator = doa_estimator
        if beamformer is not _UNSET:
            self.beamformer = beamformer
        if echo_canceller is not _UNSET:
            self.echo_canceller = echo_canceller
        if filters is not _UNSET:
            self.filters = filters
        if agc is not _UNSET:
            self.agc = agc
        if processing_input_channel is not _UNSET:
            self._processing_input_channel = int(processing_input_channel)

        # Clear cached output/state to avoid stale blocks after topology switch.
        with self._lock:
            self._latest_block = None
            self._latest_per_mic = {}
            self._latest_doa = None
            self._latest_beamformed = None
        with self._output_fifo_lock:
            self._output_fifo.clear()
            self._output_current_chunk = np.zeros(0, dtype=np.float32)
            self._output_buffered_samples = 0

        if was_running and restart_if_running:
            self.stop_realtime()
            self.start_realtime(blocksize=requested_blocksize)
            if restore_output_monitoring and output_was_running:
                self.start_output_monitoring()

    def _extract_processing_mono(self, block: np.ndarray) -> np.ndarray:
        """Fallback mono extraction when no beamformer is active."""
        if block.ndim == 1:
            return np.asarray(block, dtype=np.float32).reshape(-1)
        if block.ndim != 2 or block.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)

        ch = int(self._processing_input_channel)
        ch = max(0, min(ch, block.shape[1] - 1))
        return np.asarray(block[:, ch], dtype=np.float32).reshape(-1)

    def stop_realtime(self):
        if not self._is_running:
            self.logger.warning("Not running, ignoring stop_realtime() call")
            return

        self.logger.info("Stopping realtime audio processing")
        self.stop_output_monitoring()
        self._processing_stop_event.set()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.logger.debug("Audio stream stopped and closed")
        
        if self._processing_thread is not None:
            self.logger.debug("Waiting for processing thread to join...")
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
            self.logger.debug("Processing thread joined")
        
        self._is_running = False
        self.logger.info("Realtime audio processing stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Minimal callback that just captures audio data and queues it for processing.
        This keeps the callback fast to avoid buffer overflows.
        
        Audio is normalized from int16 (±32768 from sounddevice) to float32 (±1.0) at capture.
        This ensures all downstream components receive consistent normalized audio.
        """
        if status:
            self.logger.warning(f"[Audio callback] {status}")
        
        # Normalize int16 capture (±32768) to float32 (±1.0) immediately
        block = np.copy(indata).astype(np.float32) / 32768.0
        try:
            self._audio_queue.put_nowait(block)
        except queue.Full:
            self.logger.debug("Queue full, dropping oldest block")
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(block)
            except queue.Empty:
                pass

    def _downsample_block(self, block: np.ndarray) -> np.ndarray:
        """
        Downsample audio block using vectorized polyphase resampling.
        
        Parameters:
        - block: Input audio block with shape (n_samples, n_channels)
        
        Output:
        - downsampled block with shape (resampled_samples, n_channels)
        """
        if self.downsample_rate is None or self.downsample_rate >= self.sampling_rate:
            return block

        in_len, n_channels = block.shape
        cache_valid = (
            self._downsample_cache_rate == self.sampling_rate
            and self._downsample_cache_in_len == in_len
            and self._downsample_cache_channels == n_channels
            and self._downsample_scratch is not None
        )

        if not cache_valid:
            out_probe = signal.resample_poly(block.astype(np.float32), self.downsample_rate, self.sampling_rate, axis=0)
            self._downsample_scratch = np.empty_like(out_probe, dtype=np.float32)

            self._downsample_cache_rate = self.sampling_rate
            self._downsample_cache_in_len = in_len
            self._downsample_cache_channels = n_channels

        downsampled = signal.resample_poly(block.astype(np.float32), self.downsample_rate, self.sampling_rate, axis=0)
        if downsampled.shape != self._downsample_scratch.shape:
            self._downsample_scratch = np.empty_like(downsampled, dtype=np.float32)
            self._downsample_cache_in_len = in_len
            self._downsample_cache_channels = n_channels

        self._downsample_scratch[:, :] = downsampled

        # Return a copy because the scratch buffer is reused for the next block.
        return self._downsample_scratch.copy()

    def _resample_to_playback_rate(self, mono: np.ndarray, in_rate: int) -> np.ndarray:
        """Resample mono chunk to output playback rate when processing is downsampled."""
        if in_rate <= 0 or in_rate == self._output_playback_rate:
            return mono

        from scipy import signal

        g = gcd(int(self._output_playback_rate), int(in_rate))
        up = int(self._output_playback_rate // g)
        down = int(in_rate // g)
        return signal.resample_poly(mono, up, down).astype(np.float32, copy=False)

    def _send_codec_chunk(self, mono_out: np.ndarray, sample_rate: int):
        if not self._codec_stream_active:
            return
        if self.codec is None:
            return
        try:
            encode_packets = getattr(self.codec, "encode_packets", None)
            if callable(encode_packets):
                packets = encode_packets(mono_out, sample_rate)
                self.codec.send_packets(packets)
            else:
                payload = self.codec.encode(mono_out, sample_rate)
                self.codec.send_payload(payload)
        except Exception as codec_err:
            self.logger.warning(f"[Codec Output] {type(codec_err).__name__}: {codec_err}")

    def _process_audio_thread(self):
        """
        Processing thread that handles computationally expensive operations:
        DOA estimation and beamforming. Runs independently from the audio callback.
        """
        self.logger.info(f"Processing thread started (downsample_rate: {self.downsample_rate})")
        block_count = 0
        start_time = time.monotonic()
        
        # Timing accumulators for performance statistics
        timing_accumulators = {
            'doa_ms': [],
            'beamforming_ms': [],
            'filters_ms': [],
            'agc_ms': [],
            'resampling_ms': [],
        }

        while not self._processing_stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.1)
                block_count += 1
                queue_size = self._audio_queue.qsize()
            except queue.Empty:
                self.logger.debug("[Processing] Queue timeout, no block received")
                continue

            if queue_size > self._queue_catchup_trigger:
                dropped_stale = 0
                while self._audio_queue.qsize() > self._queue_keep_depth:
                    try:
                        block = self._audio_queue.get_nowait()
                        dropped_stale += 1
                    except queue.Empty:
                        break
                queue_size = self._audio_queue.qsize()
                now_log = time.monotonic()
                if dropped_stale > 0 and (now_log - self._last_catchup_log) > 0.5:
                    self.logger.debug(f"[Catch-up] Dropped {dropped_stale} stale queued blocks, queue_size now {queue_size}")
                    self._last_catchup_log = now_log

            original_rate = self.sampling_rate
            if self.downsample_rate is not None and self.downsample_rate < self.sampling_rate:
                block = self._downsample_block(block)
                self.sampling_rate = self.downsample_rate
                if self.beamformer is not None and hasattr(self.beamformer, "sample_rate"):
                    self.beamformer.sample_rate = int(self.downsample_rate)
                doa_bf = getattr(self.doa_estimator, "beamformer", None)
                if doa_bf is not None and hasattr(doa_bf, "sample_rate"):
                    doa_bf.sample_rate = int(self.downsample_rate)
                self.logger.debug(f"[Processing] Downsampled to {self.downsample_rate} Hz, new block shape: {block.shape}")

            per_mic = {}
            for idx, mic in enumerate(self.mic_list):
                channel_index = mic.channel_number if mic.channel_number is not None else idx
                if 0 <= channel_index < block.shape[1]:
                    per_mic[mic.channel_number] = block[:, channel_index].copy()

            doa_value = None
            start_doa = time.monotonic()
            should_skip_doa = queue_size > self._skip_doa_queue_threshold and self._latest_doa is not None
            doa_fn = getattr(self.doa_estimator, "estimate_doa", None)
            if should_skip_doa:
                doa_value = self._latest_doa
                self.logger.debug(f"[Processing] Skipping DOA update due to backlog (queue_size={queue_size}), reusing {doa_value:.1f}°")
            elif callable(doa_fn):
                try:
                    doa_value = doa_fn(block)
                    # Use DOAEstimator's own computation time measurement (excludes early returns)
                    if hasattr(self.doa_estimator, 'last_process_time_ms'):
                        timing_accumulators['doa_ms'].append(self.doa_estimator.last_process_time_ms)
                    else:
                        elapsed_doa = time.monotonic() - start_doa
                        timing_accumulators['doa_ms'].append(elapsed_doa * 1000.0)
                except Exception as e:
                    elapsed_doa = time.monotonic() - start_doa
                    timing_accumulators['doa_ms'].append(elapsed_doa * 1000.0)
                    self.logger.error(f"[DOA Estimation Error] {type(e).__name__}: {e} (after {elapsed_doa*1000:.2f}ms)")
                    doa_value = None

            beamformed_value = None
            mono_out = np.zeros(0, dtype=np.float32)
            start_bf = time.monotonic()
            try:
                if self.beamformer is not None and isinstance(doa_value, (int, float, np.integer, np.floating)):
                    if hasattr(self.beamformer, "set_steering_angle"):
                        self.beamformer.set_steering_angle(float(doa_value))

                apply_fn = getattr(self.beamformer, "apply", None)
                if callable(apply_fn):
                    beamformed = apply_fn(block)
                    if beamformed is not None:
                        beamformed_arr = np.asarray(beamformed, dtype=np.float32).reshape(-1)
                        if beamformed_arr.size > 0:
                            beamformed_value = beamformed_arr.copy()
                            mono_out = beamformed_arr
                            # Use beamformer's own timing measurement
                            if hasattr(self.beamformer, 'last_process_time_ms'):
                                timing_accumulators['beamforming_ms'].append(self.beamformer.last_process_time_ms)
                            else:
                                elapsed_bf = time.monotonic() - start_bf
                                timing_accumulators['beamforming_ms'].append(elapsed_bf * 1000.0)
                else:
                    elapsed_bf = time.monotonic() - start_bf
                    mono_out = self._extract_processing_mono(block)
            except Exception as e:
                elapsed_bf = time.monotonic() - start_bf
                timing_accumulators['beamforming_ms'].append(elapsed_bf * 1000.0)
                self.logger.error(f"[Beamforming Error] {type(e).__name__}: {e} (after {elapsed_bf*1000:.2f}ms)")
                beamformed_value = None

            if mono_out.size > 0:
                now_alert = time.monotonic()
                peak = float(np.max(np.abs(mono_out)))
                if self.beamformer is not None and peak > 1.0 and (now_alert - self._alert_state["beamformer_peak_last"]) >= self._warn_interval_s:
                    self.logger.warning(
                        f"[Peak Alert] Beamformer output peak={peak:.3f} (>1.0). Upstream energy may be too high for headroom."
                    )
                    self._alert_state["beamformer_peak_last"] = now_alert

                for filt in self.filters or []:
                    if callable(getattr(filt, "apply", None)):
                        try:
                            mono_out = np.asarray(filt.apply(mono_out), dtype=np.float32).reshape(-1)
                            # Use filter's own timing measurement
                            if hasattr(filt, 'last_process_time_ms'):
                                timing_accumulators['filters_ms'].append(filt.last_process_time_ms)
                        except Exception as filter_err:
                            self.logger.warning(f"[Filter] {type(filter_err).__name__}: {filter_err}")

                output_sample_rate = self.downsample_rate if self.downsample_rate is not None else self.sampling_rate
                if self.agc is not None:
                    agc_start = time.monotonic()
                    mono_out = np.asarray(self.agc.process(mono_out, sample_rate=output_sample_rate), dtype=np.float32).reshape(-1)
                    agc_elapsed = time.monotonic() - agc_start
                    if hasattr(self.agc, 'last_process_time_ms'):
                        timing_accumulators['agc_ms'].append(self.agc.last_process_time_ms)
                    else:
                        timing_accumulators['agc_ms'].append(agc_elapsed * 1000.0)

                post_agc_peak = float(np.max(np.abs(mono_out))) if mono_out.size > 0 else 0.0
                if post_agc_peak >= self._post_agc_warn_threshold and (now_alert - self._alert_state["post_agc_peak_last"]) >= self._warn_interval_s:
                    self.logger.warning(
                        f"[Peak Alert] Post-AGC peak={post_agc_peak:.3f} (threshold {self._post_agc_warn_threshold:.2f}). "
                        "Dynamics stage may be near saturation."
                    )
                    self._alert_state["post_agc_peak_last"] = now_alert

                mono_out = mono_out * self.monitor_gain
                final_peak = float(np.max(np.abs(mono_out))) if mono_out.size > 0 else 0.0
                if final_peak >= self._final_warn_threshold and (now_alert - self._alert_state["final_peak_last"]) >= self._warn_interval_s:
                    self.logger.warning(
                        f"[Peak Alert] Final monitor peak={final_peak:.3f} (threshold {self._final_warn_threshold:.2f}). "
                        "Output callback clipping risk is high."
                    )
                    self._alert_state["final_peak_last"] = now_alert

                processing_rate = self.downsample_rate if self.downsample_rate is not None else self.sampling_rate
                resample_start = time.monotonic()
                mono_out = self._resample_to_playback_rate(mono_out, int(processing_rate))
                resample_elapsed = time.monotonic() - resample_start
                if resample_elapsed > 0.0001:  # Only accumulate if actually resampled
                    timing_accumulators['resampling_ms'].append(resample_elapsed * 1000.0)

                self._accumulate_side_door_measurement(
                    input_mono=self._extract_processing_mono(block),
                    input_allch=block,
                    output_mono=mono_out,
                    doa_value=doa_value,
                    doa_conf_db=getattr(self.doa_estimator, "latest_confidence_db", None),
                )

                if self.output_mode == "codec":
                    self._send_codec_chunk(mono_out, int(processing_rate))
                else:
                    with self._output_fifo_lock:
                        self._output_fifo.append(mono_out)
                        self._output_buffered_samples += mono_out.size
                        while self._output_buffered_samples > self._output_max_buffer_samples and len(self._output_fifo) > 0:
                            old = self._output_fifo.popleft()
                            self._output_buffered_samples -= old.size

            if self.sampling_rate != original_rate:
                self.sampling_rate = original_rate
                if self.beamformer is not None and hasattr(self.beamformer, "sample_rate"):
                    self.beamformer.sample_rate = int(original_rate)
                doa_bf = getattr(self.doa_estimator, "beamformer", None)
                if doa_bf is not None and hasattr(doa_bf, "sample_rate"):
                    doa_bf.sample_rate = int(original_rate)

            total_time = time.monotonic() - start_doa
            with self._lock:
                self._latest_block = block
                self._latest_per_mic = per_mic
                self._latest_doa = doa_value
                self._latest_beamformed = beamformed_value

            if block_count % 10 == 0:
                # Calculate average delays from accumulated timings
                avg_delays = {
                    'doa_ms': np.mean(timing_accumulators['doa_ms']) if timing_accumulators['doa_ms'] else 0.0,
                    'beamforming_ms': np.mean(timing_accumulators['beamforming_ms']) if timing_accumulators['beamforming_ms'] else 0.0,
                    'filters_ms': np.mean(timing_accumulators['filters_ms']) if timing_accumulators['filters_ms'] else 0.0,
                    'agc_ms': np.mean(timing_accumulators['agc_ms']) if timing_accumulators['agc_ms'] else 0.0,
                    'resampling_ms': np.mean(timing_accumulators['resampling_ms']) if timing_accumulators['resampling_ms'] else 0.0,
                }
                
                total_processing_delay = sum(avg_delays.values())
                
                delay_breakdown = " | ".join(filter(None, [
                    f"DOA: {avg_delays['doa_ms']:.2f}ms" if avg_delays['doa_ms'] > 0 else None,
                    f"Beamforming: {avg_delays['beamforming_ms']:.2f}ms" if avg_delays['beamforming_ms'] > 0 else None,
                    f"Filters: {avg_delays['filters_ms']:.2f}ms" if avg_delays['filters_ms'] > 0 else None,
                    f"AGC: {avg_delays['agc_ms']:.2f}ms" if avg_delays['agc_ms'] > 0 else None,
                    f"Resampling: {avg_delays['resampling_ms']:.2f}ms" if avg_delays['resampling_ms'] > 0 else None,
                ]))
                
                elapsed_total = time.monotonic() - start_time
                blocks_per_sec = block_count / elapsed_total if elapsed_total > 0 else 0
                
                self.logger.debug(
                    f"[Timing Summary] Total processing delay: {total_processing_delay:.2f}ms | "
                    f"{delay_breakdown} | "
                    f"Overall: {block_count} blocks in {elapsed_total:.1f}s ({blocks_per_sec:.1f} blocks/sec)"
                )
                
                # Clear accumulators for next window
                timing_accumulators = {
                    'doa_ms': [],
                    'beamforming_ms': [],
                    'filters_ms': [],
                    'agc_ms': [],
                    'resampling_ms': [],
                }

        total_time = time.monotonic() - start_time
        self.logger.info(f"Processing thread stopped after {block_count} blocks in {total_time:.1f}s")

    def get_latest_block(self) -> np.ndarray | None:
        """
        Get the latest audio block received from the stream. This will return a copy of the data to ensure thread safety.
        """
        with self._lock:
            if self._latest_block is None:
                return None
            return self._latest_block.copy()

    def get_latest_doa(self):
        """
        Get the latest DOA estimate. This will return a copy of the data to ensure thread safety.
        """
        with self._lock:
            if self._latest_doa is None:
                return None
            if isinstance(self._latest_doa, np.ndarray):
                return self._latest_doa.copy()
            if isinstance(self._latest_doa, (int, float, np.integer, np.floating)):
                return self._latest_doa
            if hasattr(self._latest_doa, 'copy'):
                return self._latest_doa.copy()
            return self._latest_doa

    def get_latest_beamformed(self) -> np.ndarray | None:
        """
        Get the latest beamformed mono block.
        """
        with self._lock:
            if self._latest_beamformed is None:
                return None
            return self._latest_beamformed.copy()

    def get_side_door_measurement_stats(self) -> dict:
        """Get the current live measurement statistics without copying audio buffers."""
        return self.get_side_door_measurement_snapshot(reset=False)

    def start_output_monitoring(self, blocksize: int = 0):
        """
        Start monitoring the beamformed audio output through the speaker.
        This opens an output stream and plays the latest beamformed audio.
        """
        if self.output_mode == "codec":
            if self.codec is None:
                raise ValueError("Codec output mode selected but no codec instance is configured")
            self._codec_stream_active = True
            self._stream_start_time = time.monotonic()
            target_host = getattr(self.codec, "remote_host", None)
            target_port = getattr(self.codec, "remote_port", None)
            self.logger.info(f"Codec output monitoring started -> {target_host}:{target_port}")
            return

        if self._output_stream is not None:
            return

        self._stream_start_time = time.monotonic()  # Start timer for initial silence
        self._adapt_log_state = {
            "started_logged": False,
            "last_print": 0.0,
            "setup_logged": False,
        }
        output_sample_rate = self._output_playback_rate
        with self._output_fifo_lock:
            self._output_fifo.clear()
            self._output_current_chunk = np.zeros(0, dtype=np.float32)
            self._output_buffered_samples = 0
        self._output_prev_sample = 0.0

        stream_kwargs = {
            "samplerate": output_sample_rate,
            "channels": 1,
            "dtype": "float32",
            "latency": "low",
            "blocksize": blocksize,
            "callback": self._output_callback,
        }

        # Prefer configured/default output device, but recover automatically when
        # PortAudio default output is invalid (common on Linux/RPi headless setups).
        try:
            default_device = sd.default.device
            if isinstance(default_device, (tuple, list)) and len(default_device) >= 2:
                default_out = default_device[1]
                if isinstance(default_out, (int, np.integer)) and int(default_out) >= 0:
                    stream_kwargs["device"] = int(default_out)
        except Exception:
            pass

        try:
            self._output_stream = sd.OutputStream(**stream_kwargs)
            self._output_stream.start()
            return
        except Exception as open_err:
            self.logger.warning(f"[Output] Failed to open default output device: {open_err}")

        # Fallback: probe all output-capable devices and pick the first valid one.
        try:
            devices = sd.query_devices()
            candidates = [
                idx for idx, info in enumerate(devices)
                if int(info.get("max_output_channels", 0)) > 0
            ]
            if not candidates:
                raise RuntimeError("No output-capable audio devices found")

            selected_device = None
            for idx in candidates:
                try:
                    sd.check_output_settings(
                        device=idx,
                        channels=1,
                        dtype="float32",
                        samplerate=output_sample_rate,
                    )
                    selected_device = idx
                    break
                except Exception:
                    continue

            if selected_device is None:
                selected_device = candidates[0]

            stream_kwargs["device"] = int(selected_device)
            self.logger.info(f"[Output] Using fallback output device index {selected_device}")
            self._output_stream = sd.OutputStream(**stream_kwargs)
            self._output_stream.start()
        except Exception as fallback_err:
            raise RuntimeError(f"Unable to open any output audio device: {fallback_err}") from fallback_err

    def stop_output_monitoring(self):
        """
        Stop the output monitoring stream.
        """
        if self.output_mode == "codec":
            self._codec_stream_active = False
            close_transport = getattr(self.codec, "close_transport", None)
            if callable(close_transport):
                try:
                    close_transport()
                except Exception:
                    pass
            return

        if self._output_stream is None:
            return
        
        self._output_stream.stop()
        self._output_stream.close()
        self._output_stream = None
        with self._output_fifo_lock:
            self._output_fifo.clear()
            self._output_current_chunk = np.zeros(0, dtype=np.float32)
            self._output_buffered_samples = 0
        self._output_prev_sample = 0.0
        self._adapt_log_state = {
            "started_logged": False,
            "last_print": 0.0,
            "setup_logged": False,
        }

    def _output_callback(self, outdata, frames, time_info, status):
        """
        Callback for the output stream. Retrieves the latest beamformed audio
        and outputs it to the speaker.
        """
        if status:
            now = time.monotonic()
            if (now - self._output_status_state["last_print"]) >= 2.0:
                self.logger.warning(f"[Output callback] {status}")
                self._output_status_state["last_print"] = now

        chunk = np.zeros(frames, dtype=np.float32)
        write_idx = 0

        while write_idx < frames:
            if self._output_current_chunk.size == 0:
                with self._output_fifo_lock:
                    if len(self._output_fifo) > 0:
                        self._output_current_chunk = self._output_fifo.popleft()
                        self._output_buffered_samples -= self._output_current_chunk.size
                    else:
                        break

            remaining = frames - write_idx
            take = min(remaining, self._output_current_chunk.size)
            if take > 0:
                chunk[write_idx:write_idx + take] = self._output_current_chunk[:take]
                self._output_current_chunk = self._output_current_chunk[take:]
                write_idx += take

        # Apply initial silence period for filter adaptation
        if self._stream_start_time is not None and self.initial_silence_duration > 0:
            elapsed = time.monotonic() - self._stream_start_time
            if elapsed < self.initial_silence_duration:
                chunk[:] = 0.0  # Silence during adaptation period
                remaining = self.initial_silence_duration - elapsed
                now = time.monotonic()
                if not self._adapt_log_state["started_logged"]:
                    self.logger.info(f"[Output] Adapting filters... {remaining:.1f}s remaining")
                    self._adapt_log_state["started_logged"] = True
                    self._adapt_log_state["last_print"] = now
                elif (now - float(self._adapt_log_state["last_print"])) >= 1.0:
                    self.logger.info(f"[Output] Adapting filters... {remaining:.1f}s remaining")
                    self._adapt_log_state["last_print"] = now
            elif not self._adapt_log_state["setup_logged"]:
                self.logger.info("[Output] Filters are setup")
                self._adapt_log_state["setup_logged"] = True

        # Optional boundary fade (disabled by default) to smooth hard chunk joins.
        # Keeping this off avoids introducing periodic artifacts at block boundaries.
        if chunk.size > 0:
            fade_n = min(self._output_fade_samples, chunk.size)
            if fade_n > 0:
                hann_window = np.hanning(fade_n * 2)[:fade_n].astype(np.float32)
                hann_fade = 1.0 - hann_window
                start = float(self._output_prev_sample)
                end = float(chunk[0])
                linear = np.linspace(start, end, num=fade_n, endpoint=False, dtype=np.float32)
                chunk[:fade_n] = linear * hann_fade + chunk[:fade_n] * (1.0 - hann_fade)
            self._output_prev_sample = float(chunk[-1])
        
        outdata[:, 0] = np.clip(chunk, -1.0, 1.0)