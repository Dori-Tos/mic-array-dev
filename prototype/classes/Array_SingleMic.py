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

class Array_RealTime(Array):
    """
    Real-time audio processing array class that extends the base Array with real-time capabilities.
    
    :param logger: logging.Logger instance for logging messages.
    :param monitor_gain: Gain factor applied to the beamformed output for monitoring through speakers (default: 0.35).
    :param downsample_rate: If set, downsample the input audio to this rate for processing
        (e.g., 16000 Hz) to reduce computational load. The original sample rate is restored after processing.
    """ 
    
    def __init__(self, *args, logger: logging.Logger, monitor_gain: float = 0.35, downsample_rate: int | None = None, **kwargs):
        super().__init__(*args, logger=logger, **kwargs)
        self._latest_block = None
        self._latest_per_mic = {}
        self._latest_doa = None
        self._latest_beamformed = None
        
        self.monitor_gain = monitor_gain
        self.downsample_rate = downsample_rate  # Downsample to this rate (e.g., 16000 Hz)
        self._output_playback_rate = self.sampling_rate
        self._output_stream = None
        self._output_status_state = {"last_print": 0.0}
        self._output_fifo = deque()
        self._output_fifo_lock = threading.Lock()
        self._output_current_chunk = np.zeros(0, dtype=np.float32)
        self._output_buffered_samples = 0
        self._output_max_buffer_samples = int(self._output_playback_rate * 1.0)

        # Cache for downsampling internals so output-size probing is not repeated every block.
        self._downsample_cache_rate = None
        self._downsample_cache_factor = None
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
        """
        if status:
            self.logger.warning(f"[Audio callback] {status}")
        
        block = np.copy(indata)
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
        Downsample audio block using scipy.signal.decimate.
        
        Parameters:
        - block: Input audio block with shape (n_samples, n_channels)
        
        Output:
        - downsampled block with shape (n_samples//decimation_factor, n_channels)
        """
        if self.downsample_rate is None or self.downsample_rate >= self.sampling_rate:
            return block
        
        decimation_factor = self.sampling_rate // self.downsample_rate
        if decimation_factor <= 1:
            return block

        in_len, n_channels = block.shape
        cache_valid = (
            self._downsample_cache_rate == self.sampling_rate
            and self._downsample_cache_factor == decimation_factor
            and self._downsample_cache_in_len == in_len
            and self._downsample_cache_channels == n_channels
            and self._downsample_scratch is not None
        )

        if not cache_valid:
            # Probe once to capture exact output length for this config.
            ch0_decimated = signal.decimate(block[:, 0].astype(np.float32), decimation_factor, zero_phase=True)
            out_len = int(ch0_decimated.size)
            self._downsample_scratch = np.empty((out_len, n_channels), dtype=np.float32)
            self._downsample_scratch[:, 0] = ch0_decimated

            self._downsample_cache_rate = self.sampling_rate
            self._downsample_cache_factor = decimation_factor
            self._downsample_cache_in_len = in_len
            self._downsample_cache_channels = n_channels
        else:
            self._downsample_scratch[:, 0] = signal.decimate(
                block[:, 0].astype(np.float32), decimation_factor, zero_phase=True
            )

        # Decimate remaining channels into cached scratch buffer.
        for ch in range(1, n_channels):
            self._downsample_scratch[:, ch] = signal.decimate(
                block[:, ch].astype(np.float32), decimation_factor, zero_phase=True
            )

        # Return a copy because the scratch buffer is reused on the next block.
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

    def _process_audio_thread(self):
        """
        Processing thread that handles computationally expensive operations:
        DOA estimation and beamforming. Runs independently from the audio callback.
        """
        self.logger.info(f"Processing thread started (downsample_rate: {self.downsample_rate})")
        block_count = 0
        start_time = time.monotonic()
        
        # Store original sample rate for restoration
        self._original_sampling_rate = self.sampling_rate
        original_bf_rate = getattr(self.beamformer, "sample_rate", None)
        doa_beamformer = getattr(self.doa_estimator, "beamformer", None)
        original_doa_bf_rate = getattr(doa_beamformer, "sample_rate", None)
        
        while not self._processing_stop_event.is_set():
            try:
                block = self._audio_queue.get(timeout=0.1)
                block_count += 1
                queue_size = self._audio_queue.qsize()
                self.logger.debug(f"[Processing] Received block #{block_count}, shape: {block.shape}, queue_size: {queue_size}")
            except queue.Empty:
                self.logger.debug("[Processing] Queue timeout, no block received")
                continue

            # Catch-up mode: when backlog grows, skip stale blocks and keep newest ones.
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

            # Apply downsampling if needed
            original_rate = self.sampling_rate
            if self.downsample_rate is not None and self.downsample_rate < self.sampling_rate:
                block = self._downsample_block(block)
                self.sampling_rate = self.downsample_rate  # Temporarily change sample rate
                if hasattr(self.beamformer, "sample_rate"):
                    self.beamformer.sample_rate = int(self.downsample_rate)
                if doa_beamformer is not None and hasattr(doa_beamformer, "sample_rate"):
                    doa_beamformer.sample_rate = int(self.downsample_rate)
                self.logger.debug(f"[Processing] Downsampled to {self.downsample_rate} Hz, new block shape: {block.shape}")

            # Extract per-microphone samples
            per_mic = {}
            for idx, mic in enumerate(self.mic_list):
                channel_index = mic.channel_number if mic.channel_number is not None else idx
                if 0 <= channel_index < block.shape[1]:
                    per_mic[mic.channel_number] = block[:, channel_index].copy()

            # Estimate DOA
            doa_value = None
            start_doa = time.monotonic()
            should_skip_doa = queue_size > self._skip_doa_queue_threshold and self._latest_doa is not None
            if should_skip_doa:
                doa_value = self._latest_doa
                self.logger.debug(f"[Processing] Skipping DOA update due to backlog (queue_size={queue_size}), reusing {doa_value:.1f}°")
            elif callable(self.doa_estimator.estimate_doa):
                try:
                    self.logger.debug(f"[Processing] Starting DOA estimation")
                    doa_value = self.doa_estimator.estimate_doa(block)
                    elapsed_doa = time.monotonic() - start_doa
                    self.logger.info(f"[DOA] Estimated: {doa_value:.1f}° (took {elapsed_doa*1000:.2f}ms)")
                except Exception as e:
                    elapsed_doa = time.monotonic() - start_doa
                    self.logger.error(f"[DOA Estimation Error] {type(e).__name__}: {e} (after {elapsed_doa*1000:.2f}ms)")
                    doa_value = None

            # Apply beamforming
            beamformed_value = None
            start_bf = time.monotonic()
            try:
                if isinstance(doa_value, (int, float, np.integer, np.floating)):
                    # Only update steering angle if it actually changed (avoid redundant updates)
                    current_angle = self.beamformer.get_steering_angle() if hasattr(self.beamformer, "get_steering_angle") else None
                    if current_angle is None or not np.isclose(float(doa_value), float(current_angle), atol=1e-4):
                        self.beamformer.set_steering_angle(float(doa_value))
                        self.logger.debug(f"[Processing] Steering angle set to {doa_value:.1f}°")

                if callable(getattr(self.beamformer, "apply", None)):
                    self.logger.debug(f"[Processing] Starting beamformer apply")
                    beamformed = self.beamformer.apply(block)
                    if beamformed is not None:
                        beamformed_arr = np.asarray(beamformed)
                        if beamformed_arr.size > 0:
                            beamformed_value = beamformed_arr.copy()
                            # Queue audio chunk for real-time output playback
                            mono_raw = np.asarray(beamformed_arr, dtype=np.float32).reshape(-1)
                            peak = float(np.max(np.abs(mono_raw))) if mono_raw.size > 0 else 0.0
                            # If values look int16-like, scale; otherwise keep as normalized float.
                            mono_out = mono_raw / 32768.0 if peak > 4.0 else mono_raw
                            # Apply optional post-beamforming filters before AGC.
                            for filt in self.filters:
                                if callable(getattr(filt, "apply", None)):
                                    try:
                                        mono_out = np.asarray(filt.apply(mono_out), dtype=np.float32).reshape(-1)
                                    except Exception as filter_err:
                                        self.logger.warning(f"[Filter] {type(filter_err).__name__}: {filter_err}")
                            output_sample_rate = self.downsample_rate if self.downsample_rate is not None else self.sampling_rate
                            mono_out = self.agc.process(mono_out, sample_rate=output_sample_rate)
                            mono_out = np.clip(mono_out * self.monitor_gain, -0.95, 0.95)
                            processing_rate = self.downsample_rate if self.downsample_rate is not None else self.sampling_rate
                            mono_out = self._resample_to_playback_rate(mono_out, int(processing_rate))
                            with self._output_fifo_lock:
                                self._output_fifo.append(mono_out)
                                self._output_buffered_samples += mono_out.size
                                while self._output_buffered_samples > self._output_max_buffer_samples and len(self._output_fifo) > 0:
                                    old = self._output_fifo.popleft()
                                    self._output_buffered_samples -= old.size
                            elapsed_bf = time.monotonic() - start_bf
                            self.logger.debug(f"[Beamforming] Output shape: {beamformed_arr.shape} (took {elapsed_bf*1000:.2f}ms)")
                    else:
                        self.logger.warning("[Beamforming] apply() returned None")
            except Exception as e:
                elapsed_bf = time.monotonic() - start_bf
                self.logger.error(f"[Beamforming Error] {type(e).__name__}: {e} (after {elapsed_bf*1000:.2f}ms)")
                beamformed_value = None
            
            # Restore original sample rate if downsampled
            if self.sampling_rate != original_rate:
                self.sampling_rate = original_rate
                if hasattr(self.beamformer, "sample_rate") and original_bf_rate is not None:
                    self.beamformer.sample_rate = int(original_bf_rate)
                if doa_beamformer is not None and hasattr(doa_beamformer, "sample_rate") and original_doa_bf_rate is not None:
                    doa_beamformer.sample_rate = int(original_doa_bf_rate)

            # Store results
            total_time = time.monotonic() - start_doa
            with self._lock:
                self._latest_block = block
                self._latest_per_mic = per_mic
                self._latest_doa = doa_value
                self._latest_beamformed = beamformed_value
            
            # Periodic performance report
            if block_count % 5 == 0:
                elapsed_total = time.monotonic() - start_time
                blocks_per_sec = block_count / elapsed_total
                self.logger.info(f"[Performance] Processed {block_count} blocks in {elapsed_total:.1f}s ({blocks_per_sec:.1f} blocks/sec), avg {total_time*1000:.0f}ms/block")
        
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

    def start_output_monitoring(self, blocksize: int = 0):
        """
        Start monitoring the beamformed audio output through the speaker.
        This opens an output stream and plays the latest beamformed audio.
        """
        if self._output_stream is not None:
            return

        output_sample_rate = self._output_playback_rate
        with self._output_fifo_lock:
            self._output_fifo.clear()
            self._output_current_chunk = np.zeros(0, dtype=np.float32)
            self._output_buffered_samples = 0
        
        self._output_stream = sd.OutputStream(
            samplerate=output_sample_rate,
            channels=1,
            dtype='float32',
            latency='low',
            blocksize=blocksize,
            callback=self._output_callback,
        )
        self._output_stream.start()

    def stop_output_monitoring(self):
        """
        Stop the output monitoring stream.
        """
        if self._output_stream is None:
            return
        
        self._output_stream.stop()
        self._output_stream.close()
        self._output_stream = None
        with self._output_fifo_lock:
            self._output_fifo.clear()
            self._output_current_chunk = np.zeros(0, dtype=np.float32)
            self._output_buffered_samples = 0

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

        outdata[:, 0] = np.clip(chunk, -1.0, 1.0)