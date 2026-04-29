"""
Save Processed Output (Protocol 4)
=================================

Continuously captures multichannel audio, applies the same processing chain as
test_pipeline.py, and writes processed mono output to a WAV file.

Behavior:
- Press Ctrl+C to stop and save the current recording.
- Warn when output file grows beyond a configured size threshold.
- Abort without saving if output exceeds a hard safety limit.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
from collections import deque
import logging
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.AGC import AdaptiveAmplifier, AGCChain, NoiseAwareAdaptiveAmplifier, PedalboardAGC
from classes.Array_RealTime import apply_realtime_processing_chain
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Filter import BandPassFilter, SpectralSubtractionFilter


def _resolve_geometry_path(geometry_value: int) -> Path:
	geometry_dir = Path(__file__).resolve().parent.parent / "array_geometries"
	target_prefix = str(int(geometry_value))
	matches = sorted(
		p for p in geometry_dir.glob("*.xml")
		if p.stem.split("_", 1)[0] == target_prefix
	)
	if not matches:
		available = sorted(p.name for p in geometry_dir.glob("*.xml"))
		raise ValueError(
			f"No geometry XML found for selector '{geometry_value}'. "
			f"Expected a file like '{geometry_value}_*.xml' in {geometry_dir}. "
			f"Available: {available}"
		)
	return matches[0]


def _bytes_to_human(value: int) -> str:
	units = ["B", "KB", "MB", "GB", "TB"]
	size = float(value)
	idx = 0
	while size >= 1024.0 and idx < len(units) - 1:
		size /= 1024.0
		idx += 1
	return f"{size:.2f} {units[idx]}"


def _safe_dbfs(value: float, floor: float = 1e-10) -> float:
	return 20.0 * np.log10(max(float(value), floor))


def _build_pipeline(
	*,
	logger: logging.Logger,
	sample_rate: int,
	num_mics: int,
	geometry: int,
):
	geometry_path = _resolve_geometry_path(int(geometry))
	mic_channel_numbers = list(range(int(num_mics)))
	mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))

	das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
	)

	beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.95,
        diagonal_loading=0.15,
        spectral_whitening_factor=0.12,
        weight_smooth_alpha=0.72,
        max_adaptive_loading_scale=4.0,
        coherence_suppression_strength=0.8,
        weight_smooth_alpha_min=0.45,
        weight_smooth_alpha_max=0.82,
        snr_threshold_for_sharpening=2.0,
        backward_null_strength=0.9,
	)
 
 
	doa_estimator = IterativeDOAEstimator(
		logger=logger,
		update_rate=3.0,
		angle_range=(-25.0, 25.0),
		doa_beamformer=das_beamformer,
		beamformer=beamformer,
		scan_step_deg=5.0,
		local_search_radius_deg=10.0,
		periodic_full_scan_blocks=20,
	)

	filters = [
		BandPassFilter(
			logger=logger,
			sample_rate=sample_rate,
			low_cutoff=300.0,
			high_cutoff=4000.0,
			order=4,
		),
		SpectralSubtractionFilter(
			logger=logger,
			sample_rate=sample_rate,
			noise_factor=0.65,
			gain_floor=0.55,
			noise_alpha=0.995,
			noise_update_snr_db=8.0,
			gain_smooth_alpha=0.92,
		)
    ]

	agc = AGCChain(logger=logger, stages=[
		NoiseAwareAdaptiveAmplifier(
			logger=logger,
			target_rms=0.08,
			min_gain=0.7,
			max_gain_baseline=6.0,
			gain_up_alpha=0.008,
			gain_down_alpha=0.15,
			snr_threshold_db=8.0,
			noise_floor_alpha=0.997,
			activity_hold_ms=100.0,
			peak_protect_threshold=0.30,
			peak_protect_strength=1.0,
		),
		PedalboardAGC(
			logger=logger,
			sample_rate=sample_rate,
			threshold_db=-20.0,
			ratio=2.0,
			attack_ms=3.0,
			release_ms=140.0,
			limiter_threshold_db=-7.0,
			limiter_release_ms=50.0,
		),
	])

	return geometry_path, mic_channel_numbers, mic_positions, beamformer, doa_estimator, filters, agc


def run_save_output(
	device_index: int | None = None,
	output_dir: str = "data/test_protocol/4_save_output",
	sample_rate: int | None = None,
	blocksize: int = 960,
	num_mics: int = 8,
	geometry: int = 2,
	freeze_beamformer: bool = False,
	freeze_angle_deg: float = 0.0,
	listen_output: bool = False,
	monitor_gain: float = 0.22,
	warmup_seconds: float = 2.0,
	warn_size_mb: float = 512.0,
	hard_limit_mb: float = 2048.0,
):
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	if device_index is None:
		device_info = sd.query_devices(kind="input")
		device_index = int(device_info["index"])
	else:
		device_info = sd.query_devices(device_index)

	device_name = str(device_info["name"])
	effective_sample_rate = int(sample_rate or int(device_info["default_samplerate"]))
	max_input_channels = int(device_info.get("max_input_channels", 0))
	if max_input_channels < int(num_mics):
		raise ValueError(
			f"Selected device supports only {max_input_channels} input channels, "
			f"but --num-mics requires {num_mics}."
		)

	logger = logging.getLogger("SaveOutput")
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setLevel(logging.INFO)
		handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
		logger.addHandler(handler)

	geometry_path, mic_channel_numbers, _mic_positions, beamformer, doa_estimator, filters, agc = _build_pipeline(
		logger=logger,
		sample_rate=effective_sample_rate,
		num_mics=num_mics,
		geometry=geometry,
	)

	warn_bytes = int(max(1.0, float(warn_size_mb)) * 1024 * 1024)
	hard_bytes = int(max(float(warn_size_mb) + 1.0, float(hard_limit_mb)) * 1024 * 1024)

	ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	final_file = output_path / f"beamformed_output_{ts}.wav"
	temp_file = output_path / f"beamformed_output_{ts}.tmp.wav"

	print("\n" + "=" * 72)
	print("Protocol 4 - Save Processed Output")
	print("=" * 72)
	print(f"  Device: {device_name} (index {device_index})")
	print(f"  Sample rate: {effective_sample_rate} Hz")
	print(f"  Blocksize: {blocksize} samples")
	print(f"  Num mics: {num_mics}")
	print(f"  Geometry: {geometry_path.name}")
	print("  Pipeline: MVDR + BandPass + SpectralSubtraction(ON) + AGC(ON)")
	print(f"  Freeze beamformer: {freeze_beamformer}")
	if freeze_beamformer:
		print(f"  Freeze angle: {float(freeze_angle_deg):.1f} deg")
	print(f"  Listen output: {'ON' if listen_output else 'OFF'}")
	if listen_output:
		print(f"  Monitor gain: {monitor_gain:.2f}")
	print(f"  Warning size threshold: {_bytes_to_human(warn_bytes)}")
	print(f"  Hard stop threshold: {_bytes_to_human(hard_bytes)} (abort without saving)")
	print(f"  Temporary output: {temp_file}")
	print(f"  Final output: {final_file}")
	print("=" * 72)
	print("Press Ctrl+C to stop and save.")

	bytes_written = 0
	warned = False
	save_allowed = True
	interrupted_by_user = False
	start_time = time.time()
	monitor_lock = threading.Lock()
	monitor_fifo: deque[np.ndarray] = deque()
	monitor_current = np.zeros(0, dtype=np.float32)
	monitor_stream = None

	def _monitor_callback(outdata, frames, time_info, status):
		nonlocal monitor_current
		if status:
			pass
		chunk = np.zeros(frames, dtype=np.float32)
		write_idx = 0
		with monitor_lock:
			while write_idx < frames:
				if monitor_current.size == 0:
					if not monitor_fifo:
						break
					monitor_current = monitor_fifo.popleft()
				remaining = frames - write_idx
				take = min(remaining, monitor_current.size)
				if take <= 0:
					break
				chunk[write_idx:write_idx + take] = monitor_current[:take]
				monitor_current = monitor_current[take:]
				write_idx += take
		outdata[:, 0] = np.clip(chunk * float(monitor_gain), -1.0, 1.0)

	if listen_output:
		monitor_stream = sd.OutputStream(
			samplerate=effective_sample_rate,
			channels=1,
			dtype="float32",
			latency="low",
			blocksize=int(blocksize),
			callback=_monitor_callback,
		)

	wav_writer = wave.open(str(temp_file), "wb")
	wav_writer.setnchannels(1)
	wav_writer.setsampwidth(2)
	wav_writer.setframerate(effective_sample_rate)

	try:
		with sd.InputStream(
			samplerate=effective_sample_rate,
			channels=int(num_mics),
			device=int(device_index),
			dtype="float32",
			blocksize=int(blocksize),
		) as stream:
			if monitor_stream is not None:
				monitor_stream.start()
			warmup_end = time.time() + max(0.0, float(warmup_seconds))

			while True:
				block, _overflowed = stream.read(int(blocksize))
				block = np.asarray(block, dtype=np.float32)

				if not bool(freeze_beamformer):
					try:
						doa = doa_estimator.estimate_doa(block)
						# Only update steering angle if it actually changed (avoid redundant updates)
						if doa is not None and hasattr(beamformer, "set_steering_angle"):
							current_angle = beamformer.get_steering_angle() if hasattr(beamformer, "get_steering_angle") else None
							if current_angle is None or not np.isclose(float(doa), float(current_angle), atol=1e-4):
								beamformer.set_steering_angle(float(doa))
					except Exception as doa_err:
						logger.warning(f"DOA update warning: {type(doa_err).__name__}: {doa_err}")
				elif hasattr(beamformer, "set_steering_angle"):
					current_angle = beamformer.get_steering_angle() if hasattr(beamformer, "get_steering_angle") else None
					if current_angle is None or not np.isclose(float(freeze_angle_deg), float(current_angle), atol=1e-4):
						beamformer.set_steering_angle(float(freeze_angle_deg))

				processed = apply_realtime_processing_chain(
					block=block,
					beamformer=beamformer,
					filters=filters,
					agc=agc,
					sample_rate=effective_sample_rate,
					monitor_gain=1.0,
					theta_deg=None,
					freeze_beamformer=bool(freeze_beamformer),
					freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
				)

				mono = np.asarray(processed, dtype=np.float32).reshape(-1)
				mono = np.clip(mono, -1.0, 1.0)
				if listen_output:
					with monitor_lock:
						monitor_fifo.append(mono.copy())
						while len(monitor_fifo) > 12:
							monitor_fifo.popleft()
				pcm16 = (mono * 32767.0).astype(np.int16)

				if time.time() < warmup_end:
					continue

				wav_writer.writeframes(pcm16.tobytes())
				bytes_written += int(pcm16.nbytes)

				if (not warned) and bytes_written >= warn_bytes:
					warned = True
					print(
						f"WARNING: output is getting large ({_bytes_to_human(bytes_written)}). "
						f"Hard stop at {_bytes_to_human(hard_bytes)}."
					)

				if bytes_written >= hard_bytes:
					print(
						f"ERROR: hard output limit exceeded ({_bytes_to_human(bytes_written)}). "
						"Aborting and discarding recording."
					)
					save_allowed = False
					break

	except KeyboardInterrupt:
		interrupted_by_user = True
		print("\nCtrl+C received. Finalizing output...")
	finally:
		if monitor_stream is not None:
			try:
				monitor_stream.stop()
				monitor_stream.close()
			except Exception:
				pass
		wav_writer.close()

	if not save_allowed:
		try:
			if temp_file.exists():
				temp_file.unlink()
		finally:
			print("Recording aborted due to hard size limit. No file saved.")
		return

	temp_file.replace(final_file)
	elapsed = time.time() - start_time
	print("\nSaved processed output WAV:")
	print(f"  File: {final_file}")
	print(f"  Duration (approx): {elapsed:.1f} s")
	print(f"  Audio bytes: {_bytes_to_human(bytes_written)}")
	if not interrupted_by_user:
		print("  Capture ended normally.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Save processed beamforming output to WAV using the test_pipeline processing chain."
	)
	parser.add_argument("--device", type=int, default=None, help="Input device index (default: system default input)")
	parser.add_argument("--output", type=str, default="data/test_protocol/4_save_output", help="Output directory")
	parser.add_argument("--sample-rate", type=int, default=None, help="Override sample rate (Hz)")
	parser.add_argument("--blocksize", type=int, default=960, help="Input blocksize in samples (default: 960)")
	parser.add_argument("--num-mics", type=int, default=8, help="Number of input microphones/channels (default: 8)")
	parser.add_argument("--geometry", type=int, default=2, help="Geometry selector by XML filename prefix (default: 2)")
	parser.add_argument("--freeze-beamformer", action=argparse.BooleanOptionalAction, default=False, help="Freeze beamformer steering")
	parser.add_argument("--freeze-angle", type=float, default=0.0, help="Steering angle when beamformer is frozen")
	parser.add_argument("--listen-output", action=argparse.BooleanOptionalAction, default=False, help="Play the processed output live while recording")
	parser.add_argument("--monitor-gain", type=float, default=0.22, help="Playback gain for live monitoring (default: 0.22)")
	parser.add_argument("--warmup-seconds", type=float, default=2.0, help="Warmup time before writing audio")
	parser.add_argument("--warn-size-mb", type=float, default=512.0, help="Warn when output size exceeds this MB")
	parser.add_argument("--hard-limit-mb", type=float, default=2048.0, help="Abort without saving when output exceeds this MB")

	args = parser.parse_args()

	run_save_output(
		device_index=args.device,
		output_dir=args.output,
		sample_rate=args.sample_rate,
		blocksize=args.blocksize,
		num_mics=args.num_mics,
		geometry=args.geometry,
		freeze_beamformer=args.freeze_beamformer,
		freeze_angle_deg=args.freeze_angle,
		listen_output=args.listen_output,
		monitor_gain=args.monitor_gain,
		warmup_seconds=args.warmup_seconds,
		warn_size_mb=args.warn_size_mb,
		hard_limit_mb=args.hard_limit_mb,
	)
