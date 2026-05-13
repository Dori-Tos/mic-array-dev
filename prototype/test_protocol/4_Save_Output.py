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
import logging
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))

from classes.AGC import AdaptiveAmplifier, AGCChain, NoiseAwareAdaptiveAmplifier, PedalboardAGC
from classes.Array_RealTime import Array_RealTime
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Filter import BandPassFilter, SpectralSubtractionFilter, WienerFilter
from classes.Microphone import Microphone


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
	sample_rate: int = 48000,
	num_mics: int,
	geometry: int,
	filter_type: str = "spectral",
	single_mic: bool = False,
	low_pass_cutoff: float = 300.0,
	high_pass_cutoff: float = 6000.0,
	enable_frequency_smoothing: bool = True,
	frequency_smoothing_strength: float = 0.3,
	enable_eigenvalue_suppression: bool = True,
	enable_adaptive_loading: bool = True,
	enable_weight_smoothing: bool = True,
	enable_coherence_suppression: bool = True,
	enable_backward_null_constraint: bool = True,
	enable_output_crossfade: bool = True,
	max_beamform_freq: float = 8000.0,
):
	geometry_path = _resolve_geometry_path(int(geometry))
	mic_channel_numbers = [0] if single_mic else list(range(int(num_mics)))
	mic_positions = None  # Initialize; will be loaded if not in single-mic mode

	# Skip beamforming components in single-mic mode
	if single_mic:
		beamformer = None
		doa_estimator = None
		das_beamformer = None
	else:
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
			enable_frequency_smoothing=enable_frequency_smoothing,
			frequency_smoothing_strength=frequency_smoothing_strength,
			enable_eigenvalue_suppression=enable_eigenvalue_suppression,
			enable_adaptive_loading=enable_adaptive_loading,
			enable_weight_smoothing=enable_weight_smoothing,
			enable_coherence_suppression=enable_coherence_suppression,
			enable_backward_null_constraint=enable_backward_null_constraint,
			enable_output_crossfade=enable_output_crossfade,
			max_beamform_freq=max_beamform_freq,
		)
	 
	 
		doa_estimator = IterativeDOAEstimator(
			logger=logger,
			doa_beamformer=das_beamformer,
			beamformer=beamformer,
		)

	# Select filter based on filter_type parameter
	if filter_type.lower() == "wiener":
		denoiser = WienerFilter(
			logger=logger,
			sample_rate=sample_rate,
		)
	else:  # default to spectral
		denoiser = SpectralSubtractionFilter(
			logger=logger,
			sample_rate=sample_rate,
		)

	filters = [
		BandPassFilter(
			logger=logger,
			sample_rate=sample_rate,
			low_cutoff=low_pass_cutoff,
			high_cutoff=high_pass_cutoff,
			order=4,
		),
		denoiser,
	]

	agc = AGCChain(logger=logger, stages=[
		NoiseAwareAdaptiveAmplifier(
			logger=logger,
		),
		PedalboardAGC(
			logger=logger,
			sample_rate=sample_rate,
		),
	])

	return geometry_path, mic_channel_numbers, mic_positions, beamformer, doa_estimator, filters, agc


def run_save_output(
	device_index: int | None = None,
	output_dir: str = "data/test_protocol/4_save_output",
	sample_rate: int | None = 48000,
	blocksize: int = 960,
	num_mics: int = 14,
	geometry: int = 5,
	freeze_beamformer: bool = False,
	freeze_angle_deg: float = 0.0,
	listen_output: bool = True,
	monitor_gain: float = 0.22,
	warmup_seconds: float = 2.0,
	warn_size_mb: float = 512.0,
	hard_limit_mb: float = 2048.0,
	filter_type: str = "spectral",
	single_mic: bool = False,
	low_pass_cutoff: float = 300.0,
	high_pass_cutoff: float = 6000.0,
	enable_frequency_smoothing: bool = True,
	frequency_smoothing_strength: float = 0.3,
	enable_eigenvalue_suppression: bool = True,
	enable_adaptive_loading: bool = True,
	enable_weight_smoothing: bool = True,
	enable_coherence_suppression: bool = True,
	enable_backward_null_constraint: bool = True,
	enable_output_crossfade: bool = True,
	max_beamform_freq_hz: float = 8000.0,
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
	
	# In single-mic mode, only need 1 input channel; otherwise need all num_mics channels
	input_channels = 1 if single_mic else int(num_mics)
	if max_input_channels < input_channels:
		raise ValueError(
			f"Selected device supports only {max_input_channels} input channels, "
			f"but requires {input_channels}."
		)

	logger = logging.getLogger("SaveOutput")
	# Configure root logging so all module loggers propagate to the console
	# Use basicConfig only if the root logger has no handlers yet.
	log_level = logging.DEBUG if args.debug else logging.INFO
	if not logging.getLogger().handlers:
		logging.basicConfig(
			level=log_level,
			stream=sys.stdout,
			format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
		)
	# Ensure this module logger is set and will propagate to root handlers
	logger.setLevel(log_level)
	logger.propagate = True
	if log_level == logging.DEBUG:
		logger.info("DEBUG logging ENABLED - expect verbose output from all pipeline components")

	# Build the pipeline with configured parameters
	geometry_path, mic_channel_numbers, _mic_positions, beamformer, doa_estimator, filters, agc = _build_pipeline(
		logger=logger,
		sample_rate=effective_sample_rate,
		num_mics=num_mics,
		geometry=geometry,
		filter_type=filter_type,
		single_mic=single_mic,
		low_pass_cutoff=low_pass_cutoff,
		high_pass_cutoff=high_pass_cutoff,	
		enable_frequency_smoothing=enable_frequency_smoothing,
		frequency_smoothing_strength=frequency_smoothing_strength,
		enable_eigenvalue_suppression=enable_eigenvalue_suppression,
		enable_adaptive_loading=enable_adaptive_loading,
		enable_weight_smoothing=enable_weight_smoothing,
		enable_coherence_suppression=enable_coherence_suppression,
		enable_backward_null_constraint=enable_backward_null_constraint,
		enable_output_crossfade=enable_output_crossfade,
		max_beamform_freq=max_beamform_freq_hz,
	)

	warn_bytes = int(max(1.0, float(warn_size_mb)) * 1024 * 1024)
	hard_bytes = int(max(float(warn_size_mb) + 1.0, float(hard_limit_mb)) * 1024 * 1024)

	ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	final_file = output_path / f"beamformed_output_{ts}.wav"
	temp_file = output_path / f"beamformed_output_{ts}.tmp.wav"

	print("\n" + "=" * 72)
	print("Protocol 4 - Save Processed Output (via Array_RealTime)")
	print("=" * 72)
	print(f"  Device: {device_name} (index {device_index})")
	print(f"  Sample rate: {effective_sample_rate} Hz")
	print(f"  Blocksize: {blocksize} samples")
	print(f"  Num mics: {num_mics}")
	print(f"  Geometry: {geometry_path.name}")
	# Display pipeline based on mode
	if single_mic:
		filter_name = "WienerFilter" if filter_type.lower() == "wiener" else "SpectralSubtractionFilter"
		print(f"  Pipeline: SINGLE MIC (NO BEAMFORMING) + BandPass + {filter_name}(ON) + AGC(ON)")
	else:
		filter_name = "WienerFilter" if filter_type.lower() == "wiener" else "SpectralSubtractionFilter"
		print(f"  Pipeline: MVDR + BandPass + {filter_name}(ON) + AGC(ON) [via Array_RealTime]")
	print(f"  Optional MVDR stages: freq_smooth={'ON' if enable_frequency_smoothing else 'OFF'}, "
		  f"eigen_sup={'ON' if enable_eigenvalue_suppression else 'OFF'}, "
		  f"adaptive_load={'ON' if enable_adaptive_loading else 'OFF'}, "
		  f"weight_smooth={'ON' if enable_weight_smoothing else 'OFF'}, "
		  f"coherence={'ON' if enable_coherence_suppression else 'OFF'}, "
		  f"back_null={'ON' if enable_backward_null_constraint else 'OFF'}, "
		  f"crossfade={'ON' if enable_output_crossfade else 'OFF'}")
	print(f"  Max beamform freq: {float(max_beamform_freq_hz):.0f} Hz")
	print(f"  Freeze beamformer: {freeze_beamformer}")
	if freeze_beamformer:
		print(f"  Freeze angle: {float(freeze_angle_deg):.1f} deg")
	print(f"  Monitor gain: {monitor_gain:.2f} (output monitoring disabled)")
	print(f"  Warning size threshold: {_bytes_to_human(warn_bytes)}")
	print(f"  Hard stop threshold: {_bytes_to_human(hard_bytes)} (abort without saving)")
	print(f"  Temporary output: {temp_file}")
	print(f"  Final output: {final_file}")
	print("=" * 72)
	print("Press Ctrl+C to stop and save.")

	# Create microphone list for Array_RealTime
	mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=effective_sample_rate) 
	            for i in mic_channel_numbers]

	# Instantiate Array_RealTime with "local_save" mode for simultaneous playback + WAV writing
	array = Array_RealTime(
		id_vendor=0x2752,
		id_product=0x0019,
		logger=logger,
		mic_list=mic_list,
		sampling_rate=effective_sample_rate,
		doa_estimator=doa_estimator if not freeze_beamformer else None,  # None if frozen
		beamformer=beamformer,
		echo_canceller=None,
		filters=filters,
		agc=agc,
		codec=None,
		monitor_gain=monitor_gain,
		output_mode="local_save",  # ← Enable simultaneous playback + WAV saving
		save_wav_path=str(temp_file),
		save_warn_bytes=warn_bytes,
		save_hard_limit_bytes=hard_bytes,
		output_boundary_fade_ms=0.0,
		downsample_rate=None,
		post_beamforming_block_ms=10.0,
		initial_silence_duration=warmup_seconds,  # ← WAV writer will respect this for silence at startup
	)

	interrupted_by_user = False
	save_allowed = True
	start_time = time.time()

	try:
		array.start_realtime(blocksize=blocksize)
		if listen_output:
			array.start_output_monitoring(blocksize=blocksize)
		
		# Monitor WAV writing status in a simple loop
		last_status_print = time.monotonic()
		while True:
			time.sleep(0.5)  # Check status every 500ms
			
			save_stats = array.get_save_stats()
			
			# Check for hard limit exceeded (set by Array_RealTime in output callback)
			if save_stats["hard_limit_exceeded"]:
				print(
					f"ERROR: hard output limit exceeded ({save_stats['bytes_written_mb']:.1f} MB). "
					"Aborting and discarding recording."
				)
				save_allowed = False
				break
			
			# Periodically print status (every 10 seconds)
			now = time.monotonic()
			if (now - last_status_print) >= 10.0:
				print(f"  Recording: {save_stats['bytes_written_mb']:.1f} MB so far...")
				last_status_print = now

	except KeyboardInterrupt:
		interrupted_by_user = True
		print("\nCtrl+C received. Finalizing output...")
	finally:
		try:
			array.stop_realtime()
		except Exception as e:
			logger.warning(f"Error stopping Array_RealTime: {e}")

	if not save_allowed:
		try:
			if temp_file.exists():
				temp_file.unlink()
		finally:
			print("Recording aborted due to hard size limit. No file saved.")
		return

	# Verify WAV file exists and get final statistics
	save_stats = array.get_save_stats()
	if not temp_file.exists():
		print("ERROR: WAV file was not created.")
		return

	temp_file.replace(final_file)
	elapsed = time.time() - start_time
	print("\nSaved processed output WAV:")
	print(f"  File: {final_file}")
	print(f"  Duration (approx): {elapsed:.1f} s")
	print(f"  Audio bytes: {_bytes_to_human(save_stats['bytes_written'])}")
	if not interrupted_by_user:
		print("  Capture ended normally.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Save processed beamforming output to WAV using the test_pipeline processing chain."
	)
	parser.add_argument("--device", type=int, default=None, help="Input device index (default: system default input)")
	parser.add_argument("--output", type=str, default="data/test_protocol/4_save_output", help="Output directory")
	parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging to diagnose pipeline issues")
	parser.add_argument("--sample-rate", type=int, default=48000, help="Override sample rate (Hz)")
	parser.add_argument("--blocksize", type=int, default=960, help="Input blocksize in samples (default: 960)")
	parser.add_argument("--num-mics", type=int, default=14, help="Number of input microphones/channels (default: 8)")
	parser.add_argument("--geometry", type=int, default=5, help="Geometry selector by XML filename prefix (default: 2)")
	parser.add_argument("--freeze-beamformer", action=argparse.BooleanOptionalAction, default=False, help="Freeze beamformer steering")
	parser.add_argument("--freeze-angle", type=float, default=0.0, help="Steering angle when beamformer is frozen")
	parser.add_argument("--listen-output", action=argparse.BooleanOptionalAction, default=False, help="Play the processed output live while recording")
	parser.add_argument("--monitor-gain", type=float, default=0.22, help="Playback gain for live monitoring (default: 0.22)")
	parser.add_argument("--warmup-seconds", type=float, default=2.0, help="Warmup time before writing audio")
	parser.add_argument("--warn-size-mb", type=float, default=512.0, help="Warn when output size exceeds this MB")
	parser.add_argument("--hard-limit-mb", type=float, default=2048.0, help="Abort without saving when output exceeds this MB")
	parser.add_argument("--filter", type=str, default="spectral", choices=["spectral", "wiener"], help="Denoiser filter: spectral (SpectralSubtractionFilter) or wiener (WienerFilter) (default: spectral)")
	parser.add_argument("--single-mic", action=argparse.BooleanOptionalAction, default=False, help="Use only the first microphone (no beamforming/DOA)")
	parser.add_argument("--low-pass-cutoff", type=float, default=300.0, help="Low passband filter cutoff frequency (Hz)")
	parser.add_argument("--high-pass-cutoff", type=float, default=6000.0, help="High passband filter cutoff frequency (Hz)")
	parser.add_argument("--no-frequency-smoothing", action="store_true", help="Disable covariance frequency smoothing")
	parser.add_argument("--frequency-smoothing-strength", type=float, default=0.3, help="Blend factor for covariance frequency smoothing (default: 0.3)")
	parser.add_argument("--no-eigenvalue-suppression", action="store_true", help="Disable eigenvalue-based diagonal loading boost")
	parser.add_argument("--no-adaptive-loading", action="store_true", help="Disable SNR-dependent adaptive diagonal loading")
	parser.add_argument("--no-weight-smoothing", action="store_true", help="Disable temporal MVDR weight smoothing")
	parser.add_argument("--no-coherence-suppression", action="store_true", help="Disable coherence-based output modulation")
	parser.add_argument("--no-backward-null", action="store_true", help="Disable the 180-degree backward null constraint")
	parser.add_argument("--no-output-crossfade", action="store_true", help="Disable beamformer output crossfade")
	parser.add_argument("--max-beamform-freq", type=float, default=8000.0, help="Maximum frequency (Hz) to perform MVDR; higher bins bypass MVDR and use pass-through/DAS fallback (default: 6000)")

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
		filter_type=args.filter,
		single_mic=args.single_mic,
		low_pass_cutoff=args.low_pass_cutoff,
		high_pass_cutoff=args.high_pass_cutoff,
		enable_frequency_smoothing=not args.no_frequency_smoothing,
		frequency_smoothing_strength=args.frequency_smoothing_strength,
		enable_eigenvalue_suppression=not args.no_eigenvalue_suppression,
		enable_adaptive_loading=not args.no_adaptive_loading,
		enable_weight_smoothing=not args.no_weight_smoothing,
		enable_coherence_suppression=not args.no_coherence_suppression,
		enable_backward_null_constraint=not args.no_backward_null,
		enable_output_crossfade=not args.no_output_crossfade,
		max_beamform_freq_hz=args.max_beamform_freq,
	)
