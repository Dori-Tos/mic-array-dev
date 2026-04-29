"""
Continuous Gain Drift Test
==========================

Continuously captures audio and applies the same processing chain used by
1_Polar_Pattern.py to detect whether output gain drifts down over time.

Processing chain (default):
- MVDR beamformer
- BandPass (300-4000 Hz)
- Spectral subtraction
- AGC chain
"""

from datetime import datetime
from pathlib import Path
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import sounddevice as sd

# Import processing classes from the shared codebase
sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.Beamformer import MVDRBeamformer
from classes.Array_RealTime import apply_realtime_processing_chain
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, PedalboardAGC, AGCChain


def _safe_dbfs(rms_value: float, floor: float = 1e-10) -> float:
	return 20.0 * np.log10(max(float(rms_value), floor))


def _fit_trend(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
	"""Return slope, intercept, and R^2 for y = slope*x + intercept."""
	if len(x) < 2:
		return 0.0, float(y[0]) if len(y) else 0.0, 0.0

	slope, intercept = np.polyfit(x, y, deg=1)
	y_pred = (slope * x) + intercept
	ss_res = float(np.sum((y - y_pred) ** 2))
	ss_tot = float(np.sum((y - np.mean(y)) ** 2))
	r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
	return float(slope), float(intercept), float(r2)


def test_continuous_gain(
	duration_seconds=120.0,
	device_index=None,
	sample_duration=0.8,
	update_interval=1.0,
	baseline_samples=10,
	output_dir='data/test_protocol/continuous_gain',
	freeze_beamformer=True,
	freeze_angle_deg=0.0,
	enable_agc=False,
	enable_spectral_filter=True,
	use_pipeline=True,
):
	"""
	Measure if processed output level continuously diminishes over time.

	Args:
		duration_seconds: Total test duration.
		device_index: Input device index (None = default input).
		sample_duration: Length of one capture block.
		update_interval: Minimum time between capture starts.
		baseline_samples: Initial samples used as baseline reference.
		output_dir: Directory to save CSV result.
		freeze_beamformer: Keep steering fixed while testing.
		freeze_angle_deg: Steering angle for freeze mode.
		enable_agc: If True, enable both AGC stages (default: False).
		enable_spectral_filter: If True, enable spectral subtraction (default: True).
		use_pipeline: Apply the same chain as protocol 1 (default True).
	"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	if device_index is None:
		device_info = sd.query_devices(kind='input')
		device_index = device_info['index']
	else:
		device_info = sd.query_devices(device_index)

	sample_rate = int(device_info['default_samplerate'])
	device_name = device_info['name']
	num_channels = 4 if use_pipeline else 1

	print("\n" + "=" * 72)
	print("Continuous Gain Drift Test")
	print("=" * 72)
	print(f"  Device: {device_name} (index {device_index})")
	print(f"  Sample rate: {sample_rate} Hz")
	print(f"  Channels: {num_channels}")
	print(f"  Total duration: {duration_seconds:.1f} s")
	print(f"  Sample duration: {sample_duration:.2f} s")
	print(f"  Update interval: {update_interval:.2f} s")
	print(f"  Baseline samples: {baseline_samples}")
	print(f"  Processing: {'FULL PIPELINE' if use_pipeline else 'RAW ONLY'}")
	if use_pipeline:
		print(f"    - Spectral subtraction: {'ON' if enable_spectral_filter else 'OFF'}")
		print(f"    - AGC chain (both stages): {'ON' if enable_agc else 'OFF'}")
	print(f"  Freeze beamformer: {'ON' if freeze_beamformer else 'OFF'}")
	if freeze_beamformer:
		print(f"  Freeze angle: {float(freeze_angle_deg):.1f} deg")
	print("=" * 72 + "\n")

	logger = logging.getLogger("ContinuousGainTest")
	logger.setLevel(logging.INFO)

	if use_pipeline:
		mic_channel_numbers = [0, 1, 2, 3]
		geometry_path = Path(__file__).resolve().parent.parent / "array_geometries" / "1_Square.xml"
		mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))

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
		)

		filters = [
			BandPassFilter(
				logger=logger,
				sample_rate=sample_rate,
				low_cutoff=300.0,
				high_cutoff=4000.0,
				order=4,
			),
		]

		if enable_spectral_filter:
			filters.append(
				SpectralSubtractionFilter(
					logger=logger,
					sample_rate=sample_rate,
					noise_factor=0.65,
					gain_floor=0.55,
					noise_alpha=0.995,
					noise_update_snr_db=8.0,
					gain_smooth_alpha=0.92,
				)
			)

		if enable_agc:
			agc = AGCChain(logger=logger, stages=[
				AdaptiveAmplifier(
					logger=logger,
					target_rms=0.08,
					min_gain=1.0,
					max_gain=6.0,
					adapt_alpha=0.04,
					speech_activity_rms=0.00012,
					silence_decay_alpha=0.008,
					activity_hold_ms=600.0,
					peak_protect_threshold=0.30,
					peak_protect_strength=1.0,
					max_gain_warn_rms_min=0.001,
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
		else:
			agc = None
	else:
		beamformer = None
		filters = []
		agc = None

	print("Warming up pipeline for 2.0 s...")
	warmup_audio = sd.rec(
		int(2.0 * sample_rate),
		samplerate=sample_rate,
		channels=num_channels,
		device=device_index,
		blocking=True,
	)
	warmup_audio = np.asarray(warmup_audio, dtype=np.float32)
	warmup_peak = np.max(np.abs(warmup_audio)) if warmup_audio.size else 0.0
	if warmup_peak > 1.0:
		warmup_audio = warmup_audio / warmup_peak
	if use_pipeline:
		_ = apply_realtime_processing_chain(
			block=warmup_audio,
			beamformer=beamformer,
			filters=filters,
			agc=agc,
			sample_rate=sample_rate,
			monitor_gain=1.0,
			theta_deg=0.0,
			freeze_beamformer=bool(freeze_beamformer),
			freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
		)

	input("Warm-up complete. Press Enter to start continuous test... ")

	rows = []
	baseline_output = []
	baseline_input = []

	test_start = time.time()
	next_capture = test_start
	idx = 0

	print("\nRunning... Press Ctrl+C to stop early.\n")
	try:
		while True:
			now = time.time()
			elapsed = now - test_start
			if elapsed >= duration_seconds:
				break

			wait_time = next_capture - now
			if wait_time > 0:
				time.sleep(wait_time)

			block = sd.rec(
				int(sample_duration * sample_rate),
				samplerate=sample_rate,
				channels=num_channels,
				device=device_index,
				blocking=True,
			)
			block = np.asarray(block, dtype=np.float32)
			block_peak = np.max(np.abs(block)) if block.size else 0.0
			if block_peak > 1.0:
				block = block / block_peak

			raw_mono = block[:, 0] if block.ndim > 1 else block
			input_rms = float(np.sqrt(np.mean(raw_mono ** 2)))
			input_dbfs = _safe_dbfs(input_rms)

			if use_pipeline:
				processed = apply_realtime_processing_chain(
					block=block,
					beamformer=beamformer,
					filters=filters,
					agc=agc,
					sample_rate=sample_rate,
					monitor_gain=1.0,
					theta_deg=0.0,
					freeze_beamformer=bool(freeze_beamformer),
					freeze_angle_deg=float(freeze_angle_deg) if freeze_angle_deg is not None else None,
				)
			else:
				processed = np.asarray(raw_mono, dtype=np.float32)

			output_rms = float(np.sqrt(np.mean(processed ** 2)))
			output_dbfs = _safe_dbfs(output_rms)
			transfer_db = output_dbfs - input_dbfs

			idx += 1
			capture_time = time.time() - test_start

			if idx <= baseline_samples:
				baseline_input.append(input_dbfs)
				baseline_output.append(output_dbfs)

			baseline_output_db = float(np.mean(baseline_output)) if baseline_output else output_dbfs
			output_delta = output_dbfs - baseline_output_db

			rows.append({
				'index': idx,
				'elapsed_s': capture_time,
				'input_rms': input_rms,
				'input_dbfs': input_dbfs,
				'output_rms': output_rms,
				'output_dbfs': output_dbfs,
				'transfer_db': transfer_db,
				'output_delta_from_baseline_db': output_delta,
			})

			phase = "BASELINE" if idx <= baseline_samples else "MONITOR"
			print(
				f"[{phase:8s} {idx:4d}] "
				f"t={capture_time:7.1f}s | "
				f"In={input_dbfs:7.2f} dBFS | "
				f"Out={output_dbfs:7.2f} dBFS ({output_delta:+6.2f} dB) | "
				f"Out-In={transfer_db:+6.2f} dB"
			)

			next_capture += update_interval

	except KeyboardInterrupt:
		print("\nInterrupted by user. Finalizing available data...\n")

	if not rows:
		print("No data captured.")
		return None

	df = pd.DataFrame(rows)

	if len(df) > baseline_samples:
		analysis_df = df.iloc[baseline_samples:].copy()
	else:
		analysis_df = df.copy()

	x = analysis_df['elapsed_s'].to_numpy(dtype=float)
	y_out = analysis_df['output_dbfs'].to_numpy(dtype=float)
	y_in = analysis_df['input_dbfs'].to_numpy(dtype=float)
	y_transfer = analysis_df['transfer_db'].to_numpy(dtype=float)

	out_slope_s, _, out_r2 = _fit_trend(x, y_out)
	in_slope_s, _, in_r2 = _fit_trend(x, y_in)
	transfer_slope_s, _, transfer_r2 = _fit_trend(x, y_transfer)

	out_slope_min = out_slope_s * 60.0
	in_slope_min = in_slope_s * 60.0
	transfer_slope_min = transfer_slope_s * 60.0

	# Heuristic: significant continuous attenuation if output slope is strongly negative
	# and transfer also trends negative, indicating more than just source level change.
	significant_drop = (out_slope_min <= -0.5 and transfer_slope_min <= -0.2)

	ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	csv_file = output_path / f"continuous_gain_{ts}.csv"
	df.to_csv(csv_file, index=False)

	baseline_output_db = float(np.mean(baseline_output)) if baseline_output else float(df['output_dbfs'].iloc[0])
	baseline_input_db = float(np.mean(baseline_input)) if baseline_input else float(df['input_dbfs'].iloc[0])

	print("\n" + "=" * 72)
	print("Continuous Gain Drift Summary")
	print("=" * 72)
	print(f"  Samples captured: {len(df)}")
	print(f"  Baseline input (first {min(baseline_samples, len(df))}): {baseline_input_db:.2f} dBFS")
	print(f"  Baseline output (first {min(baseline_samples, len(df))}): {baseline_output_db:.2f} dBFS")
	print(f"  Final output: {df['output_dbfs'].iloc[-1]:.2f} dBFS")
	print(f"  Output slope: {out_slope_min:+.3f} dB/min (R^2={out_r2:.3f})")
	print(f"  Input slope: {in_slope_min:+.3f} dB/min (R^2={in_r2:.3f})")
	print(f"  Transfer slope (Out-In): {transfer_slope_min:+.3f} dB/min (R^2={transfer_r2:.3f})")
	print(f"  Interpretation: {'CONTINUOUS GAIN REDUCTION DETECTED' if significant_drop else 'NO STRONG CONTINUOUS REDUCTION DETECTED'}")
	print(f"  CSV saved to: {csv_file}")
	print("=" * 72)

	return df


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Continuous output-gain drift test using the same pipeline as protocol 1.'
	)
	parser.add_argument('--duration-total', type=float, default=120.0,
						help='Total test duration in seconds (default: 120)')
	parser.add_argument('--device', type=int, default=None,
						help='Audio device index (default: system default input)')
	parser.add_argument('--duration', type=float, default=0.8,
						help='Sample duration in seconds per capture (default: 0.8)')
	parser.add_argument('--interval', type=float, default=1.0,
						help='Interval in seconds between capture starts (default: 1.0)')
	parser.add_argument('--baseline', type=int, default=10,
						help='Number of baseline samples (default: 10)')
	parser.add_argument('--output', type=str, default='data/test_protocol/continuous_gain',
						help='Output directory (default: data/test_protocol/continuous_gain)')
	parser.add_argument('--freeze-beamformer', action=argparse.BooleanOptionalAction, default=True,
						help='Freeze beamformer steering during the test (default: enabled)')
	parser.add_argument('--freeze-angle', type=float, default=0.0,
						help='Steering angle used when beamformer freeze is enabled (default: 0.0)')
	parser.add_argument('--enable-agc', action=argparse.BooleanOptionalAction, default=False,
						help='Enable both AGC stages (AdaptiveAmplifier + PedalboardAGC) (default: disabled)')
	parser.add_argument('--enable-spectral-filter', action=argparse.BooleanOptionalAction, default=True,
						help='Enable spectral subtraction filter (default: enabled)')
	parser.add_argument('--no-pipeline', action='store_true',
						help='Disable processing pipeline and monitor raw channel only')
	parser.add_argument('--list-devices', action='store_true',
						help='List available audio input devices and exit')

	args = parser.parse_args()

	if args.list_devices:
		print("\nAvailable Audio Input Devices:")
		print("=" * 72)
		devices = sd.query_devices()
		for i, dev in enumerate(devices):
			if dev.get('max_input_channels', 0) > 0:
				default_marker = ' [DEFAULT INPUT]' if i == sd.default.device[0] else ''
				print(f"  {i:2d}: {dev['name']:<40} ({dev['max_input_channels']} ch, {int(dev['default_samplerate'])} Hz){default_marker}")
		print("=" * 72)
		raise SystemExit(0)

	test_continuous_gain(
		duration_seconds=args.duration_total,
		device_index=args.device,
		sample_duration=args.duration,
		update_interval=args.interval,
		baseline_samples=args.baseline,
		output_dir=args.output,
		freeze_beamformer=args.freeze_beamformer,
		freeze_angle_deg=args.freeze_angle,
		enable_agc=args.enable_agc,
		enable_spectral_filter=args.enable_spectral_filter,
		use_pipeline=not args.no_pipeline,
	)

