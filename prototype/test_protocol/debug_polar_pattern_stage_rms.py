"""Stage-by-stage RMS diagnostic for the polar-pattern processing chain.

This script captures one block from the selected input device, runs it through
the same processing stages used by ``1_Polar_Pattern.py``, and prints the RMS
after each stage so you can see where attenuation is introduced.

It is intentionally narrow: no CSV writing, no turntable loop, no plotting.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).parent.parent))
from classes.AGC import AGCChain, AdaptiveAmplifier, PedalboardAGC
from classes.Beamformer import MVDRBeamformer
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


def _rms_level(samples: np.ndarray) -> float:
    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data ** 2)))


def _dbfs(level: float) -> float:
    return 20.0 * math.log10(max(level, 1e-12))


def _compute_input_reference_levels(audio_block: np.ndarray, gain_input_reference: str) -> tuple[float, float]:
    data = np.asarray(audio_block, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
        data = data.reshape(-1, 1)

    if data.size == 0 or data.shape[1] == 0:
        return 0.0, 0.0

    if gain_input_reference == "ch0":
        ref = data[:, 0]
        ref_rms = float(np.sqrt(np.mean(ref ** 2))) if ref.size else 0.0
        ref_peak = float(np.max(np.abs(ref))) if ref.size else 0.0
        return ref_rms, ref_peak

    per_channel_rms = np.sqrt(np.mean(data ** 2, axis=0))
    per_channel_peak = np.max(np.abs(data), axis=0)
    reducer = np.mean if gain_input_reference == "mean" else np.median
    ref_rms = float(reducer(per_channel_rms)) if per_channel_rms.size else 0.0
    ref_peak = float(reducer(per_channel_peak)) if per_channel_peak.size else 0.0
    return ref_rms, ref_peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Print RMS after each polar-pattern processing stage")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--geometry", type=int, default=2, help="Geometry selector (XML prefix)")
    parser.add_argument("--num-mics", type=int, default=8, help="Number of microphones used by the beamformer")
    parser.add_argument("--sample-duration", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument("--freeze-beamformer", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze steering at a fixed angle")
    parser.add_argument("--freeze-angle-deg", type=float, default=0.0, help="Fixed steering angle when frozen")
    parser.add_argument("--enable-spectral-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable spectral subtraction filter")
    parser.add_argument("--enable-agc", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable the AGC chain")
    parser.add_argument("--gain-input-reference", choices=("mean", "median", "ch0"), default="mean",
                        help="Input reference used for the gain denominator")
    parser.add_argument("--process-block-ms", type=float, default=20.0,
                        help="Processing chunk size in milliseconds")
    args = parser.parse_args()

    logger = logging.getLogger("PolarPatternStageRMS")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    if args.device is None:
        device_info = sd.query_devices(kind="input")
        device_index = device_info["index"]
        device_name = device_info["name"]
    else:
        device_info = sd.query_devices(args.device)
        device_index = device_info["index"]
        device_name = device_info["name"]

    sample_rate = int(device_info["default_samplerate"])
    num_channels = int(args.num_mics)
    max_input_channels = int(device_info.get("max_input_channels", 0))
    if max_input_channels < num_channels:
        raise ValueError(
            f"Selected input device supports only {max_input_channels} input channels, "
            f"but this run requires {num_channels}."
        )

    geometry_path = _resolve_geometry_path(int(args.geometry))
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    mic_channel_numbers = list(range(num_channels))

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

    filters: list[object] = [
        BandPassFilter(
            logger=logger,
            sample_rate=sample_rate,
            low_cutoff=300.0,
            high_cutoff=4000.0,
            order=4,
        ),
    ]
    if args.enable_spectral_filter:
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

    if args.enable_agc:
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

    capture_samples = int(round(args.sample_duration * sample_rate))
    process_block_samples = int(max(32, round((args.process_block_ms / 1000.0) * sample_rate)))
    process_block_samples = int(min(process_block_samples, capture_samples))

    print(f"Device: {device_name} (index {device_index})")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Geometry: {geometry_path.name}")
    print(f"Channels: {num_channels}")
    print(f"Processing chunk: {process_block_samples} samples")
    print(f"Gain input reference: {args.gain_input_reference}")
    print(f"Spectral subtraction: {'on' if args.enable_spectral_filter else 'off'}")
    print(f"AGC: {'on' if args.enable_agc else 'off'}")

    input(f"Place the array and press Enter to capture {args.sample_duration:.2f}s...")
    audio_data = sd.rec(
        capture_samples,
        samplerate=sample_rate,
        channels=num_channels,
        device=device_index,
        dtype="float32",
        blocking=True,
    )

    audio_data = np.asarray(audio_data, dtype=np.float32)
    max_val = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
    if max_val > 1.0:
        audio_data = audio_data / max_val

    input_rms_level, input_peak_level = _compute_input_reference_levels(audio_data, args.gain_input_reference)
    print()
    print("Input reference:")
    print(f"  RMS:  {input_rms_level:.8f} ({_dbfs(input_rms_level):.2f} dBFS)")
    print(f"  Peak: {input_peak_level:.8f} ({_dbfs(input_peak_level):.2f} dBFS)")

    beamformed = beamformer.apply(audio_data, theta_deg=float(args.freeze_angle_deg))
    beamformed = np.asarray(beamformed, dtype=np.float32).reshape(-1)
    beamformed_rms = _rms_level(beamformed)
    print()
    print("After beamformer:")
    print(f"  RMS:  {beamformed_rms:.8f} ({_dbfs(beamformed_rms):.2f} dBFS)")
    print(f"  Gain vs input: {_dbfs(beamformed_rms) - _dbfs(input_rms_level):.2f} dB")

    stage_output = beamformed
    for index, stage in enumerate(filters, start=1):
        stage_output = np.asarray(stage.apply(stage_output), dtype=np.float32).reshape(-1)
        stage_rms = _rms_level(stage_output)
        print()
        print(f"After filter {index} ({stage.__class__.__name__}):")
        print(f"  RMS:  {stage_rms:.8f} ({_dbfs(stage_rms):.2f} dBFS)")
        print(f"  Gain vs input: {_dbfs(stage_rms) - _dbfs(input_rms_level):.2f} dB")

    if agc is not None:
        stage_output = np.asarray(agc.process(stage_output, sample_rate), dtype=np.float32).reshape(-1)
        stage_rms = _rms_level(stage_output)
        print()
        print("After AGC chain:")
        print(f"  RMS:  {stage_rms:.8f} ({_dbfs(stage_rms):.2f} dBFS)")
        print(f"  Gain vs input: {_dbfs(stage_rms) - _dbfs(input_rms_level):.2f} dB")

    final_rms = _rms_level(stage_output)
    print()
    print("Final output:")
    print(f"  RMS:  {final_rms:.8f} ({_dbfs(final_rms):.2f} dBFS)")
    print(f"  Total gain vs input: {_dbfs(final_rms) - _dbfs(input_rms_level):.2f} dB")


if __name__ == "__main__":
    main()