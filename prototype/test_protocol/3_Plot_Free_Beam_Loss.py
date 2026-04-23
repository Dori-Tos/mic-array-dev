"""Plot Free Beam Loss / Gain (Protocol 3)
=====================================

This script plots the results produced by the manual Protocol 3 capture
(`3_Free_Beam_Loss.py`).

It supports two common workflows:

1) Single frequency/signal (one CSV):
   - Plot gain vs angle.

2) Sweep across frequencies (multiple CSVs):
   - Plot gain vs frequency with one curve per angle.

Notes
-----
- Protocol 3 CSV filenames often do not encode the frequency. For the multi-CSV
  workflow, provide frequencies explicitly via --freqs, or ensure your filenames
  contain a recognizable "500Hz" / "2kHz" pattern.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class _CsvSeries:
	path: Path
	freq_hz: float | None
	df: pd.DataFrame


def _parse_freq_from_text(text: str) -> float | None:
	"""Best-effort frequency parser from a filename/label.

	Recognizes patterns like:
	- 500Hz, 500 Hz
	- 2kHz, 2 kHz
	- 2.5kHz
	"""
	if not text:
		return None

	lowered = text.lower()
	# Prefer kHz first to avoid matching "2" in "2kHz" as 2 Hz.
	m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*k\s*hz", lowered)
	if m:
		return float(m.group("val")) * 1000.0
	m = re.search(r"(?P<val>\d+(?:\.\d+)?)\s*hz", lowered)
	if m:
		return float(m.group("val"))
	return None


def _load_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	if df.empty:
		raise ValueError(f"CSV is empty: {path}")
	return df


def _detect_angle_column(df: pd.DataFrame, preferred: str | None = None) -> str:
	if preferred:
		if preferred not in df.columns:
			raise ValueError(f"Requested angle column '{preferred}' not present in CSV")
		return preferred
	for candidate in ("expected_angle", "angle_deg", "angle"):
		if candidate in df.columns:
			return candidate
	raise ValueError("No angle column found; expected one of: expected_angle, angle_deg, angle")


def _detect_gain_column(df: pd.DataFrame, preferred: str | None = None) -> str:
	if preferred:
		if preferred not in df.columns:
			raise ValueError(f"Requested gain column '{preferred}' not present in CSV")
		return preferred
	for candidate in ("gain_db", "gain", "rms_dbfs"):
		if candidate in df.columns:
			return candidate
	raise ValueError("No gain column found; expected one of: gain_db, gain, rms_dbfs")


def _as_float_series(series: pd.Series) -> np.ndarray:
	return pd.to_numeric(series, errors="coerce").astype(float).to_numpy()


def _prepare_series(paths: list[Path], freqs_hz: list[float | None] | None) -> list[_CsvSeries]:
	if freqs_hz is not None and len(freqs_hz) != len(paths):
		raise ValueError("--freqs must have the same count as CSV files")

	series_list: list[_CsvSeries] = []
	for idx, path in enumerate(paths):
		df = _load_csv(path)
		freq = None
		if freqs_hz is not None:
			freq = freqs_hz[idx]
		if freq is None:
			freq = _parse_freq_from_text(path.stem)
		series_list.append(_CsvSeries(path=path, freq_hz=freq, df=df))
	return series_list


def plot_gain_vs_angle(
	csv_path: Path,
	*,
	angle_col: str | None = None,
	gain_col: str | None = None,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
):
	df = _load_csv(csv_path)
	angle_col = _detect_angle_column(df, angle_col)
	gain_col = _detect_gain_column(df, gain_col)

	angles = _as_float_series(df[angle_col])
	gains = _as_float_series(df[gain_col])
	order = np.argsort(angles)
	angles = angles[order]
	gains = gains[order]

	freq = _parse_freq_from_text(csv_path.stem)
	default_title = "Free beam gain vs angle"
	if freq is not None:
		default_title += f" ({freq:.0f} Hz)"

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111)
	ax.plot(angles, gains, marker="o", linewidth=2)
	ax.set_xlabel("Signal Angle")
	ax.set_ylabel("Gain (dB)")
	ax.set_title(title or default_title)
	ax.grid(True, alpha=0.3)
	ax.set_xlim(float(np.nanmin(angles)), float(np.nanmax(angles)))

	# Optional: visualize DOA estimate (if present).
	if "doa_deg" in df.columns:
		doa = _as_float_series(df["doa_deg"])[order]
		if np.isfinite(doa).any():
			ax2 = ax.twinx()
			ax2.plot(angles, doa, color="tab:orange", linestyle="--", marker="x", alpha=0.7)
			ax2.axhline(0.0, color="0.75", linewidth=1.0, alpha=0.5, linestyle="-")
			ax2.set_ylabel("Estimated Angle (doa)")

	fig.tight_layout()

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=160)

	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_gain_vs_frequency_per_angle(
	csv_paths: list[Path],
	*,
	freqs_hz: list[float | None] | None = None,
	angle_col: str | None = None,
	gain_col: str | None = None,
	title: str | None = None,
	save_path: Path | None = None,
	show: bool = True,
):
	series_list = _prepare_series(csv_paths, freqs_hz)
	if any(s.freq_hz is None for s in series_list):
		missing = [str(s.path) for s in series_list if s.freq_hz is None]
		raise ValueError(
			"Could not infer frequencies for one or more CSVs. "
			"Provide --freqs (e.g. '--freqs 500,750,1000') or encode like '500Hz' in filenames. "
			f"Missing: {missing}"
		)

	# Use the first CSV to pick column names.
	angle_col = _detect_angle_column(series_list[0].df, angle_col)
	gain_col = _detect_gain_column(series_list[0].df, gain_col)

	# Gather (freq, gain) points for each angle.
	angle_to_points: dict[float, list[tuple[float, float]]] = {}
	for s in series_list:
		freq = float(s.freq_hz)
		df = s.df
		if angle_col not in df.columns or gain_col not in df.columns:
			raise ValueError(
				f"CSV {s.path} missing required columns '{angle_col}' and/or '{gain_col}'."
			)

		angles = _as_float_series(df[angle_col])
		gains = _as_float_series(df[gain_col])
		for a, g in zip(angles, gains, strict=False):
			if not np.isfinite(a) or not np.isfinite(g):
				continue
			angle_key = float(a)
			angle_to_points.setdefault(angle_key, []).append((freq, float(g)))

	if not angle_to_points:
		raise ValueError("No valid (angle, gain) points found across the provided CSVs")

	# Sort angles for deterministic legend order.
	angles_sorted = sorted(angle_to_points.keys())

	fig = plt.figure(figsize=(11, 6))
	ax = fig.add_subplot(111)

	for angle in angles_sorted:
		points = angle_to_points[angle]
		points.sort(key=lambda p: p[0])
		freqs = np.array([p[0] for p in points], dtype=float)
		gains = np.array([p[1] for p in points], dtype=float)
		ax.plot(freqs, gains, marker="o", linewidth=2, label=f"{angle:g}°")

	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel(f"{gain_col} (dB)")
	ax.set_title(title or "Free beam gain vs frequency (one curve per angle)")
	ax.grid(True, alpha=0.3)
	ax.legend(title=f"{angle_col}", ncol=2, fontsize=9)
	ax.set_xscale("log")
	ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))

	fig.tight_layout()

	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(save_path, dpi=160)

	if show:
		plt.show()
	else:
		plt.close(fig)


def _parse_freq_list(raw: str) -> list[float]:
	values: list[float] = []
	for part in raw.split(","):
		text = part.strip()
		if not text:
			continue
		values.append(float(text))
	if not values:
		raise ValueError("--freqs cannot be empty")
	return values


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Plot Protocol 3 free beam gain outputs")
	parser.add_argument("csv_files", nargs="+", type=str, help="One or more Protocol 3 CSV files")
	parser.add_argument(
		"--mode",
		choices=("auto", "angle", "freq"),
		default="auto",
		help="auto: 1 CSV -> angle, N CSVs -> freq; angle: gain vs angle; freq: gain vs frequency per angle",
	)
	parser.add_argument(
		"--freqs",
		type=str,
		default=None,
		help="Comma-separated frequencies (Hz) corresponding to each CSV file in order (required if filenames have no Hz/kHz)",
	)
	parser.add_argument("--angle-col", type=str, default=None, help="Angle column to use (default: auto)")
	parser.add_argument("--gain-col", type=str, default=None, help="Gain column to use (default: gain_db)")
	parser.add_argument("--title", type=str, default=None, help="Optional plot title")
	parser.add_argument("--no-show", action="store_true", help="Do not show an interactive window")
	parser.add_argument("--no-save", action="store_true", help="Do not save the plot")
	parser.add_argument(
		"--out",
		type=str,
		default=None,
		help="Output PNG path (default: alongside first CSV)",
	)

	args = parser.parse_args(argv)

	csv_paths = [Path(p) for p in args.csv_files]
	for p in csv_paths:
		if not p.exists():
			raise FileNotFoundError(str(p))

	show = not bool(args.no_show)
	mode = args.mode
	if mode == "auto":
		mode = "angle" if len(csv_paths) == 1 else "freq"

	out_path: Path | None
	if args.no_save:
		out_path = None
	elif args.out is not None:
		out_path = Path(args.out)
	else:
		# Single file: use source CSV name; multiple files: use timestamped generic name
		if len(csv_paths) == 1:
			out_path = csv_paths[0].with_suffix(".png")
		else:
			timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			out_path = csv_paths[0].parent / f"free_beam_loss_{timestamp}.png"

	freqs = _parse_freq_list(args.freqs) if args.freqs is not None else None

	if mode == "angle":
		if len(csv_paths) != 1:
			raise ValueError("mode=angle expects exactly 1 CSV")
		plot_gain_vs_angle(
			csv_paths[0],
			angle_col=args.angle_col,
			gain_col=args.gain_col,
			title=args.title,
			save_path=out_path,
			show=show,
		)
		return 0

	if mode == "freq":
		plot_gain_vs_frequency_per_angle(
			csv_paths,
			freqs_hz=(list(freqs) if freqs is not None else None),
			angle_col=args.angle_col,
			gain_col=args.gain_col,
			title=args.title,
			save_path=out_path,
			show=show,
		)
		return 0

	raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
	raise SystemExit(main())

