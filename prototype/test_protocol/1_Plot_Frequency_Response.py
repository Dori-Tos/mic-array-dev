"""Plot frequency response from polar-pattern CSV measurements.

This script reads a hard-coded list of polar-pattern CSV files, extracts the
measurement at a target angle (either `expected_angle` or `relative_angle`)
and plots the gain (dB) versus frequency. The file list can be edited in the
`DEFAULT_FILES` list.

Usage:
    python 1_Plot_Frequency_Response.py [--files file1.csv file2.csv ...] [--angle 0] [--extrapolate-angle] [--output-dir out_dir] [--name filename]

If a CSV contains a `gain_db` column that value is used. Otherwise the script
will fall back to `rms_dbfs` or compute dB from `rms_level` if available.
Use `--use-rms-level-only` to ignore `gain_db` and `rms_dbfs` and plot only
from the `rms_level` column. Use `--rms-reference` to choose whether that
level is normalized against the input level or the per-file maximum.
When `--extrapolate-angle` is enabled, gains are linearly interpolated or
extrapolated from the nearest available angle samples if the exact target angle
is not present.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/4_single_corner_70dB")

DEFAULT_FILES = [
    # Edit this list to include the CSVs to use for the frequency response plot.
    Path("polar_pattern_300_Hz_60cm.csv"),
    Path("polar_pattern_500_Hz_60cm.csv"),
    Path("polar_pattern_1000_Hz_60cm.csv"),
    Path("polar_pattern_2500_Hz_60cm.csv"),
    Path("polar_pattern_2000_Hz_60cm.csv"),
    Path("polar_pattern_2500_Hz_60cm.csv"),
    Path("polar_pattern_3000_Hz_60cm.csv"),
    Path("polar_pattern_3500_Hz_60cm.csv"),
    Path("polar_pattern_4000_Hz_60cm.csv"),
    Path("polar_pattern_4500_Hz_60cm.csv"),    
]

ANGLES = [0, 180]
ANGLE_TOLERANCE_DEG = 1.0


def parse_frequency_from_filename(p: Path) -> float | None:
    m = re.search(r"(\d+)[_\- ]?Hz", p.name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: look for first integer in filename
    m2 = re.search(r"(\d{2,5})", p.name)
    if m2:
        return float(m2.group(1))
    return None


def _get_angle_column(df: pd.DataFrame) -> str | None:
    for col in ("relative_angle", "expected_angle", "angle", "reference_angle"):
        if col in df.columns:
            return col
    return None


def _get_gain_series(df: pd.DataFrame, use_rms_level_only: bool = False) -> Tuple[str | None, pd.Series | None]:
    if use_rms_level_only:
        if "rms_level" not in df.columns:
            return None, None
        series = pd.to_numeric(df["rms_level"], errors="coerce")
        return "rms_level", series
    if "gain_db" in df.columns:
        return "gain_db", pd.to_numeric(df["gain_db"], errors="coerce")
    if "rms_dbfs" in df.columns:
        return "rms_dbfs", pd.to_numeric(df["rms_dbfs"], errors="coerce")
    if "rms_level" in df.columns:
        series = pd.to_numeric(df["rms_level"], errors="coerce")
        return "rms_level", pd.Series(20.0 * np.log10(np.maximum(series.to_numpy(dtype=float), 1e-12)), index=df.index)
    return None, None


def _interpolate_or_extrapolate(angle_values: np.ndarray, gain_values: np.ndarray, target_angle: float) -> float | None:
    if angle_values.size == 0 or gain_values.size == 0:
        return None

    order = np.argsort(angle_values)
    angle_values = angle_values[order]
    gain_values = gain_values[order]

    unique_angles, inverse = np.unique(angle_values, return_inverse=True)
    unique_gains = np.array([
        float(np.mean(gain_values[inverse == index]))
        for index in range(unique_angles.size)
    ])

    if unique_angles.size == 1:
        return float(unique_gains[0])

    if target_angle <= unique_angles[0]:
        slope = (unique_gains[1] - unique_gains[0]) / (unique_angles[1] - unique_angles[0])
        return float(unique_gains[0] + slope * (target_angle - unique_angles[0]))

    if target_angle >= unique_angles[-1]:
        slope = (unique_gains[-1] - unique_gains[-2]) / (unique_angles[-1] - unique_angles[-2])
        return float(unique_gains[-1] + slope * (target_angle - unique_angles[-1]))

    return float(np.interp(target_angle, unique_angles, unique_gains))


def extract_gain_at_angle(
    csv_path: Path,
    target_angle_deg: float = 0.0,
    extrapolate_angle: bool = False,
    use_rms_level_only: bool = False,
    rms_reference: str = "input",
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
) -> float | None:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    angle_col = _get_angle_column(df)
    gain_col, gain_series = _get_gain_series(df, use_rms_level_only=use_rms_level_only)
    if angle_col is None:
        if gain_series is None or len(gain_series) == 0:
            return None
        if use_rms_level_only:
            reference_series = _get_rms_reference_series(df, gain_series, rms_reference)
            if reference_series is None:
                return None
            ratio = gain_series.iloc[0] / reference_series.iloc[0]
            return float(20.0 * np.log10(max(ratio, 1e-12)))
        return float(gain_series.iloc[0])

    if gain_series is None or gain_col is None:
        return None

    if use_rms_level_only:
        reference_series = _get_rms_reference_series(df, gain_series, rms_reference)
        if reference_series is None:
            return None
        gain_series = pd.Series(
            20.0 * np.log10(np.maximum(gain_series.to_numpy(dtype=float) / np.maximum(reference_series.to_numpy(dtype=float), 1e-12), 1e-12)),
            index=df.index,
        )

    angle_series = pd.to_numeric(df[angle_col], errors="coerce")
    valid = angle_series.notna() & gain_series.notna()
    if not valid.any():
        return None

    angle_values = angle_series[valid].to_numpy(dtype=float)
    gain_values = gain_series[valid].to_numpy(dtype=float)

    close_mask = np.isclose(angle_values, target_angle_deg, atol=angle_tolerance_deg)
    if close_mask.any():
        return float(np.mean(gain_values[close_mask]))

    if not extrapolate_angle:
        nearest_index = int(np.abs(angle_values - target_angle_deg).argmin())
        return float(gain_values[nearest_index])

    return _interpolate_or_extrapolate(angle_values, gain_values, target_angle_deg)


def _get_rms_reference_series(
    df: pd.DataFrame,
    rms_level_series: pd.Series,
    rms_reference: str,
) -> pd.Series | None:
    if rms_reference == "input":
        if "input_rms_level" not in df.columns:
            return None
        reference_series = pd.to_numeric(df["input_rms_level"], errors="coerce")
        return reference_series

    if rms_reference == "max":
        max_value = float(np.nanmax(rms_level_series.to_numpy(dtype=float)))
        if not np.isfinite(max_value) or max_value <= 0.0:
            return None
        return pd.Series(np.full(len(df), max_value, dtype=float), index=df.index)

    raise ValueError(f"Unsupported rms_reference: {rms_reference}")


def plot_frequency_response(
    files: List[Path],
    output: Path | None = None,
    target_angles_deg: List[float] | None = None,
    extrapolate_angle: bool = False,
    use_rms_level_only: bool = False,
    rms_reference: str = "input",
    ylim: Tuple[float, float] | None = None,
) -> None:
    if not target_angles_deg:
        target_angles_deg = [0.0]

    angle_freqs: dict[float, List[float]] = {angle: [] for angle in target_angles_deg}
    angle_gains: dict[float, List[float]] = {angle: [] for angle in target_angles_deg}
    missing: List[Path] = []

    for p in files:
        p = Path(p)
        if not p.exists():
            missing.append(p)
            continue
        freq = parse_frequency_from_filename(p)
        if freq is None:
            missing.append(p)
            continue

        for angle in target_angles_deg:
            gain = extract_gain_at_angle(
                p,
                target_angle_deg=angle,
                extrapolate_angle=extrapolate_angle,
                use_rms_level_only=use_rms_level_only,
                rms_reference=rms_reference,
            )
            if gain is None:
                continue
            angle_freqs[angle].append(freq)
            angle_gains[angle].append(gain)

    available_angles = [angle for angle in target_angles_deg if angle_freqs[angle]]
    if not available_angles:
        raise SystemExit("No valid files found or no angle gains available. Check DEFAULT_FILES or --files")

    plt.figure(figsize=(8, 4.5))
    for index, angle in enumerate(available_angles):
        freq_values = np.array(angle_freqs[angle])
        gain_values = np.array(angle_gains[angle])
        order = np.argsort(freq_values)
        freq_values = freq_values[order]
        gain_values = gain_values[order]
        plt.plot(
            freq_values,
            gain_values,
            marker="o",
            linestyle="-",
            color=f"C{index % 10}",
            label=f"{angle:.0f}°",
        )

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("Array Frequency Response by Angle")
    plt.legend(title="Angle", loc="best")
    if use_rms_level_only:
        all_gains = np.concatenate([np.asarray(values, dtype=float) for values in angle_gains.values() if values])
        if all_gains.size:
            ymin = float(np.nanmin(all_gains))
            ymax = float(np.nanmax(all_gains))
            spread = max(5.0, (ymax - ymin) * 0.12)
            plt.ylim(ymin - spread, ymax + spread)
    else:
        plt.ylim(-20, 20)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()

    if missing:
        print("Skipped missing or invalid files:")
        for m in missing:
            print(" -", m)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frequency response from polar CSV files")
    parser.add_argument("--files", nargs="*", type=str, default=None,
                        help="Optional list of CSV files to use (overrides built-in DEFAULT_FILES)")
    parser.add_argument("--angle", type=float, default=None,
                        help="Optional single target angle in degrees to extract from each CSV. If omitted, all ANGLES are plotted.")
    parser.add_argument("--extrapolate-angle", action="store_true",
                        help="Interpolate or extrapolate angle samples when the target angle is not present")
    parser.add_argument("--use-rms-level-only", action="store_true",
                        help="Force use of the rms_level column only, ignoring gain_db and rms_dbfs")
    parser.add_argument("--rms-reference", choices=("input", "max"), default="input",
                        help="Reference used when --use-rms-level-only is active: input_rms_level or the per-file maximum rms_level")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output image (if omitted the plot is shown)")
    parser.add_argument("--name", type=str, default="frequency_response.png",
                        help="Filename to use when saving the plot")
    parser.add_argument("--ylim", nargs=2, type=float, default=None,
                        help="Optional y-axis limits as two floats: ymin ymax (e.g. --ylim -20 20)")
    args = parser.parse_args()

    REAL_PATH_FILES = [BASE_PATH / f for f in DEFAULT_FILES]
    files = [Path(f) for f in args.files] if args.files else REAL_PATH_FILES

    # Build output path from directory + name. If no directory provided, show plot instead.
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / args.name
    else:
        output = None

    target_angles = [float(args.angle)] if args.angle is not None else [float(angle) for angle in ANGLES]

    plot_frequency_response(
        files,
        output,
        target_angles_deg=target_angles,
        extrapolate_angle=args.extrapolate_angle,
        use_rms_level_only=args.use_rms_level_only,
        rms_reference=args.rms_reference,
        ylim=tuple(args.ylim) if args.ylim else None,
    )


if __name__ == "__main__":
    main()
