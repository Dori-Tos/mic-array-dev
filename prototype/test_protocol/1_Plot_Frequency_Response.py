"""Plot frequency response from polar-pattern CSV measurements.

This script reads a hard-coded list of polar-pattern CSV files, extracts the
measurement at a target angle (either `expected_angle` or `relative_angle`)
and plots the gain (dB) versus frequency. The file list can be edited in the
`DEFAULT_FILES` list.

Usage:
    python 1_Plot_Frequency_Response.py [--files file1.csv file2.csv ...] [--angle 0] [--extrapolate-angle] [--extrapolation-points 2] [--output-dir out_dir] [--name filename]

If a CSV contains a `gain_db` column that value is used. Otherwise the script
will fall back to `rms_dbfs` or compute dB from `rms_level` if available.
Use `--use-rms-level-only` to ignore `gain_db` and `rms_dbfs` and plot only
from the `rms_level` column. Use `--rms-reference` to choose whether that
level is normalized against the input level or the per-file maximum.
When `--extrapolate-angle` is enabled, gains are linearly interpolated or
extrapolated from the nearest available angle samples if the exact target angle
is not present. For the 0° point, the script can instead extrapolate from the
`n` measurements before and after 0° using `--extrapolation-points`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATTERN = 3

if PATTERN == 1:
    BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/1_square_70dB")
    DEFAULT_FILES = [
        Path("polar_pattern_300_Hz_60cm_2.csv"),
        Path("polar_pattern_1500_Hz_60cm_2.csv"),
        Path("polar_pattern_2500_Hz_60cm_2.csv"),
        Path("polar_pattern_3500_Hz_60cm_2.csv"),
        Path("polar_pattern_4500_Hz_60cm_2.csv"),    
        Path("polar_pattern_5000_Hz_60cm_2.csv"),    
        Path("polar_pattern_6000_Hz_60cm_2.csv"),    
    ]
elif PATTERN == 2:
    BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/2_corners_70dB")
    DEFAULT_FILES = [
        Path("polar_pattern_300_Hz_60cm.csv"),
        Path("polar_pattern_500_Hz_60cm.csv"),
        Path("polar_pattern_1000_Hz.csv"),
        Path("polar_pattern_1500_Hz.csv"),
        Path("polar_pattern_2000_Hz.csv"),
        Path("polar_pattern_2500_Hz_60cm.csv"),
        Path("polar_pattern_3000_Hz_60cm.csv"),
        Path("polar_pattern_3500_Hz_60cm.csv"),
        Path("polar_pattern_4000_Hz_60cm.csv"),
        Path("polar_pattern_4500_Hz_60cm_2.csv"),   
        Path("polar_pattern_5000_Hz_60cm_2.csv"),    
        Path("polar_pattern_6000_Hz_60cm_2.csv"),     
    ]
elif PATTERN == 3:
    BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/3_rim_70dB")
    DEFAULT_FILES = [
        Path("polar_pattern_300_Hz_60cm.csv"),
        Path("polar_pattern_500_Hz_60cm.csv"),
        Path("polar_pattern_1000_Hz_60cm.csv"),
        Path("polar_pattern_1500_Hz_60cm.csv"),
        Path("polar_pattern_2000_Hz_60cm.csv"),
        Path("polar_pattern_2500_Hz_60cm.csv"),
        Path("polar_pattern_3000_Hz_60cm.csv"),
        Path("polar_pattern_3500_Hz_60cm.csv"),
        Path("polar_pattern_4000_Hz_60cm_2.csv"),
        Path("polar_pattern_4500_Hz_60cm_2.csv"),      
        Path("polar_pattern_5000_Hz_60cm_2.csv"),     
        Path("polar_pattern_6000_Hz_60cm_2.csv"), 
    ]
elif PATTERN == 4:
    BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/4_single_corner_70dB")
    DEFAULT_FILES = [
        Path("polar_pattern_300_Hz_60cm.csv"),
        Path("polar_pattern_500_Hz_60cm.csv"),
        Path("polar_pattern_1000_Hz_60cm.csv"),
        Path("polar_pattern_1500_Hz_60cm.csv"),
        Path("polar_pattern_2000_Hz_60cm.csv"),
        Path("polar_pattern_2500_Hz_60cm.csv"),
        Path("polar_pattern_3000_Hz_60cm.csv"),
        Path("polar_pattern_3500_Hz_60cm.csv"),
        Path("polar_pattern_4000_Hz_60cm.csv"),
        Path("polar_pattern_4500_Hz_60cm_2.csv"),      
        Path("polar_pattern_5000_Hz_60cm_2.csv"),        
        Path("polar_pattern_6000_Hz_60cm_2.csv"),
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
    """Compute true input-vs-output power transfer ratio from raw RMS values.
    
    This ALWAYS computes gain as: 20*log10(rms_level / input_rms_level)
    which represents the true power loss/gain between input and output.
    
    The protocol's precomputed gain_db and rms_dbfs columns are NOT used,
    as they are relative to each file's dBFS baseline, not true power transfer.
    """
    if "rms_level" not in df.columns or "input_rms_level" not in df.columns:
        return None, None

    rms = pd.to_numeric(df["rms_level"], errors="coerce").to_numpy(dtype=float)
    rms_in = pd.to_numeric(df["input_rms_level"], errors="coerce").to_numpy(dtype=float)
    ratio = np.divide(rms, np.maximum(rms_in, 1e-12))
    series = pd.Series(20.0 * np.log10(np.maximum(ratio, 1e-12)), index=df.index)
    return "gain_from_rms_true_power_transfer", series


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


def _extrapolate_zero_angle_from_neighbors(
    angle_values: np.ndarray,
    gain_values: np.ndarray,
    extrapolation_points: int,
    optimism: float = 0.5,
) -> float | None:
    """Estimate the 0° value from the nearest samples on both sides of 0°.

    Angles are wrapped to the signed range [-180, 180] so samples near 360°
    become negative angles (e.g. 345° -> -15°).
    """
    if angle_values.size == 0 or gain_values.size == 0:
        return None

    signed_angles = np.where(angle_values > 180.0, angle_values - 360.0, angle_values)
    valid = np.isfinite(signed_angles) & np.isfinite(gain_values)
    if not valid.any():
        return None

    signed_angles = signed_angles[valid]
    gain_values = gain_values[valid]

    if extrapolation_points <= 0:
        extrapolation_points = 1

    left_mask = signed_angles < 0.0
    right_mask = signed_angles > 0.0
    left_angles = signed_angles[left_mask]
    left_gains = gain_values[left_mask]
    right_angles = signed_angles[right_mask]
    right_gains = gain_values[right_mask]

    if left_angles.size == 0 or right_angles.size == 0:
        return None

    left_order = np.argsort(np.abs(left_angles))[:extrapolation_points]
    right_order = np.argsort(np.abs(right_angles))[:extrapolation_points]

    fit_angles = np.concatenate([left_angles[left_order], right_angles[right_order]])
    fit_gains = np.concatenate([left_gains[left_order], right_gains[right_order]])

    if fit_angles.size < 2:
        return None

    try:
        slope, intercept = np.polyfit(fit_angles, fit_gains, deg=1)
    except Exception:
        return None

    # optimism in [0,1]: 0 => return linear intercept (conservative)
    #                    1 => return the maximum neighbor gain (very optimistic)
    optimism = float(np.clip(optimism, 0.0, 1.0))
    neighbor_max = float(np.max(fit_gains))
    optimistic_val = intercept + optimism * (neighbor_max - intercept)
    # Never exceed the observed neighbor max
    optimistic_val = min(optimistic_val, neighbor_max)
    return float(optimistic_val)


def extract_gain_at_angle(
    csv_path: Path,
    target_angle_deg: float = 0.0,
    extrapolate_angle: bool = False,
    extrapolation_points: int = 4,
    print_extrapolated_values: bool = False,
    extrapolation_optimism: float = 0.5,
    use_rms_level_only: bool = False,
    rms_reference: str = "input",
    angle_tolerance_deg: float = ANGLE_TOLERANCE_DEG,
    debug: bool = False,
) -> float | None:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    
    # Validate: warn if RMS columns are missing
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

    if np.isclose(target_angle_deg, 0.0, atol=angle_tolerance_deg):
        zero_value = _extrapolate_zero_angle_from_neighbors(
            angle_values,
            gain_values,
            extrapolation_points=extrapolation_points,
            optimism=extrapolation_optimism,
        )
        if zero_value is not None:
            if print_extrapolated_values:
                print(f"{csv_path.name}: extrapolated 0° = {zero_value:.3f} dB")
            return zero_value

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
    extrapolation_points: int = 2,
    print_extrapolated_values: bool = False,
    extrapolation_optimism: float = 0.5,
    use_rms_level_only: bool = False,
    rms_reference: str = "input",
    ylim: Tuple[float, float] | None = None,
    debug: bool = False,
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
                extrapolation_points=extrapolation_points,
                print_extrapolated_values=print_extrapolated_values,
                extrapolation_optimism=extrapolation_optimism,
                use_rms_level_only=use_rms_level_only,
                rms_reference=rms_reference,
                debug=debug,
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
    parser.add_argument("--extrapolation-points", type=int, default=4,
                        help="Number of samples on each side of 0° to use when extrapolating the 0° value")
    parser.add_argument("--print-extrapolated-values", action="store_true",
                        help="Print the extrapolated 0° value for each CSV while plotting")
    parser.add_argument("--extrapolation-optimism", type=float, default=0.5,
                        help="Bias factor in [0..1] that shifts the 0° extrapolated value toward the neighbor max (1=use neighbor max)")
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
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output to check RMS column availability")
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
        extrapolation_points=args.extrapolation_points,
        print_extrapolated_values=args.print_extrapolated_values,
        extrapolation_optimism=args.extrapolation_optimism,
        use_rms_level_only=args.use_rms_level_only,
        rms_reference=args.rms_reference,
        ylim=tuple(args.ylim) if args.ylim else None,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
