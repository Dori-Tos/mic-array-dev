"""Plot frequency response from 0° gains in polar-pattern CSV measurements.

This script reads a hard-coded list of polar-pattern CSV files, extracts the
measurement at 0° (either `expected_angle` or `relative_angle`) and plots the
gain (dB) versus frequency. The file list can be edited in the `DEFAULT_FILES`
list.

Usage:
    python 1_Plot_Frequency_Response.py [--files file1.csv file2.csv ...] [--output-dir out_dir] [--name filename]

If a CSV contains a `gain_db` column that value is used. Otherwise the script
will fall back to `rms_dbfs` or compute dB from `rms_level` if available.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_PATH = Path("Python/Tests/mic-array-dev/data/test_protocol/1_polar_pattern/2_corners_70dB")

DEFAULT_FILES = [
    # Edit this list to include the CSVs to use for the frequency response plot.
    Path("polar_pattern_500_Hz_40cm.csv"),
    Path("polar_pattern_1000_Hz.csv"),
    Path("polar_pattern_2500_Hz.csv"),
    Path("polar_pattern_2000_Hz.csv"),
    Path("polar_pattern_2500_Hz_40cm.csv"),
    Path("polar_pattern_3000_Hz.csv"),
    Path("polar_pattern_3500_Hz_40cm.csv"),
    Path("polar_pattern_4000_Hz_40cm.csv"),    
]


def parse_frequency_from_filename(p: Path) -> float | None:
    m = re.search(r"(\d+)[_\- ]?Hz", p.name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    # fallback: look for first integer in filename
    m2 = re.search(r"(\d{2,5})", p.name)
    if m2:
        return float(m2.group(1))
    return None


def extract_0deg_gain(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)

    # Candidate angle columns (priority order)
    angle_cols = [c for c in ("relative_angle", "expected_angle", "angle", "reference_angle") if c in df.columns]

    if not angle_cols:
        # No angle column found, assume single-value file and take first row
        row = df.iloc[0]
    else:
        # Prefer relative_angle then expected_angle
        sel = None
        for col in angle_cols:
            # Find rows where angle is (close to) zero
            mask = np.isclose(df[col].astype(float), 0.0, atol=1e-6)
            if mask.any():
                sel = df[mask].iloc[0]
                break
        if sel is None:
            # fallback: choose row with minimum absolute angle
            col = angle_cols[0]
            idx = int(np.abs(df[col].astype(float)).argmin())
            sel = df.iloc[idx]
        row = sel

    if "rms_level" in df.columns:
        # Convert to dB relative to max in file to get meaningful scale
        val = float(row["rms_level"])
        maxv = float(df["rms_level"].max()) if df["rms_level"].max() > 0 else val
        if maxv <= 0:
            return None
        return 20.0 * np.log10(max(val, 1e-12) / maxv)

    return None


def plot_frequency_response(files: List[Path], output: Path | None = None) -> None:
    freqs: List[float] = []
    gains: List[float] = []
    missing: List[Path] = []

    for p in files:
        p = Path(p)
        if not p.exists():
            missing.append(p)
            continue
        freq = parse_frequency_from_filename(p)
        gain = extract_0deg_gain(p)
        if gain is None or freq is None:
            missing.append(p)
            continue
        freqs.append(freq)
        gains.append(gain)

    if not freqs:
        raise SystemExit("No valid files found or no 0° gains available. Check DEFAULT_FILES or --files")

    # Sort by frequency
    freqs = np.array(freqs)
    gains = np.array(gains)
    order = np.argsort(freqs)
    freqs = freqs[order]
    gains = gains[order]

    plt.figure(figsize=(8, 4.5))
    plt.plot(freqs, gains, marker="o", linestyle="-", color="C0")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("Array Frequency Response (0°)")
    plt.ylim(-20, 20)
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
    parser = argparse.ArgumentParser(description="Plot frequency response from 0° gains in polar CSV files")
    parser.add_argument("--files", nargs="*", type=str, default=None,
                        help="Optional list of CSV files to use (overrides built-in DEFAULT_FILES)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output image (if omitted the plot is shown)")
    parser.add_argument("--name", type=str, default="frequency_response.png",
                        help="Filename to use when saving the plot")
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

    plot_frequency_response(files, output)


if __name__ == "__main__":
    main()
