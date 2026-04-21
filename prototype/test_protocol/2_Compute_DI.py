"""
Directivity Index (DI) Frequency Analysis
==========================================

Computes the directivity index (DI) of an array across multiple frequencies.

IMPORTANT:
- DI is defined from the *same* frequency's directional pattern by comparing on-direction
    power to the spatial average power over all angles.
- This script assumes the gain-like column is expressed in dB (e.g., dBFS or gain_db).
    DI is computed using POWER averaging: mean(10^(dB/10)), and DI = 10*log10(P_dir / P_avg).

This avoids the common pitfall of using a separate "noise" measurement as an omnidirectional
reference (which is generally not equivalent to a diffuse-field spatial average).

Usage:
    python 2_Compute_DI.py

Parameters (configured manually at the top of the script):
    - NOISE_CSV_FILE: (optional) Path to a noise measurement CSV (legacy; not used by default)
    - SIGNAL_CSV_FILES: Dict mapping frequency (Hz) to signal CSV file path
    - MEASUREMENT_DIRECTIONS: List of directions (degrees) to extract from CSV files
    - OUTPUT_DIR: Directory to save results
    - GAIN_COLUMN_NAME: Name of the gain column in CSV files (typically 'gain_db' or 'rms_dbfs')
    - ANGLE_COLUMN_NAME: Name of the angle column in CSV files (typically 'angle_deg' or 'expected_angle')
    - INTERPOLATE_ANGLES: If True, interpolate gain values at exact requested directions
"""

from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION PARAMETERS (modify these manually)
# ============================================================================

BASE_DIR = "Python/Tests/mic-array-dev/"

# Path to noise measurement CSV (averaged across all directions in the file)
NOISE_CSV_FILE = BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_pink_noise.csv"

# Dictionary mapping frequency (Hz) to signal CSV file path
# Example: {1000: "path/to/1khz.csv", 2000: "path/to/2khz.csv", ...}
SIGNAL_CSV_FILES = {
    500: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_500_Hz.csv",
    1000: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_1000_Hz.csv",
    1500: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_1500_Hz.csv",
    1750: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_1750_Hz.csv",
    2000: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_2000_Hz.csv",
    2500: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_2500_Hz.csv",
    3000: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_3000_Hz.csv",
    3500: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_3500_Hz.csv",
    4000: BASE_DIR + "data/test_protocol/1_polar_pattern/1_square_65dB/polar_pattern_4000_Hz.csv",
}

# List of directions (degrees) to analyze
MEASUREMENT_DIRECTIONS = [0, 15, 30, 45]

# Output directory for results
OUTPUT_DIR = BASE_DIR + "data/test_protocol/2_directivity_index/1_square"

# Column names in CSV files (update if your CSVs use different column names)
# Prefer gain_db if present (output/input), otherwise fall back to rms_dbfs.
GAIN_COLUMN_NAME = "gain_db"
ANGLE_COLUMN_NAME = "expected_angle"

# If True, interpolate gain values at exact requested directions
# If False, find the closest direction in the CSV
INTERPOLATE_ANGLES = True

# Plot settings
PLOT_FIGSIZE = (12, 7)
PLOT_DPI = 100
PLOT_LINE_WIDTH = 2.0
PLOT_MARKER_SIZE = 8


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_dbfs(value: float, floor: float = 1e-10) -> float:
    """Convert linear value to dB FS."""
    return 20.0 * np.log10(max(float(value), floor))


def _power_mean_from_db_series(db_values: pd.Series) -> float:
    """Return mean power in linear domain from a series of dB values."""
    if db_values.empty:
        return float('nan')
    db = pd.to_numeric(db_values, errors='coerce').dropna().astype(float)
    if db.empty:
        return float('nan')
    # If values are in dB (amplitude dB or power dB), treating them as power-dB here is
    # the correct approach for DI: P_linear = 10^(dB/10), then spatial average in linear.
    p = 10.0 ** (db / 10.0)
    return float(p.mean())


def _find_csv_file(pattern: str) -> str | None:
    """Find the first file matching the glob pattern."""
    from glob import glob
    files = glob(pattern)
    return files[0] if files else None


def _interpolate_gain_at_angle(df: pd.DataFrame, target_angle: float, 
                               angle_col: str, gain_col: str) -> float | None:
    """
    Interpolate gain value at a specific angle using circular interpolation.
    
    Handles wraparound at 360° for circular angle data.
    """
    if df.empty or gain_col not in df.columns or angle_col not in df.columns:
        return None
    
    angles = df[angle_col].to_numpy(dtype=float)
    gains = df[gain_col].to_numpy(dtype=float)
    
    target_angle = float(target_angle) % 360.0
    
    # Try direct match first
    match_idx = np.argmin(np.abs(angles - target_angle))
    if abs(angles[match_idx] - target_angle) < 0.1:
        return float(gains[match_idx])
    
    # Circular interpolation: find two closest angles
    distances = np.minimum(np.abs(angles - target_angle), 360.0 - np.abs(angles - target_angle))
    sorted_idx = np.argsort(distances)
    
    if len(sorted_idx) >= 2:
        idx1, idx2 = sorted_idx[0], sorted_idx[1]
        angle1, angle2 = angles[idx1], angles[idx2]
        gain1, gain2 = gains[idx1], gains[idx2]
        
        # Compute circular distance and weight
        dist1 = min(abs(angle1 - target_angle), 360.0 - abs(angle1 - target_angle))
        dist2 = min(abs(angle2 - target_angle), 360.0 - abs(angle2 - target_angle))
        
        if dist1 + dist2 > 1e-6:
            weight1 = dist2 / (dist1 + dist2)
            return float(weight1 * gain1 + (1.0 - weight1) * gain2)
    
    return float(gains[match_idx]) if len(gains) > 0 else None


def _extract_gain_at_angle(df: pd.DataFrame, target_angle: float,
                           angle_col: str, gain_col: str,
                           interpolate: bool = True) -> float | None:
    """Extract or interpolate gain at a specific angle."""
    if interpolate:
        return _interpolate_gain_at_angle(df, target_angle, angle_col, gain_col)
    else:
        # Find closest angle
        if df.empty or angle_col not in df.columns or gain_col not in df.columns:
            return None
        angles = df[angle_col].to_numpy(dtype=float)
        gains = df[gain_col].to_numpy(dtype=float)
        target_angle = float(target_angle) % 360.0
        
        distances = np.minimum(
            np.abs(angles - target_angle),
            360.0 - np.abs(angles - target_angle)
        )
        closest_idx = np.argmin(distances)
        return float(gains[closest_idx]) if len(gains) > 0 else None


def compute_di_analysis(
    noise_csv_file: str | None = None,
    signal_csv_files: dict | None = None,
    measurement_directions: list | None = None,
    output_dir: str = "data/test_protocol/di_analysis",
    gain_column: str = "gain_db",
    angle_column: str = "expected_angle",
    interpolate_angles: bool = True,
    use_noise_reference: bool = False,
):
    """
    Compute directivity index across frequencies.
    
    Args:
        noise_csv_file: Path to noise measurement CSV (or glob pattern)
        signal_csv_files: Dict mapping frequency (Hz) to CSV file path
        measurement_directions: List of directions (degrees) to analyze
        output_dir: Directory to save results
        gain_column: Name of gain column in CSV files
        angle_column: Name of angle column in CSV files
        interpolate_angles: If True, interpolate at exact directions
    
    Returns:
        DataFrame with frequencies as rows and directions as columns (DI values)
    """
    noise_csv_file = noise_csv_file or NOISE_CSV_FILE
    signal_csv_files = signal_csv_files or SIGNAL_CSV_FILES
    measurement_directions = measurement_directions or MEASUREMENT_DIRECTIONS
    gain_column = gain_column or GAIN_COLUMN_NAME
    angle_column = angle_column or ANGLE_COLUMN_NAME
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("DIAnalysis")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    print(f"\n{'='*70}")
    print(f"Directivity Index Analysis")
    print(f"{'='*70}\n")
    
    noise_ref_power: float | None = None
    if use_noise_reference:
        print("Step 1: Reading noise measurement (legacy reference mode)...")
        noise_file = _find_csv_file(noise_csv_file)
        if not noise_file:
            print(f"ERROR: Could not find noise CSV matching pattern: {noise_csv_file}")
            return None
        try:
            df_noise = pd.read_csv(noise_file)
        except Exception as e:
            logger.error(f"Failed to read noise CSV: {e}")
            return None

        if gain_column not in df_noise.columns:
            # Fallback for older CSVs
            if 'rms_dbfs' in df_noise.columns:
                logger.warning(f"Column '{gain_column}' not found in noise CSV; falling back to 'rms_dbfs'.")
                gain_column = 'rms_dbfs'
            else:
                logger.error(f"Column '{gain_column}' not found in noise CSV. Available columns: {df_noise.columns.tolist()}")
                return None

        noise_ref_power = _power_mean_from_db_series(df_noise[gain_column])
        print(f"  Noise file: {Path(noise_file).name}")
        print(f"  Noise reference mean power (linear): {noise_ref_power:.6g}\n")

    # Step 2: Read signal CSV files and extract DI at requested directions
    print("Step 2: Reading signal measurements and computing DI...\n")
    
    di_results = {}  # frequency -> {direction -> DI_db}
    
    sorted_frequencies = sorted(signal_csv_files.keys())
    for freq in sorted_frequencies:
        signal_file = signal_csv_files[freq]
        
        try:
            df_signal = pd.read_csv(signal_file)
        except Exception as e:
            logger.warning(f"Failed to read signal CSV for {freq} Hz: {e}")
            continue
        
        effective_gain_col = gain_column
        if effective_gain_col not in df_signal.columns:
            if 'gain_db' in df_signal.columns:
                effective_gain_col = 'gain_db'
                logger.warning(f"Column '{gain_column}' not found for {freq} Hz; using 'gain_db'.")
            elif 'rms_dbfs' in df_signal.columns:
                effective_gain_col = 'rms_dbfs'
                logger.warning(f"Column '{gain_column}' not found for {freq} Hz; using 'rms_dbfs'.")
            else:
                logger.warning(f"No usable gain column found for {freq} Hz. Available: {df_signal.columns.tolist()}")
                continue

        # Spatial-average reference power from this frequency's own pattern (preferred DI definition).
        if use_noise_reference:
            if noise_ref_power is None:
                logger.warning(f"Noise reference mode enabled but no noise reference is available; skipping {freq} Hz.")
                continue
            ref_power = float(noise_ref_power)
        else:
            ref_power = float(_power_mean_from_db_series(df_signal[effective_gain_col]))

        if not np.isfinite(ref_power) or ref_power <= 0.0:
            logger.warning(f"Invalid reference power for {freq} Hz; skipping.")
            continue
        
        ref_mode = "noise" if use_noise_reference else "spatial-mean"
        print(f"  {freq} Hz: {Path(signal_file).name} (ref={ref_mode})")
        di_results[freq] = {}
        
        for direction in measurement_directions:
            signal_gain_db = _extract_gain_at_angle(
                df_signal,
                target_angle=direction,
                angle_col=angle_column,
                gain_col=effective_gain_col,
                interpolate=interpolate_angles
            )
            
            if signal_gain_db is None:
                logger.warning(f"    Could not extract gain at {direction}°")
                continue
            
            # DI uses power ratio: P_dir / P_avg, where P_linear = 10^(dB/10)
            p_dir = 10.0 ** (float(signal_gain_db) / 10.0)
            di_db = 10.0 * np.log10(max(p_dir / ref_power, 1e-20))
            di_results[freq][direction] = di_db

            print(f"    {direction:6.1f}°: {effective_gain_col} {signal_gain_db:7.2f} dB → DI {di_db:+7.2f} dB")
        
        print()
    
    # Step 3: Create results DataFrame
    print("Step 3: Organizing results...\n")
    
    # Convert to DataFrame: frequencies as rows, directions as columns
    df_di = pd.DataFrame.from_dict(di_results, orient='index')
    df_di.index.name = 'Frequency (Hz)'
    df_di.columns.name = 'Direction (°)'
    df_di = df_di.sort_index()
    
    print(f"  Total frequencies analyzed: {len(df_di)}")
    print(f"  Directions: {df_di.columns.tolist()}")
    print(f"  Shape: {df_di.shape}\n")
    
    # Step 4: Save to CSV
    test_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_file = output_path / f"di_analysis_{test_timestamp}.csv"
    df_di.to_csv(csv_file)
    print(f"Results saved to: {csv_file}\n")
    
    # Step 5: Generate plot
    print("Step 5: Generating plot...\n")
    
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    
    frequencies = df_di.index.to_numpy(dtype=float)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(x) for x in np.linspace(0, 1, max(1, len(df_di.columns)))]
    
    for col_idx, direction in enumerate(df_di.columns):
        di_values = df_di[direction].to_numpy(dtype=float)
        
        # Skip if all NaN
        if np.all(np.isnan(di_values)):
            continue
        
        ax.plot(
            frequencies,
            di_values,
            marker='o',
            linewidth=PLOT_LINE_WIDTH,
            markersize=PLOT_MARKER_SIZE,
            label=f"{direction:.0f}°",
            color=colors[col_idx % len(colors)]
        )
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Directivity Index (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Directivity Index vs Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Reference DI benchmarks
    # Use y-axis transform so X is in axes fraction (stable even with log-x).
    ref_lines = [
        (0.0, 'Omnidirectionnal (0dB)'),
        (6.0, 'Cardioid (6dB)'),
        (8.0, 'Typical Array (8dB)'),
    ]
    for y, label in ref_lines:
        ax.axhline(y=y, color='black', linestyle=':', linewidth=1.5, alpha=0.75, zorder=0)
        ax.text(
            0.15,
            y,
            label,
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),
        )

    ax.legend(title='Direction', loc='best', fontsize=10)
    
    # Set log scale for frequency if spanning multiple octaves
    if len(frequencies) > 1 and frequencies[-1] / frequencies[0] > 4:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    png_file = output_path / f"di_analysis_{test_timestamp}.png"
    fig.savefig(png_file, dpi=PLOT_DPI)
    print(f"Plot saved to: {png_file}\n")
    
    plt.close(fig)
    
    # Step 6: Print summary statistics
    print("Step 6: Summary Statistics\n")
    print(df_di.to_string())
    print()
    
    print(f"{'='*70}")
    print(f"Analysis complete!")
    print(f"{'='*70}\n")
    
    return df_di


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute directivity index (DI) across frequencies"
    )
    parser.add_argument(
        "--noise-csv",
        type=str,
        default=NOISE_CSV_FILE,
        help="Path or glob pattern for noise measurement CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory (default: data/test_protocol/di_analysis)"
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable angle interpolation (use nearest neighbor)"
    )
    parser.add_argument(
        "--use-noise-reference",
        action="store_true",
        help="LEGACY: compute DI relative to the provided noise CSV instead of each frequency's spatial mean."
    )
    
    args = parser.parse_args()
    
    if not SIGNAL_CSV_FILES:
        print("ERROR: SIGNAL_CSV_FILES is empty. Please configure it in the script.")
        print("Example:")
        print("  SIGNAL_CSV_FILES = {")
        print('      1000: "path/to/1khz_signal.csv",')
        print('      2000: "path/to/2khz_signal.csv",')
        print("  }")
        sys.exit(1)
    
    df_results = compute_di_analysis(
        noise_csv_file=args.noise_csv,
        signal_csv_files=SIGNAL_CSV_FILES,
        measurement_directions=MEASUREMENT_DIRECTIONS,
        output_dir=args.output,
        gain_column=GAIN_COLUMN_NAME,
        angle_column=ANGLE_COLUMN_NAME,
        interpolate_angles=not args.no_interpolate,
        use_noise_reference=args.use_noise_reference,
    )
