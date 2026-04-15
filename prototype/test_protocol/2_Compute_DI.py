"""
Directivity Index (DI) Frequency Analysis
==========================================

Computes the directivity index of an array across multiple frequencies by:
1. Reading a noise CSV file and averaging gain across all directions (omnidirectional reference)
2. Reading signal CSV files for each frequency and extracting gains at specified directions
3. Computing DI = 20 * log10(signal_gain / noise_average) for each direction
4. Generating a frequency-dependent DI plot (DI vs frequency for each direction)
5. Exporting results to CSV and PNG

Usage:
    python 2_Compute_DI.py

Parameters (configured manually at the top of the script):
    - NOISE_CSV_FILE: Path to the noise measurement CSV
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

# Path to noise measurement CSV (averaged across all directions in the file)
NOISE_CSV_FILE = "data/test_protocol/di_noise/di_noise_averaged_*.csv"

# Dictionary mapping frequency (Hz) to signal CSV file path
# Example: {1000: "path/to/1khz.csv", 2000: "path/to/2khz.csv", ...}
SIGNAL_CSV_FILES = {
    # 500: "data/test_protocol/di_signal/snr_signal_averaged_500hz.csv",
    # 1000: "data/test_protocol/di_signal/snr_signal_averaged_1khz.csv",
    # 2000: "data/test_protocol/di_signal/snr_signal_averaged_2khz.csv",
    # 4000: "data/test_protocol/di_signal/snr_signal_averaged_4khz.csv",
}

# List of directions (degrees) to analyze
MEASUREMENT_DIRECTIONS = [0, 15, 30, 45, 90, 180, 270, 315, 330, 345]

# Output directory for results
OUTPUT_DIR = "data/test_protocol/di_analysis"

# Column names in CSV files (update if your CSVs use different column names)
GAIN_COLUMN_NAME = "gain_db"      # or "rms_dbfs" depending on your CSV
ANGLE_COLUMN_NAME = "angle_deg"   # or "expected_angle"

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
    
    angles = df[angle_col].values
    gains = df[gain_col].values
    
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
        angles = df[angle_col].values
        gains = df[gain_col].values
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
    angle_column: str = "angle_deg",
    interpolate_angles: bool = True,
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
    
    # Step 1: Read noise CSV and compute omnidirectional reference
    print("Step 1: Reading noise measurement...")
    
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
        logger.error(f"Column '{gain_column}' not found in noise CSV. Available columns: {df_noise.columns.tolist()}")
        return None
    
    # Average gain across all directions (omnidirectional reference)
    noise_gain_avg_linear = 10.0 ** (df_noise[gain_column].mean() / 20.0)
    noise_gain_avg_db = _safe_dbfs(noise_gain_avg_linear)
    
    print(f"  Noise file: {Path(noise_file).name}")
    print(f"  Omnidirectional noise reference: {noise_gain_avg_db:.2f} dBFS (linear: {noise_gain_avg_linear:.4f})")
    print(f"  Directions in noise CSV: {df_noise[angle_column].min():.1f}° to {df_noise[angle_column].max():.1f}°\n")
    
    # Step 2: Read signal CSV files and extract gains at requested directions
    print("Step 2: Reading signal measurements at requested directions...\n")
    
    di_results = {}  # frequency -> {direction -> DI_db}
    
    sorted_frequencies = sorted(signal_csv_files.keys())
    for freq in sorted_frequencies:
        signal_file = signal_csv_files[freq]
        
        try:
            df_signal = pd.read_csv(signal_file)
        except Exception as e:
            logger.warning(f"Failed to read signal CSV for {freq} Hz: {e}")
            continue
        
        if gain_column not in df_signal.columns:
            logger.warning(f"Column '{gain_column}' not found in signal CSV for {freq} Hz")
            continue
        
        print(f"  {freq} Hz: {Path(signal_file).name}")
        di_results[freq] = {}
        
        for direction in measurement_directions:
            signal_gain_db = _extract_gain_at_angle(
                df_signal,
                target_angle=direction,
                angle_col=angle_column,
                gain_col=gain_column,
                interpolate=interpolate_angles
            )
            
            if signal_gain_db is None:
                logger.warning(f"    Could not extract gain at {direction}°")
                continue
            
            # Convert to linear domain for DI calculation
            signal_gain_linear = 10.0 ** (signal_gain_db / 20.0)
            
            # DI = 20 * log10(signal_gain / noise_gain)
            di_db = 20.0 * np.log10(max(signal_gain_linear / noise_gain_avg_linear, 1e-10))
            di_results[freq][direction] = di_db
            
            print(f"    {direction:6.1f}°: Signal {signal_gain_db:7.2f} dBFS → DI {di_db:+7.2f} dB")
        
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
    
    frequencies = df_di.index.values
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_di.columns)))
    
    for col_idx, direction in enumerate(df_di.columns):
        di_values = df_di[direction].values
        
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
    )
