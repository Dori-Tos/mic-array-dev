import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
from pathlib import Path
import argparse
from typing import Any


def _close_loop(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return np.concatenate((values, np.array([float(values[0])], dtype=float)))


def plot_directivity(
    csv_file,
    save_plot=True,
    save_location=None,
    reference_max_rms=None,
    min_scale=None,
    quarter_graph=False,
    half_rotation=False,
    side=False,
    source="rms",
    show_plot=True,
):
    """
    Plot directivity pattern from measurement data.
    
    Args:
        csv_file: Path to CSV file with directivity measurements
        save_plot: Whether to save plot to file (default: True)
        save_location: Optional path to save plot (if None, saves in same directory as CSV)
        reference_max_rms: Optional external reference RMS value for recalculating dB values
                          (allows comparing multiple measurements on same scale).
                          If None, automatically uses the maximum RMS from the data as baseline.
        min_scale: Optional minimum dB value for graph scaling (e.g., -30 for -30 dB floor).
                   If provided, all plots will use this as the minimum y-axis value.
                   If None, y-axis uses autoscaling based on data.
        quarter_graph: If True, mirror front-quarter measurements to the opposite front side
                  (left-right symmetry) and show only the front hemisphere.
        half_rotation: If True, handle half-pattern measurements.
                  Combine with --side for side pattern (0-180° → 0-360° with left-right mirroring).
                  Without --side, shows front pattern (-90 to 90°, no mirroring).
        side: If True with --half-rotation, apply side pattern mode (0-180°).
              Data is mirrored left-right (around vertical axis) to create full circle.
              If False with --half-rotation, apply front pattern mode (-90 to 90°, no mirroring).
        source: Which values to plot for the main directivity curve: "rms" uses the
                smoothed RMS-derived values, "gain" uses the unedited gain_db column.
        show_plot: If True, display the plot window after creating the figure.
    """
    # Load data
    df = pd.read_csv(csv_file)
    source = source.lower().strip()
    if source not in ("rms", "gain"):
        raise ValueError("source must be either 'rms' or 'gain'")

    main_value_column = "rms_dbfs" if source == "rms" else "gain_db"
    main_value_label = "RMS Level (dBFS)" if source == "rms" else "Unedited Gain (dB)"
    plot_name_suffix = "_unedited" if source == "gain" else ""
    title_suffix = "(Unedited)" if source == "gain" else ""
    
    # Determine reference value for dB calculation
    min_level = 1e-10
    if reference_max_rms is not None:
        # External reference provided by user
        ref_value = reference_max_rms
        ref_source = "external"
    else:
        # Automatically find maximum RMS as baseline
        ref_value = df['rms_level'].max()
        ref_source = "auto-detected (max RMS)"
    
    # Recalculate dB values using the reference
    df['rms_dbfs'] = 20 * np.log10(np.maximum(df['rms_level'], min_level) / ref_value)
    if main_value_column not in df.columns:
        raise ValueError(f"source='{source}' requires column '{main_value_column}' in the CSV")

    # Detect CSV structure
    has_doa_angle = 'doa_angle' in df.columns
    # Check if relative_angle is available (when beamformer is locked)
    use_relative_angle = 'relative_angle' in df.columns and df['relative_angle'].notna().any()
    angle_column = 'relative_angle' if use_relative_angle else 'expected_angle'
    angle_label = 'Relative Angle (from locked DOA)' if use_relative_angle else 'Angle'

    # Handle half-rotation modes (side or front pattern)
    if half_rotation:
        if side:
            # Side pattern (0-180°) with left-right mirroring around the vertical axis
            # Mirror formula: 360° - angle creates the opposite side
            # Original data: 0° to 180°, Mirrored: 180° to 360° (excludes duplicates at 0° and 180°)
            mirrored = df.copy()
            mirrored_angles = 360.0 - mirrored[angle_column]
            mirrored[angle_column] = mirrored_angles
            # Avoid duplicate points at 0° and 180° when mirroring
            mirrored = mirrored[~np.isclose(mirrored[angle_column], df[angle_column])]
            df = pd.concat([df, mirrored], ignore_index=True)
            df = df.sort_values(angle_column).reset_index(drop=True)
            theta_ticks = np.arange(0, 361, 45)
        else:
            # Front pattern (-90 to 90°) without mirroring
            # Convert to continuous front-domain: 0° stays 0°, range is -90 to 90
            df[angle_column] = ((df[angle_column] + 180.0) % 360.0) - 180.0
            df = df[(df[angle_column] >= -90.0) & (df[angle_column] <= 90.0)]
            df = df.sort_values(angle_column).reset_index(drop=True)
            theta_ticks = np.arange(-90, 91, 30)
    # Optional quarter-front mirror mode (existing):
    # measured front quarter (typically 0..90) is mirrored to 360-angle (270..360)
    # to visualize left-right symmetry while excluding the back hemisphere.
    elif quarter_graph:
        mirrored = df.copy()
        mirrored[angle_column] = (360.0 - mirrored[angle_column]) % 360.0
        # Avoid duplicate points at 0° and 180° when mirroring.
        mirrored = mirrored[~np.isclose(mirrored[angle_column], df[angle_column])]
        df = pd.concat([df, mirrored], ignore_index=True)
        # Convert to a continuous front-hemisphere angle domain to avoid wrap gaps at 0°.
        # Example: 350° -> -10°, 270° -> -90°, 0..90° unchanged.
        df[angle_column] = ((df[angle_column] + 180.0) % 360.0) - 180.0
        df = df[(df[angle_column] >= -90.0) & (df[angle_column] <= 90.0)]
        df = df.sort_values(angle_column).reset_index(drop=True)
        theta_ticks = np.arange(-90, 91, 30)
    else:
        # Full circle mode (default)
        theta_ticks = np.arange(0, 360, 45)

    # Build integer-only dB ticks to remove decimal labels like 0.0, -2.5, ...
    data_min_db = float(np.floor(df[main_value_column].min()))
    if min_scale is not None:
        data_min_db = float(np.floor(min_scale))
    db_floor = int(5 * np.floor(data_min_db / 5.0))
    db_ticks = np.arange(db_floor, 1, 5)
    
    # Get reference RMS info for title
    reference_info = ""
    ref_rms = ref_value
    ref_dbv = 20 * np.log10(ref_rms)
    reference_info = f"\nReference ({ref_source}): {ref_rms:.6f} RMS ({ref_dbv:.1f} dB)"
    # Build the single directivity plot using the selected source values only.
    fig = plt.figure(figsize=(8, 6))
    ax1: Any = plt.subplot(111, projection='polar')

    if quarter_graph or half_rotation:
        angles_plot = df[angle_column].to_numpy(dtype=float)
        value_plot = df[main_value_column].to_numpy(dtype=float)
    else:
        angles_plot = _close_loop(df[angle_column])
        value_plot = _close_loop(df[main_value_column])
    angles_rad = np.deg2rad(angles_plot)

    ax1.plot(angles_rad, value_plot, 'b-o', linewidth=2, markersize=4)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_thetagrids(theta_ticks)
    if quarter_graph:
        ax1.set_thetamin(-90)
        ax1.set_thetamax(90)
    elif half_rotation and not side:
        ax1.set_thetamin(-90)
        ax1.set_thetamax(90)
    title = main_value_label
    if use_relative_angle:
        title += ' - Relative to Locked DOA'
    ax1.set_title(title, pad=20, fontsize=12, fontweight='bold')
    ax1.grid(True)
    if min_scale is not None:
        ax1.set_ylim((min_scale, 0))
    ax1.set_yticks(db_ticks)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Add overall title with metadata
    test_name = Path(csv_file).stem + plot_name_suffix
    device_type = 'ReSpeaker' if has_doa_angle else 'Microphone'
    fig.suptitle(f'{device_type} Directivity Pattern - {test_name}{title_suffix}{reference_info}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    # Save plot
    if save_plot:
        if save_location:
            save_dir = Path(save_location)
            if save_dir.is_file() or str(save_dir).endswith('.csv'):
                save_dir = save_dir.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            output_file = save_dir / f"{test_name}.png"
        else:
            output_file = Path(csv_file).with_name(f"{Path(csv_file).stem}{plot_name_suffix}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")

    if show_plot:
        plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Directivity Statistics:")
    print("="*60)
    
    # Show reference info
    ref_dbv = 20 * np.log10(ref_value)
    print(f"Reference RMS ({ref_source}): {ref_value:.6f} ({ref_dbv:.1f} dB)")
    print()
    
    db_unit = "dBFS" if source == "rms" else "dB"
    print(f"{main_value_label}:")
    print(f"  Mean:   {df[main_value_column].mean():.2f} {db_unit}")
    print(f"  Min:    {df[main_value_column].min():.2f} {db_unit} at {df.loc[df[main_value_column].idxmin(), angle_column]:.1f}°")
    print(f"  Max:    {df[main_value_column].max():.2f} {db_unit} at {df.loc[df[main_value_column].idxmax(), angle_column]:.1f}°")
    print(f"  Range:  {df[main_value_column].max() - df[main_value_column].min():.2f} dB")
    if use_relative_angle and 'locked_doa' in df.columns:
        print(f"\nBeamformer locked at: {df['locked_doa'].iloc[0]:.0f}°")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ReSpeaker directivity measurements')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with measurements')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot window after creating the graph')
    parser.add_argument('--save-location', type=str, default=None, help='Optional directory to save plot (default: same directory as CSV file)')
    parser.add_argument('--source', choices=('rms', 'gain'), default='rms',
                        help='Choose which values to plot: rms uses the smoothed RMS-derived values, gain uses the unedited gain_db')
    parser.add_argument('--reference-max-rms', type=float, default=None,
                        help='External reference max RMS value for normalizing dB (allows comparing multiple measurements)')
    parser.add_argument('--min-scale', type=float, default=None,
                        help='Minimum dB value for y-axis scaling (e.g., -30 for -30 dB floor). Allows consistent comparison between plots.')
    parser.add_argument('--quarter-graph', action='store_true',
                        help='Mirror front-quarter data left-right and plot only the front hemisphere')
    parser.add_argument('--half-rotation', action='store_true',
                        help='Handle half-pattern measurements (side or front pattern)')
    parser.add_argument('--side', action='store_true',
                        help='With --half-rotation: apply side pattern (0-180°) with left-right mirroring to create full circle. Without --half-rotation: ignored. Default is front pattern (-90 to 90°)')
    
    args = parser.parse_args()
    
    plot_directivity(
        args.csv_file,
        save_plot=not args.no_save,
        save_location=args.save_location,
        reference_max_rms=args.reference_max_rms,
        min_scale=args.min_scale,
        quarter_graph=args.quarter_graph,
        half_rotation=args.half_rotation,
        side=args.side,
        source=args.source,
        show_plot=not args.no_show,
    )

    
    # Example:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\Python\Tests\mic-array-dev\data\test_protocol\1_Polar_Pattern\1_Square\polar_pattern_averaged_2026-04-13_17-35-20.csv 
    #
    # The script automatically detects the maximum RMS as the baseline (0 dB reference).
    # To use the same baseline across multiple measurements:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\data\...\polar_pattern.csv --reference-max-rms 47.22
    #
    # Mirror the front quarter pattern (0-90°) to left-right symmetry:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\data\...\polar_pattern.csv --quarter-graph
    #
    # Plot half-rotation front pattern (-90 to 90°, no mirroring):
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\data\...\polar_pattern.csv --half-rotation
    #
    # Plot half-rotation side pattern (0-180° with left-right mirroring to show full circle):
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\data\...\polar_pattern.csv --half-rotation --side
    #
    # Combined options with min scale:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\prototype\test_protocol\1_Plot_Directivity.py .\data\...\polar_pattern.csv --half-rotation --side --min-scale -30 