import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_directivity(csv_file, save_plot=True, save_location=None, reference_max_rms=None, min_scale=None, flip_axis=False):
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
        flip_axis: If True, flip the polar axis vertically so 0° points downward (default: False, 0° points upward).
    """
    # Load data
    df = pd.read_csv(csv_file)
    
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
    if 'peak_dbfs' in df.columns:
        df['peak_dbfs'] = 20 * np.log10(np.maximum(df['peak_level'], min_level) / ref_value)
    
    # Detect CSV structure
    has_doa_angle = 'doa_angle' in df.columns
    has_reference_rms = 'reference_rms_used' in df.columns
    has_peak_dbfs = 'peak_dbfs' in df.columns
    
    # Check if relative_angle is available (when beamformer is locked)
    use_relative_angle = 'relative_angle' in df.columns and df['relative_angle'].notna().any()
    angle_column = 'relative_angle' if use_relative_angle else 'expected_angle'
    angle_label = 'Relative Angle (from locked DOA)' if use_relative_angle else 'Angle'
    
    # Check for peaks - ReSpeaker uses 'peaks' column
    peaks_available = 'peaks' in df.columns and df['peaks'].any()
    
    # Get reference RMS info for title
    reference_info = ""
    ref_rms = ref_value
    ref_dbv = 20 * np.log10(ref_rms)
    reference_info = f"\nReference ({ref_source}): {ref_rms:.6f} RMS ({ref_dbv:.1f} dB)"
    
    # Determine theta zero location and direction based on flip_axis
    theta_zero_loc = 'S' if flip_axis else 'N'
    theta_dir = 1 if flip_axis else -1
    
    # Use multi-plot layout only for ReSpeaker with peaks
    if peaks_available and has_doa_angle:
        print("Peak measurements detected in data. Plotting both RMS and Peak levels.")

        # Create figure with polar plot
        fig = plt.figure(figsize=(12, 10))

        # Main polar plot for RMS level
        ax1 = plt.subplot(221, projection='polar')
        
        # Close the loop by adding first point at the end
        angles_closed = np.append(df[angle_column].values, df[angle_column].iloc[0])
        rms_closed = np.append(df['rms_dbfs'].values, df['rms_dbfs'].iloc[0])
        angles_rad = np.deg2rad(angles_closed)
        
        ax1.plot(angles_rad, rms_closed, 'b-o', linewidth=2, markersize=4, label='RMS')
        ax1.set_theta_zero_location(theta_zero_loc)
        ax1.set_theta_direction(theta_dir)
        ax1.set_title('RMS Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        if min_scale is not None:
            ax1.set_ylim([min_scale, 0])

        # Polar plot for Peak level
        ax2 = plt.subplot(222, projection='polar')
        
        # Close the loop by adding first point at the end
        peak_closed = np.append(df['peak_dbfs'].values, df['peak_dbfs'].iloc[0])
        
        ax2.plot(angles_rad, peak_closed, 'r-s', linewidth=2, markersize=4, label='Peak')
        ax2.set_theta_zero_location(theta_zero_loc)
        ax2.set_theta_direction(theta_dir)
        ax2.set_title('Peak Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        if min_scale is not None:
            ax2.set_ylim([min_scale, 0])

        # Cartesian plot comparing RMS and Peak
        ax3 = plt.subplot(223)
        ax3.plot(df[angle_column], df['rms_dbfs'], 'b-o', linewidth=2, markersize=4, label='RMS')
        ax3.plot(df[angle_column], df['peak_dbfs'], 'r-s', linewidth=2, markersize=4, label='Peak')
        ax3.set_xlabel(f'{angle_label} (degrees)', fontsize=10)
        ax3.set_ylabel('Level (dBFS)', fontsize=10)
        ax3.set_title('Signal Levels vs Angle', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim([0, 360])
        if min_scale is not None:
            ax3.set_ylim([min_scale, 0])

        # DOA angle plot (or relative angle plot if beamformer is locked)
        ax4 = plt.subplot(224)
        if use_relative_angle:
            ax4.plot(df[angle_column], df['rms_dbfs'], 'b-o', linewidth=2, markersize=4)
            ax4.set_xlabel(f'{angle_label} (degrees)', fontsize=10)
            ax4.set_ylabel('RMS Level (dBFS)', fontsize=10)
            ax4.set_title('Directivity Pattern (Relative to Locked DOA)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 360])
            if min_scale is not None:
                ax4.set_ylim([min_scale, 0])
        elif has_doa_angle:
            ax4.plot(df['expected_angle'], df['doa_angle'], 'g-^', linewidth=2, markersize=4)
            ax4.set_xlabel('Expected Angle (degrees)', fontsize=10)
            ax4.set_ylabel('Detected DOA Angle (degrees)', fontsize=10)
            ax4.set_title('Direction of Arrival Detection', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 360])
            ax4.set_ylim([0, 360])
        else:
            # For standard mic, show cartesian plot instead
            ax4.plot(df[angle_column], df['rms_dbfs'], 'b-o', linewidth=2, markersize=4)
            ax4.set_xlabel(f'{angle_label} (degrees)', fontsize=10)
            ax4.set_ylabel('RMS Level (dB)', fontsize=10)
            ax4.set_title('Directivity Pattern', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 360])
            if min_scale is not None:
                ax4.set_ylim([min_scale, 0])

        # Add overall title with metadata
        test_name = Path(csv_file).stem
        device_type = 'ReSpeaker' if has_doa_angle else 'Microphone'
        fig.suptitle(f'{device_type} Directivity Pattern - {test_name}{reference_info}', 
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        if save_plot:
            if save_location:
                save_dir = Path(save_location)
                # If save_location is a file, use its parent directory instead
                if save_dir.is_file() or str(save_dir).endswith('.csv'):
                    save_dir = save_dir.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                output_file = save_dir / f"{test_name}.png"
            else:
                output_file = Path(csv_file).with_suffix('.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")

        plt.show()
        
    else:
        print("Peak measurements not available in data. Plotting RMS only.")
        
        # Create figure with single polar plot for RMS level
        fig = plt.figure(figsize=(8, 6))
        ax1 = plt.subplot(111, projection='polar')
        
        # Close the loop by adding first point at the end
        angles_closed = np.append(df[angle_column].values, df[angle_column].iloc[0])
        rms_closed = np.append(df['rms_dbfs'].values, df['rms_dbfs'].iloc[0])
        angles_rad = np.deg2rad(angles_closed)
        
        ax1.plot(angles_rad, rms_closed, 'b-o', linewidth=2, markersize=4)
        ax1.set_theta_zero_location(theta_zero_loc)
        ax1.set_theta_direction(theta_dir)
        title = 'RMS Level (dBFS)'
        if use_relative_angle:
            title += ' - Relative to Locked DOA'
        ax1.set_title(title, pad=20, fontsize=12, fontweight='bold')
        ax1.grid(True)
        if min_scale is not None:
            ax1.set_ylim([min_scale, 0])

        # Add overall title with metadata
        test_name = Path(csv_file).stem
        device_type = 'ReSpeaker' if has_doa_angle else 'Microphone'
        fig.suptitle(f'{device_type} Directivity Pattern - {test_name}{reference_info}', 
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        if save_plot:
            if save_location:
                save_dir = Path(save_location)
                # If save_location is a file, use its parent directory instead
                if save_dir.is_file() or str(save_dir).endswith('.csv'):
                    save_dir = save_dir.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                output_file = save_dir / f"{test_name}.png"
            else:
                output_file = Path(csv_file).with_suffix('.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")

        plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Directivity Statistics:")
    print("="*60)
    
    # Show reference info
    ref_dbv = 20 * np.log10(ref_value)
    print(f"Reference RMS ({ref_source}): {ref_value:.6f} ({ref_dbv:.1f} dB)")
    print()
    
    db_unit = "dBFS" if has_doa_angle else "dB"
    print(f"RMS Level:")
    print(f"  Mean:   {df['rms_dbfs'].mean():.2f} {db_unit}")
    print(f"  Min:    {df['rms_dbfs'].min():.2f} {db_unit} at {df.loc[df['rms_dbfs'].idxmin(), angle_column]:.1f}°")
    print(f"  Max:    {df['rms_dbfs'].max():.2f} {db_unit} at {df.loc[df['rms_dbfs'].idxmax(), angle_column]:.1f}°")
    print(f"  Range:  {df['rms_dbfs'].max() - df['rms_dbfs'].min():.2f} dB")
    if peaks_available:
        print(f"\nPeak Level:")
        print(f"  Mean:   {df['peak_dbfs'].mean():.2f} {db_unit}")
        print(f"  Min:    {df['peak_dbfs'].min():.2f} {db_unit} at {df.loc[df['peak_dbfs'].idxmin(), angle_column]:.1f}°")
        print(f"  Max:    {df['peak_dbfs'].max():.2f} {db_unit} at {df.loc[df['peak_dbfs'].idxmax(), angle_column]:.1f}°")
        print(f"  Range:  {df['peak_dbfs'].max() - df['peak_dbfs'].min():.2f} dB")
    if use_relative_angle and 'locked_doa' in df.columns:
        print(f"\nBeamformer locked at: {df['locked_doa'].iloc[0]:.0f}°")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ReSpeaker directivity measurements')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with measurements')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')
    parser.add_argument('--save-location', type=str, default=None, help='Optional directory to save plot (default: same directory as CSV file)')
    parser.add_argument('--reference-max-rms', type=float, default=None,
                        help='External reference max RMS value for normalizing dB (allows comparing multiple measurements)')
    parser.add_argument('--min-scale', type=float, default=None,
                        help='Minimum dB value for y-axis scaling (e.g., -30 for -30 dB floor). Allows consistent comparison between plots.')
    parser.add_argument('--flip-axis', action='store_true',
                        help='Flip the polar axis vertically so 0° points downward instead of upward (default: 0° points upward)')
    
    args = parser.parse_args()
    
    plot_directivity(args.csv_file, save_plot=not args.no_save, save_location=args.save_location, reference_max_rms=args.reference_max_rms, min_scale=args.min_scale, flip_axis=args.flip_axis)

    
    # Example:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\respeaker\plot_directivity.py .\Python\Tests\mic-array-dev\data\directivity\Multipass_70cm\directivity_multipass_averaged_300_600.csv 
    #
    # The script automatically detects the maximum RMS as the baseline (0 dB reference).
    # To use the same baseline across multiple measurements:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\respeaker\plot_directivity.py .\Python\Tests\mic-array-dev\data\directivity\Multipass_70cm\another_measurement.csv --reference-max-rms 47.22
    #
    # To set a fixed minimum dB scale (e.g., -30 dB) for consistent comparison:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\respeaker\plot_directivity.py .\Python\Tests\mic-array-dev\data\directivity\Multipass_70cm\directivity_multipass_averaged_300_600.csv --min-scale -30
    #
    # To flip verticaly the axis:
    # .venv\Scripts\python.exe .\Python\Tests\mic-array-dev\respeaker\plot_directivity.py .\Python\Tests\mic-array-dev\data\directivity\Multipass_70cm\directivity_multipass_averaged_300_600.csv --flip-axis