import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_directivity(csv_file, save_plot=True, save_location=None):
    """
    Plot directivity pattern from measurement data.
    
    Args:
        csv_file: Path to CSV file with directivity measurements
        save_plot: Whether to save plot to file (default: True)
        save_location: Optional path to save plot (if None, saves in same directory as CSV)
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Check if relative_angle is available (when beamformer is locked)
    use_relative_angle = 'relative_angle' in df.columns and df['relative_angle'].notna().any()
    angle_column = 'relative_angle' if use_relative_angle else 'expected_angle'
    angle_label = 'Relative Angle (from locked DOA)' if use_relative_angle else 'Angle'
    
    peaks_available = 'peaks' in df.columns and df['peaks'].any()
    
    if peaks_available:
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
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title('RMS Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax1.grid(True)
        ax1.legend(loc='upper right')

        # Polar plot for Peak level
        ax2 = plt.subplot(222, projection='polar')
        
        # Close the loop by adding first point at the end
        peak_closed = np.append(df['peak_dbfs'].values, df['peak_dbfs'].iloc[0])
        
        ax2.plot(angles_rad, peak_closed, 'r-s', linewidth=2, markersize=4, label='Peak')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('Peak Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax2.grid(True)
        ax2.legend(loc='upper right')

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

        # DOA angle plot (or relative angle plot if beamformer is locked)
        ax4 = plt.subplot(224)
        if use_relative_angle:
            ax4.plot(df[angle_column], df['rms_dbfs'], 'b-o', linewidth=2, markersize=4)
            ax4.set_xlabel(f'{angle_label} (degrees)', fontsize=10)
            ax4.set_ylabel('RMS Level (dBFS)', fontsize=10)
            ax4.set_title('Directivity Pattern (Relative to Locked DOA)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 360])
        else:
            ax4.plot(df['expected_angle'], df['doa_angle'], 'g-^', linewidth=2, markersize=4)
            ax4.set_xlabel('Expected Angle (degrees)', fontsize=10)
            ax4.set_ylabel('Detected DOA Angle (degrees)', fontsize=10)
            ax4.set_title('Direction of Arrival Detection', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim([0, 360])
            ax4.set_ylim([0, 360])

        # Add overall title with metadata
        test_name = Path(csv_file).stem
        fig.suptitle(f'ReSpeaker Directivity Pattern - {test_name}', 
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        if save_plot:
            if save_location:
                output_file = Path(save_location) / f"{test_name}.png"
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
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        title = 'RMS Level (dBFS)'
        if use_relative_angle:
            title += ' - Relative to Locked DOA'
        ax1.set_title(title, pad=20, fontsize=12, fontweight='bold')
        ax1.grid(True)

        # Add overall title with metadata
        test_name = Path(csv_file).stem
        fig.suptitle(f'ReSpeaker Directivity Pattern - {test_name}', 
                     fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        if save_plot:
            if save_location:
                output_file = Path(save_location) / f"{test_name}.png"
            else:
                output_file = Path(csv_file).with_suffix('.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_file}")

        plt.show()

    # Print statistics
    print("\n" + "="*60)
    print("Directivity Statistics:")
    print("="*60)
    print(f"RMS Level:")
    print(f"  Mean:   {df['rms_dbfs'].mean():.2f} dBFS")
    print(f"  Min:    {df['rms_dbfs'].min():.2f} dBFS at {df.loc[df['rms_dbfs'].idxmin(), angle_column]:.1f}°")
    print(f"  Max:    {df['rms_dbfs'].max():.2f} dBFS at {df.loc[df['rms_dbfs'].idxmax(), angle_column]:.1f}°")
    print(f"  Range:  {df['rms_dbfs'].max() - df['rms_dbfs'].min():.2f} dB")
    if peaks_available:
        print(f"\nPeak Level:")
        print(f"  Mean:   {df['peak_dbfs'].mean():.2f} dBFS")
        print(f"  Min:    {df['peak_dbfs'].min():.2f} dBFS at {df.loc[df['peak_dbfs'].idxmin(), angle_column]:.1f}°")
        print(f"  Max:    {df['peak_dbfs'].max():.2f} dBFS at {df.loc[df['peak_dbfs'].idxmax(), angle_column]:.1f}°")
        print(f"  Range:  {df['peak_dbfs'].max() - df['peak_dbfs'].min():.2f} dB")
    if use_relative_angle:
        print(f"\nBeamformer locked at: {df['locked_doa'].iloc[0]:.0f}°")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ReSpeaker directivity measurements')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with measurements')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')
    parser.add_argument('--save-location', type=str, default=None, help='Optional directory to save plot (default: same as CSV)')
    
    args = parser.parse_args()
    
    plot_directivity(args.csv_file, save_plot=not args.no_save, save_location=args.save_location)
