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
    
    peaks_available = 'peaks' in df.columns and df['peaks'].any()
    
    if peaks_available:
        print("Peak measurements detected in data. Plotting both RMS and Peak levels.")

        # Create figure with polar plot
        fig = plt.figure(figsize=(12, 10))

        # Main polar plot for RMS level
        ax1 = plt.subplot(221, projection='polar')
        angles_rad = np.deg2rad(df['expected_angle'])
        ax1.plot(angles_rad, df['rms_dbfs'], 'b-o', linewidth=2, markersize=4, label='RMS')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title('RMS Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax1.grid(True)
        ax1.legend(loc='upper right')

        # Polar plot for Peak level
        ax2 = plt.subplot(222, projection='polar')
        ax2.plot(angles_rad, df['peak_dbfs'], 'r-s', linewidth=2, markersize=4, label='Peak')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('Peak Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        # Cartesian plot comparing RMS and Peak
        ax3 = plt.subplot(223)
        ax3.plot(df['expected_angle'], df['rms_dbfs'], 'b-o', linewidth=2, markersize=4, label='RMS')
        ax3.plot(df['expected_angle'], df['peak_dbfs'], 'r-s', linewidth=2, markersize=4, label='Peak')
        ax3.set_xlabel('Angle (degrees)', fontsize=10)
        ax3.set_ylabel('Level (dBFS)', fontsize=10)
        ax3.set_title('Signal Levels vs Angle', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim([0, 360])

        # DOA angle plot
        ax4 = plt.subplot(224)
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
        angles_rad = np.deg2rad(df['expected_angle'])
        ax1.plot(angles_rad, df['rms_dbfs'], 'b-o', linewidth=2, markersize=4)
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_title('RMS Level (dBFS)', pad=20, fontsize=12, fontweight='bold')
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
    print(f"  Min:    {df['rms_dbfs'].min():.2f} dBFS at {df.loc[df['rms_dbfs'].idxmin(), 'expected_angle']:.1f}°")
    print(f"  Max:    {df['rms_dbfs'].max():.2f} dBFS at {df.loc[df['rms_dbfs'].idxmax(), 'expected_angle']:.1f}°")
    print(f"  Range:  {df['rms_dbfs'].max() - df['rms_dbfs'].min():.2f} dB")
    if peaks_available:
        print(f"\nPeak Level:")
        print(f"  Mean:   {df['peak_dbfs'].mean():.2f} dBFS")
        print(f"  Min:    {df['peak_dbfs'].min():.2f} dBFS at {df.loc[df['peak_dbfs'].idxmin(), 'expected_angle']:.1f}°")
        print(f"  Max:    {df['peak_dbfs'].max():.2f} dBFS at {df.loc[df['peak_dbfs'].idxmax(), 'expected_angle']:.1f}°")
        print(f"  Range:  {df['peak_dbfs'].max() - df['peak_dbfs'].min():.2f} dB")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ReSpeaker directivity measurements')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with measurements')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')
    parser.add_argument('--save-location', type=str, default=None, help='Optional directory to save plot (default: same as CSV)')
    
    args = parser.parse_args()
    
    plot_directivity(args.csv_file, save_plot=not args.no_save, save_location=args.save_location)
