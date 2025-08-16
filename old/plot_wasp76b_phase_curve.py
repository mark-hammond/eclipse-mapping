import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_phase_curve(detector='nrs1'):
    """
    Plot the phase curve for either NRS1 or NRS2 detector.
    
    Parameters
    ----------
    detector : str
        Either 'nrs1' or 'nrs2' to select which detector's data to plot
    """
    # Read the data
    data = np.genfromtxt('WASP-76b_WhiteLight.csv', delimiter=',', skip_header=1)
    
    # Extract the data based on detector choice
    if detector.lower() == 'nrs1':
        time = data[:, 0]
        flux = data[:, 1]
        flux_err = data[:, 2]
        sys_corr = data[:, 3]
        label = 'NRS1'
    elif detector.lower() == 'nrs2':
        time = data[:, 0]
        flux = data[:, 4]
        flux_err = data[:, 5]
        sys_corr = data[:, 6]
        label = 'NRS2'
    else:
        raise ValueError("Detector must be either 'nrs1' or 'nrs2'")
    
    # Convert time to hours from start
    time = (time - time[0]) * 24
    
    # Create a figure with three subplots
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # First subplot: Light curve
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(time, flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='Raw Data')
    ax1.plot(time, sys_corr, 'C0-', linewidth=2, label='Systematics Corrected')
    ax1.set_ylabel("Relative Flux", fontsize=12)
    ax1.set_title(f"WASP-76b Phase Curve ({label})", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Systematics correction
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, sys_corr - flux, 'C1-', linewidth=2, label='Correction')
    ax2.set_ylabel("Systematics Correction", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Third subplot: Histogram of raw flux
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(flux, bins=50, density=True, color='C0', alpha=0.7, label='Raw Data')
    ax3.hist(sys_corr, bins=50, density=True, color='C1', alpha=0.7, label='Corrected')
    ax3.set_xlabel("Relative Flux", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add statistics to the histogram
    raw_mean = np.mean(flux)
    raw_std = np.std(flux)
    corr_mean = np.mean(sys_corr)
    corr_std = np.std(sys_corr)
    
    stats_text = (f"Raw Data:\nMean: {raw_mean:.6f}\nStd: {raw_std:.6f}\n\n"
                 f"Corrected:\nMean: {corr_mean:.6f}\nStd: {corr_std:.6f}")
    
    ax3.text(0.05, 0.95, stats_text,
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'wasp76b_phase_curve_{detector}.png')
    plt.close()
    
    # Print statistics
    print(f"\nStatistics for {label}:")
    print(f"Raw Data - Mean: {raw_mean:.6f}, Std: {raw_std:.6f}")
    print(f"Corrected - Mean: {corr_mean:.6f}, Std: {corr_std:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot WASP-76b phase curve data')
    parser.add_argument('--detector', type=str, default='nrs1',
                      choices=['nrs1', 'nrs2'],
                      help='Detector to plot (nrs1 or nrs2)')
    args = parser.parse_args()
    
    plot_phase_curve(args.detector) 