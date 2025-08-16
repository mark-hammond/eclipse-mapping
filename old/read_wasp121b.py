import h5py
import numpy as np
import matplotlib.pyplot as plt

# Read the HDF5 file
with h5py.File('S4_wasp121b_ap3_bg12_LCData.h5', 'r') as f:
    # Load the data
    time = f['time'][:]
    flux = f['data'][:].squeeze()
    flux_err = f['err'][:].squeeze()
    
    # Print basic info
    print("\nData Info:")
    print(f"Time shape: {time.shape}")
    print(f"Flux shape: {flux.shape}")
    print(f"Error shape: {flux_err.shape}")
    
    # Print first few values
    print("\nFirst 5 time points:", time[:5])
    print("First 5 flux values:", flux[:5])
    print("First 5 error values:", flux_err[:5])
    
    # Remove NaNs
    valid_mask = ~np.isnan(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]
    flux_err = flux_err[valid_mask]
    
    # Print info after removing NaNs
    print("\nAfter removing NaNs:")
    print(f"Time shape: {time.shape}")
    print(f"Flux shape: {flux.shape}")
    print(f"Error shape: {flux_err.shape}")
    
    # Get the median for normalization
    flux_median = np.median(flux)
    print(f"\nMedian flux for normalization: {flux_median:.6f}")
    
    # Normalize
    flux = flux / flux_median
    flux_err = flux_err / flux_median  # Use the same median for errors
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Time range: {time.min():.6f} to {time.max():.6f}")
    print(f"Time span: {time.max() - time.min():.6f} days")
    print(f"Number of points: {len(time)}")
    print(f"Median flux: {np.median(flux):.6f}")
    print(f"Mean flux: {np.mean(flux):.6f}")
    print(f"Std flux: {np.std(flux):.6f}")
    print(f"Median error: {np.median(flux_err):.6f}")
    print(f"Relative error (error/flux): {np.median(flux_err)/np.median(flux):.6f}")

# Create a figure
plt.figure(figsize=(12, 5))

# Plot the data
plt.errorbar(time, flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.xlabel("Time [BJD]")
plt.ylabel("Relative Flux")
plt.title("WASP-121b Light Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('wasp121b_lightcurve.png')
plt.close() 