# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import h5py
import exoplanet as xo
import os
import corner as corner_module
import astropy.constants as const

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
starry.config.lazy = True
starry.config.quiet = True

# Load the fitted parameters from the saved file
fitted_params = {}
with open('wasp121b_pc_params.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        param, mean, std = line.strip().split(',')
        fitted_params[param] = float(mean)

# Map the parameter names to match the code
params = {
    'fpfs': fitted_params['planet_amp'],
    'per': 1.274924838525504,      # Orbital period in days (fixed)
    't0': 58661.06381351084,       # Time of inferior conjunction (fixed)
    'a': fitted_params['semi_major'],
    'Rs': 1.437300832999673,       # Stellar radius in solar radii (fixed)
    'Rp': fitted_params['planet_radius'],
    'inc': fitted_params['inclination'],
    'c0': fitted_params['c0'],
    'c1': fitted_params['c1'],
    'r0': fitted_params['r0'],
    'r1': fitted_params['r1']
}

# Load and clean the data
with h5py.File('S4_wasp121b_ap3_bg12_LCData.h5', 'r') as f:
    # Load the data
    time = f['time'][:]
    flux = f['data'][:].squeeze()
    flux_err = f['err'][:].squeeze()
    
    # Remove NaNs
    valid_mask = ~np.isnan(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]
    flux_err = flux_err[valid_mask]
    
    # Get the median for normalization
    flux_median = np.median(flux)
    
    # Normalize
    flux = flux / flux_median
    flux_err = flux_err / flux_median
    
    # Convert to float64 for Theano compatibility
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

# Find the closest transit and eclipse times to our data
n_orbits = np.round((time[0] - params['t0']) / params['per'])
transit_time = params['t0'] + n_orbits * params['per']
eclipse_time = transit_time + params['per']/2

print(f"Data start time: {time[0]:.2f}")
print(f"Closest transit time: {transit_time:.2f}")
print(f"Closest eclipse time: {eclipse_time:.2f}")

# Define the PyMC3 model
with pm.Model() as model:
    # Create a star with uniform disk (no limb darkening needed for eclipse)
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=0, amp=1.0),
        m=0,  # Reference frame centered on star
        r=params['Rs'],  # Fixed stellar radius
        prot=1.0
    )

    # Calculate system mass from Kepler's law using original semi-major axis
    a_physical = params['a'] * params['Rs'] * const.R_sun.value  # Convert to meters
    p_physical = params['per'] * 24 * 3600  # Convert to seconds
    system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value
    
    # Create the planet with l=2 map
    planet = starry.Secondary(
        starry.Map(ydeg=2, udeg=0, amp=params['fpfs']),  # l=2 map
        m=system_mass,    # Use system mass directly (since star mass is 0)
        r=params['Rp'],  # Fixed planet radius
        inc=params['inc'],  # Fixed inclination
        a=params['a'],     # Fixed semi-major axis
        t0=params['t0']
    )
    
    # Set additional orbital parameters
    planet.porb = params['per']  # Orbital period
    planet.prot = params['per']  # Rotation period (synchronous)
    planet.theta0 = 180.0        # Initial phase angle

    # Fit for the l=2 spherical harmonics coefficients
    # Use multivariate normal prior centered at zero
    y10 = pm.Normal("y10", mu=0.0, sd=0.1, testval=0.0)
    y11c = pm.Normal("y11c", mu=0.0, sd=0.1, testval=0.0)
    y11s = pm.Normal("y11s", mu=0.0, sd=0.1, testval=0.0)
    y20 = pm.Normal("y20", mu=0.0, sd=0.1, testval=0.0)
    y21c = pm.Normal("y21c", mu=0.0, sd=0.1, testval=0.0)
    y21s = pm.Normal("y21s", mu=0.0, sd=0.1, testval=0.0)
    y22c = pm.Normal("y22c", mu=0.0, sd=0.1, testval=0.0)
    y22s = pm.Normal("y22s", mu=0.0, sd=0.1, testval=0.0)
    
    # Set the coefficients in the planet map
    planet.map[1, 0] = y10
    planet.map[1, 1] = y11c
    planet.map[1, -1] = y11s
    planet.map[2, 0] = y20
    planet.map[2, 1] = y21c
    planet.map[2, -1] = y21s
    planet.map[2, 2] = y22c
    planet.map[2, -2] = y22s

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Total system flux is star + planet
    astro_flux = star_flux + planet_flux
    
    # Use fixed systematic parameters from previous fit
    t_norm = (time - time[0]) * 86400  # Convert days to seconds
    poly_baseline = params['c0'] + params['c1'] * t_norm / 86400
    ramp = 1 + params['r0'] * pm.math.exp(-t_norm / (params['r1'] * 86400))
    
    # Add error inflation parameter
    error_inflation = pm.HalfNormal("error_inflation", sd=0.5, testval=1.0)
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
    
    # Save the map as a deterministic for plotting
    map_vals = pm.Deterministic("map", planet.map.render(projection="rect"))
    
    # Define the likelihood with error inflation
    pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)

# Run the MCMC sampling
with model:
    trace = pmx.sample(
        tune=200,
        draws=200,
        target_accept=0.9,
        return_inferencedata=False,
        cores=2,
        chains=2
    )

    # Create corner plot
    samples = np.vstack([
        trace['y10'],
        trace['y11c'],
        trace['y11s'],
        trace['y20'],
        trace['y21c'],
        trace['y21s'],
        trace['y22c'],
        trace['y22s']
    ]).T
    labels = ['Y10', 'Y11c', 'Y11s', 'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s']
    fig = corner_module.corner(samples, labels=labels, 
                       show_titles=True,
                       title_fmt='.3f',
                       use_math_text=True,
                       quiet=True)
    plt.savefig('corner_plot_map.png')
    plt.close()

# Print parameter comparison
print("\nParameter Comparison:")
print("Parameter     Prior Mean ± SD      Fitted Mean ± SD")
print("-" * 55)
print(f"Y10          0.000000 ± 0.100000  {np.mean(trace['y10']):.6f} ± {np.std(trace['y10']):.6f}")
print(f"Y11c         0.000000 ± 0.100000  {np.mean(trace['y11c']):.6f} ± {np.std(trace['y11c']):.6f}")
print(f"Y11s         0.000000 ± 0.100000  {np.mean(trace['y11s']):.6f} ± {np.std(trace['y11s']):.6f}")
print(f"Y20          0.000000 ± 0.100000  {np.mean(trace['y20']):.6f} ± {np.std(trace['y20']):.6f}")
print(f"Y21c         0.000000 ± 0.100000  {np.mean(trace['y21c']):.6f} ± {np.std(trace['y21c']):.6f}")
print(f"Y21s         0.000000 ± 0.100000  {np.mean(trace['y21s']):.6f} ± {np.std(trace['y21s']):.6f}")
print(f"Y22c         0.000000 ± 0.100000  {np.mean(trace['y22c']):.6f} ± {np.std(trace['y22c']):.6f}")
print(f"Y22s         0.000000 ± 0.100000  {np.mean(trace['y22s']):.6f} ± {np.std(trace['y22s']):.6f}")

# Plot the light curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot systematic components using fixed parameters
t_norm = time - time[0]  # Time in days
t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
baseline = params['c0'] + params['c1'] * t_norm
ramp = 1 + params['r0'] * np.exp(-t_norm_seconds / (params['r1'] * 86400))
systematic = baseline * ramp

plt.plot(t_norm, systematic, 'g--', alpha=0.7, label='Systematic trend')

# Plot the mean model fit
mean_model = np.mean(trace["flux"], axis=0)
plt.plot(t_norm, mean_model, 'C0-', linewidth=2, label='Mean model fit')

# Plot a few individual samples with very low alpha to show uncertainty
for i in np.random.choice(range(len(trace["flux"])), 10):
    plt.plot(t_norm, trace["flux"][i], 'C0-', alpha=0.1)

plt.legend(fontsize=10)
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-121b Eclipse Light Curve with l=2 Map")

plt.savefig('wasp121b_fit_map.png')
plt.close()

# Plot the maps
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Mean map from posterior
planet_mu = np.mean(trace["map"], axis=0)
im0 = ax[0].imshow(
    planet_mu,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)
plt.colorbar(im0, ax=ax[0])
ax[0].set_title("Mean Map")

# Standard deviation map
planet_std = np.std(trace["map"], axis=0)
im1 = ax[1].imshow(
    planet_std,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)
plt.colorbar(im1, ax=ax[1])
ax[1].set_title("Uncertainty Map")

plt.tight_layout()
plt.savefig('wasp121b_maps.png')
plt.close()

# Plot map posterior along equator and substellar meridian
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Get the map dimensions
n_lat, n_lon = trace["map"].shape[1:]

# Create longitude and latitude arrays matching map dimensions
lon = np.linspace(-180, 180, n_lon)
lat = np.linspace(-90, 90, n_lat)

# Get the map values along the equator (lat=0)
equator_idx = n_lat // 2  # Index corresponding to lat=0
equator_maps = trace["map"][:, equator_idx, :]
equator_mean = np.mean(equator_maps, axis=0)
equator_std = np.std(equator_maps, axis=0)

# Get the map values along the substellar meridian (lon=0)
substellar_idx = n_lon // 2  # Index corresponding to lon=0
substellar_maps = trace["map"][:, :, substellar_idx]
substellar_mean = np.mean(substellar_maps, axis=0)
substellar_std = np.std(substellar_maps, axis=0)

# Plot equator profile
ax1.fill_between(lon, 
                 equator_mean - equator_std,
                 equator_mean + equator_std,
                 alpha=0.3, color='C0', label='1σ uncertainty')
ax1.plot(lon, equator_mean, 'C0-', linewidth=2, label='Mean')
ax1.set_xlabel("Longitude [degrees]", fontsize=12)
ax1.set_ylabel("Temperature", fontsize=12)
ax1.set_title("Map Profile Along Equator")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot substellar meridian profile
ax2.fill_between(lat,
                 substellar_mean - substellar_std,
                 substellar_mean + substellar_std,
                 alpha=0.3, color='C0', label='1σ uncertainty')
ax2.plot(lat, substellar_mean, 'C0-', linewidth=2, label='Mean')
ax2.set_xlabel("Latitude [degrees]", fontsize=12)
ax2.set_ylabel("Temperature", fontsize=12)
ax2.set_title("Map Profile Through Substellar Point")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wasp121b_map_profiles.png')
plt.close()

# Calculate chi-squared
mean_model = np.mean(trace["flux"], axis=0)
chi2 = np.sum((flux - mean_model)**2 / flux_err**2)
print("\nReduced chi-squared: {:.2f}".format(chi2/len(flux)))

# Load the phase curve fit and parameters
pc_time, pc_flux = np.loadtxt('wasp121b_pc_fit.txt', unpack=True)

# Interpolate the phase curve model to our data times
pc_flux_interp = np.interp(time, pc_time, pc_flux)

# Calculate deviations from phase curve model
data_deviation = flux - pc_flux_interp
data_deviation_err = flux_err

# Bin the data deviations every 10 points
bin_size = 10
n_bins = len(time) // bin_size
binned_time = np.zeros(n_bins)
binned_data_dev = np.zeros(n_bins)
binned_data_dev_err = np.zeros(n_bins)

for i in range(n_bins):
    start_idx = i * bin_size
    end_idx = (i + 1) * bin_size
    binned_time[i] = np.mean(time[start_idx:end_idx])
    binned_data_dev[i] = np.mean(data_deviation[start_idx:end_idx])
    binned_data_dev_err[i] = np.sqrt(np.sum(data_deviation_err[start_idx:end_idx]**2)) / bin_size

# Calculate map model deviations from phase curve model
map_deviation = trace["flux"] - pc_flux_interp
map_deviation_mean = np.mean(map_deviation, axis=0)
map_deviation_std = np.std(map_deviation, axis=0)

# Calculate exact eclipse times
# Convert angles to radians for calculations
inc_rad = np.radians(params['inc'])
a = params['a']  # semi-major axis in stellar radii
R_star = params['Rs']  # stellar radius in solar radii
R_planet = params['Rp']  # planet radius in solar radii

# Calculate the impact parameter
b = a * np.cos(inc_rad)

# Calculate the total and partial eclipse durations
t_total = (params['per']/np.pi) * np.arcsin((R_star + R_planet)/a)
t_partial = (params['per']/np.pi) * np.arcsin((R_star - R_planet)/a)

# Calculate the exact ingress and egress times relative to eclipse center
ingress_start = -t_total/2  # Time before eclipse center
ingress_end = -t_partial/2  # Time before eclipse center
egress_start = t_partial/2  # Time after eclipse center
egress_end = t_total/2      # Time after eclipse center

print(f"\nEclipse timing calculations:")
print(f"Total eclipse duration: {t_total*24*60:.2f} minutes")
print(f"Partial phase duration: {t_partial*24*60:.2f} minutes")
print(f"Ingress: {ingress_start*24*60:.2f} to {ingress_end*24*60:.2f} minutes before eclipse")
print(f"Egress: {egress_start*24*60:.2f} to {egress_end*24*60:.2f} minutes after eclipse")

# Plot the deviations
plt.figure(figsize=(12, 5))

# Plot binned data points
plt.errorbar((binned_time - eclipse_time)*24*60, binned_data_dev, 
             yerr=binned_data_dev_err, fmt='k.', alpha=0.3, ms=2, 
             label='Data - PC model (binned)')

# Plot map model
plt.fill_between((time - eclipse_time)*24*60, 
                 map_deviation_mean - map_deviation_std,
                 map_deviation_mean + map_deviation_std,
                 alpha=0.3, color='C0', label='Map model - PC model (1σ)')
plt.plot((time - eclipse_time)*24*60, map_deviation_mean, 'C0-', 
         linewidth=2, label='Map model - PC model (mean)')

# Add eclipse time
plt.axvline(x=0, color='b', linestyle='--', label='Eclipse')

plt.legend(fontsize=10)
plt.xlabel("Time from Eclipse [minutes]", fontsize=12)
plt.ylabel("Deviation from PC Model", fontsize=12)
plt.title("WASP-121b Eclipse Deviations from Phase Curve Model")

plt.tight_layout()
plt.savefig('wasp121b_deviations.png')
plt.close()

# Calculate systematic components
t_norm = time - time[0]  # Time in days
t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
baseline = params['c0'] + params['c1'] * t_norm
ramp = 1 + params['r0'] * np.exp(-t_norm_seconds / (params['r1'] * 86400))
systematic = baseline * ramp

# Remove systematics from data and models
flux_no_sys = flux / systematic
flux_err_no_sys = flux_err / systematic
mean_model_no_sys = np.mean(trace["flux"], axis=0) / systematic
pc_flux_no_sys = pc_flux_interp / systematic

# Bin the data without systematics
binned_time = np.zeros(n_bins)
binned_flux = np.zeros(n_bins)
binned_flux_err = np.zeros(n_bins)

for i in range(n_bins):
    start_idx = i * bin_size
    end_idx = (i + 1) * bin_size
    binned_time[i] = np.mean(time[start_idx:end_idx])
    binned_flux[i] = np.mean(flux_no_sys[start_idx:end_idx])
    binned_flux_err[i] = np.sqrt(np.sum(flux_err_no_sys[start_idx:end_idx]**2)) / bin_size

# Plot the light curve without systematics
plt.figure(figsize=(12, 5))
plt.errorbar((binned_time - eclipse_time)*24*60, binned_flux, 
             yerr=binned_flux_err, fmt='k.', alpha=0.3, ms=2, 
             label='Data (binned)')

# Plot the phase curve model without systematics
plt.plot((time - eclipse_time)*24*60, pc_flux_no_sys, 'g-', 
         linewidth=2, label='Phase curve model')

# Plot the map model without systematics
plt.fill_between((time - eclipse_time)*24*60, 
                 mean_model_no_sys - np.std(trace["flux"]/systematic, axis=0),
                 mean_model_no_sys + np.std(trace["flux"]/systematic, axis=0),
                 alpha=0.3, color='C0', label='Map model (1σ)')
plt.plot((time - eclipse_time)*24*60, mean_model_no_sys, 'C0-', 
         linewidth=2, label='Map model (mean)')

plt.axvline(x=0, color='b', linestyle='--', label='Eclipse')
plt.legend(fontsize=10)
plt.xlabel("Time from Eclipse [minutes]", fontsize=12)
plt.ylabel("Relative Flux (Systematics Removed)", fontsize=12)
plt.title("WASP-121b Eclipse Light Curve (Systematics Removed)")

plt.tight_layout()
plt.savefig('wasp121b_fit_no_sys.png')
plt.close()

# After sampling, get the error inflation value
error_inflation_mean = np.mean(trace["error_inflation"])
print(f"\nError inflation factor: {error_inflation_mean:.3f}")

# Update the error bars in the plots
flux_err_scaled = flux_err * error_inflation_mean
data_deviation_err_scaled = data_deviation_err * error_inflation_mean
flux_err_no_sys_scaled = flux_err_no_sys * error_inflation_mean

# Update the deviation plot
plt.figure(figsize=(12, 5))
plt.errorbar((binned_time - eclipse_time)*24*60, binned_data_dev, 
             yerr=binned_data_dev_err * error_inflation_mean, fmt='k.', alpha=0.3, ms=2, 
             label='Data - PC model (binned)')

# Plot map model
plt.fill_between((time - eclipse_time)*24*60, 
                 map_deviation_mean - map_deviation_std,
                 map_deviation_mean + map_deviation_std,
                 alpha=0.3, color='C0', label='Map model - PC model (1σ)')
plt.plot((time - eclipse_time)*24*60, map_deviation_mean, 'C0-', 
         linewidth=2, label='Map model - PC model (mean)')

# Add eclipse time
plt.axvline(x=0, color='b', linestyle='--', label='Eclipse')

plt.legend(fontsize=10)
plt.xlabel("Time from Eclipse [minutes]", fontsize=12)
plt.ylabel("Deviation from PC Model", fontsize=12)
plt.title("WASP-121b Eclipse Deviations from Phase Curve Model")

plt.tight_layout()
plt.savefig('wasp121b_deviations_scaled.png')
plt.close()

# Update the no-systematics plot
plt.figure(figsize=(12, 5))
plt.errorbar((binned_time - eclipse_time)*24*60, binned_flux, 
             yerr=binned_flux_err * error_inflation_mean, fmt='k.', alpha=0.3, ms=2, 
             label='Data (binned)')

# Plot the phase curve model without systematics
plt.plot((time - eclipse_time)*24*60, pc_flux_no_sys, 'g-', 
         linewidth=2, label='Phase curve model')

# Plot the map model without systematics
plt.fill_between((time - eclipse_time)*24*60, 
                 mean_model_no_sys - np.std(trace["flux"]/systematic, axis=0),
                 mean_model_no_sys + np.std(trace["flux"]/systematic, axis=0),
                 alpha=0.3, color='C0', label='Map model (1σ)')
plt.plot((time - eclipse_time)*24*60, mean_model_no_sys, 'C0-', 
         linewidth=2, label='Map model (mean)')

plt.axvline(x=0, color='b', linestyle='--', label='Eclipse')
plt.legend(fontsize=10)
plt.xlabel("Time from Eclipse [minutes]", fontsize=12)
plt.ylabel("Relative Flux (Systematics Removed)", fontsize=12)
plt.title("WASP-121b Eclipse Light Curve (Systematics Removed)")

plt.tight_layout()
plt.savefig('wasp121b_fit_no_sys_scaled.png')
plt.close() 