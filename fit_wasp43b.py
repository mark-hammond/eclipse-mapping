# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import corner as corner_module
import astropy.constants as const

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
starry.config.lazy = True
starry.config.quiet = True

# Key model parameters
MAP_DEGREE = 2        # Degree of spherical harmonic map
FOURIER_DEGREE = 2    # Degree of Fourier series for phase curve
FIT_ORBITAL = False   # Whether to fit orbital parameters
N_SAMPLES = 200       # Number of MCMC samples
N_TUNE = 200         # Number of tuning steps
TARGET_ACCEPT = 0.9   # Target acceptance rate for MCMC
N_CHAINS = 2          # Number of MCMC chains
N_CORES = 2           # Number of CPU cores to use

# Data selection parameters
DATA_START = None     # Start index for data slice (None for beginning)
DATA_END = None      # End index for data slice (None for all data)
BIN_SIZE = 100      # Number of points to bin (None for no binning)
DEVIATION_BIN_SIZE = 1  # Number of points to bin for deviation plots

# Systematic parameters
USE_POLYNOMIAL = True   # Whether to fit polynomial baseline
USE_RAMP = True       # Whether to fit exponential ramp

# Load WASP-43b data from numpy files
print("Loading WASP-43b data...")
time = np.load('w43b_time.npy')
flux = np.load('w43b_flux.npy')
flux_err = np.load('w43b_error.npy')

print(f"Loaded data: {len(time)} points")
print(f"Time range: {time[0]:.6f} to {time[-1]:.6f}")
print(f"Flux range: {np.min(flux):.6f} to {np.max(flux):.6f}")

# Define parameters using "Eclipse Map Fit" values from the table
params = {
    'fpfs': 0.001,        # Planet-to-star flux ratio (placeholder - will be fitted)
    'per': 0.8134740621723353,  # Orbital period in days (from literature)
    't0': 55934.292283,   # Transit time (BMJD) from Eclipse Map Fit
    'a': 4.859,           # Semi-major axis in stellar radii from Eclipse Map Fit
    'Rs': 0.665,          # Stellar radius in solar radii (from literature)
    'Rp': 0.15839,        # Planet radius in stellar radii from Eclipse Map Fit
    'inc': 82.106,        # Orbital inclination in degrees from Eclipse Map Fit
    'c0': -2881.0/1e6,    # Constant baseline (ppm) from Eclipse Map Fit, converted to relative flux
    'c1': -240.0/1e6,     # Linear trend (ppm/day) from Eclipse Map Fit, converted to relative flux
    'r0': 1319.0/1e6,     # Ramp magnitude (ppm) from Eclipse Map Fit, converted to relative flux
    'r1': 3.7,            # Ramp time constant (1/day) from Eclipse Map Fit
    'u1': 0.0182,         # Limb darkening parameter q1 from Eclipse Map Fit
    'u2': 0.595           # Limb darkening parameter q2 from Eclipse Map Fit
}

print(f"\nSystem parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Select data range
if DATA_START is None:
    DATA_START = 0
if DATA_END is None:
    DATA_END = len(time)
time = time[DATA_START:DATA_END]
flux = flux[DATA_START:DATA_END]
flux_err = flux_err[DATA_START:DATA_END]

# Bin the data if requested
if BIN_SIZE is not None and BIN_SIZE > 1:
    n_bins = len(time) // BIN_SIZE
    binned_time = np.zeros(n_bins)
    binned_flux = np.zeros(n_bins)
    binned_flux_err = np.zeros(n_bins)
    
    for i in range(n_bins):
        start_idx = i * BIN_SIZE
        end_idx = (i + 1) * BIN_SIZE
        binned_time[i] = np.mean(time[start_idx:end_idx])
        binned_flux[i] = np.mean(flux[start_idx:end_idx])
        # Error propagation for binning
        binned_flux_err[i] = np.sqrt(np.sum(flux_err[start_idx:end_idx]**2)) / BIN_SIZE
    
    time = binned_time
    flux = binned_flux
    flux_err = binned_flux_err

print(f"\nData selection summary:")
print(f"Using data points {DATA_START} to {DATA_END}")
print(f"Number of points after selection: {len(time)}")
if BIN_SIZE is not None:
    print(f"Binned every {BIN_SIZE} points")
    print(f"Final number of points: {len(time)}")

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

# Plot the raw data
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='WASP-43b data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-43b Light Curve Data")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('wasp43b_raw_data.png')
plt.close()

print('creating model')
# Define the PyMC3 model
with pm.Model() as model:
    # Create a star with quadratic limb darkening
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=2, amp=1.0),  # udeg=2 for quadratic limb darkening
        m=0,  # Reference frame centered on star
        r=params['Rs'],  # Fixed stellar radius
        prot=1.0
    )
    
    # Set the limb darkening coefficients
    if FIT_ORBITAL:
        u1 = pm.Normal("u1", mu=params['u1'], sd=0.1, testval=params['u1'])
        u2 = pm.Normal("u2", mu=params['u2'], sd=0.1, testval=params['u2'])
        star.map[1] = u1  # Linear term
        star.map[2] = u2  # Quadratic term
    else:
        star.map[1] = params['u1']  # Linear term
        star.map[2] = params['u2']  # Quadratic term

    # Fit for the planet amplitude
    planet_amp = pm.Normal("planet_amp", mu=params['fpfs'], sd=0.001, testval=params['fpfs'])
    
    # Fit for t0 only if FIT_ORBITAL is true
    if FIT_ORBITAL:
        t0 = pm.Normal("t0", mu=params['t0'], sd=0.1, testval=params['t0'])
    else:
        t0 = params['t0']
    
    # Handle orbital parameters based on FIT_ORBITAL flag
    if FIT_ORBITAL:
        # Use bounded normal for planet radius to ensure it stays positive
        planet_radius = pm.Bound(pm.Normal, lower=0.0)("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
        semi_major = pm.Normal("semi_major", mu=params['a'], sd=0.1, testval=params['a'])
        inclination = pm.Normal("inclination", mu=params['inc'], sd=0.1, testval=params['inc'])
    else:
        # Fix orbital parameters
        planet_radius = params['Rp']
        semi_major = params['a']
        inclination = params['inc']
    
    # Calculate system mass from Kepler's law using original semi-major axis
    a_physical = semi_major * params['Rs'] * const.R_sun.value  # Convert to meters
    p_physical = params['per'] * 24 * 3600  # Convert to seconds
    system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value
    
    # Create the planet with l=2 map
    planet = starry.Secondary(
        starry.Map(ydeg=MAP_DEGREE, udeg=0, amp=planet_amp),  # l=2 map, no limb darkening
        m=system_mass,    # Use system mass directly (since star mass is 0)
        r=planet_radius,  # Fixed planet radius
        inc=inclination,  # Fitted inclination
        a=semi_major,     # Use original semi-major axis
        t0=t0            # Fitted t0
    )
    
    # Set additional orbital parameters
    planet.porb = params['per']  # Orbital period
    planet.prot = params['per']  # Rotation period (synchronous)
    planet.theta0 = 180.0        # Initial phase angle

    # The Ylm coefficients of the planet
    # with a zero-mean isotropic Gaussian prior
    ncoeff = planet.map.Ny - 1  # Exclude y00
    planet_mu = np.zeros(ncoeff)
    planet_cov = 0.5**2 * np.eye(ncoeff)  # 0.5 standard deviation
    planet.map[1:, :] = pm.MvNormal("planet_y", planet_mu, planet_cov, shape=(ncoeff,))

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Total system flux is star + planet
    astro_flux = star_flux + planet_flux
    
    # Initialize systematic effects to 1.0 (no effect)
    poly_baseline = 1.0
    ramp = 1.0
    
    # Add systematic effects if enabled
    t_norm = (time - time[0]) * 86400  # Convert days to seconds
    
    if USE_POLYNOMIAL:
        # Polynomial baseline parameters
        c0 = pm.Normal("c0", mu=params['c0'], sd=0.001, testval=params['c0'])
        c1 = pm.Normal("c1", mu=params['c1'], sd=0.001, testval=params['c1'])
        poly_baseline = c0 + c1 * t_norm / 86400
    
    if USE_RAMP:
        # Ramp parameters
        r0 = pm.Normal("r0", mu=params['r0'], sd=0.001, testval=params['r0'])
        r1 = pm.Normal("r1", mu=params['r1'], sd=0.1, testval=params['r1'])
        ramp = 1 + r0 * pm.math.exp(-t_norm / (r1 * 86400))
    
    # Add error inflation parameter
    error_inflation = pm.HalfNormal("error_inflation", sd=1.0, testval=1.0)
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
    
    # Save the map as a deterministic for plotting
    map_vals = pm.Deterministic("map", planet.map.render(projection="rect"))
    
    # Define the likelihood with error inflation
    pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)

print('running mcmc')
# Run the MCMC sampling
with model:
    trace = pmx.sample(
        tune=N_TUNE,
        draws=N_SAMPLES,
        target_accept=TARGET_ACCEPT,
        return_inferencedata=False,
        cores=N_CORES,
        chains=N_CHAINS
    )

# Create corner plot
if FIT_ORBITAL:
    samples_list = [trace['planet_amp'], trace['planet_radius'], trace['semi_major'], trace['inclination']]
    labels_list = ['planet_amp', 'Rp/R*', 'a/R*', 'inc']
else:
    samples_list = [trace['planet_amp']]
    labels_list = ['planet_amp']

# Add Y coefficients
samples_list.extend([trace['planet_y'][:, i] for i in range(trace['planet_y'].shape[1])])
labels_list.extend([f'Y{i}' for i in range(trace['planet_y'].shape[1])])

# Add systematic parameters if they were used
if USE_POLYNOMIAL:
    samples_list.extend([trace['c0'], trace['c1']])
    labels_list.extend(['c0', 'c1'])
if USE_RAMP:
    samples_list.extend([trace['r0'], trace['r1']])
    labels_list.extend(['r0', 'r1'])

# Add error inflation
samples_list.append(trace['error_inflation'])
labels_list.append('error_inflation')

# Stack samples
samples = np.vstack(samples_list).T

# Create corner plot
fig = corner_module.corner(samples, labels=labels_list, 
                   show_titles=True,
                   title_fmt='.3f',
                   use_math_text=True,
                   quiet=True)
plt.savefig('corner_plot_wasp43b_map.png')
plt.close()

# Print parameter comparison
print("\nParameter Comparison:")
print("Parameter     Prior Mean ± SD      Fitted Mean ± SD")
print("-" * 55)
print(f"planet_amp   {params['fpfs']:.6f} ± 0.001000  {np.mean(trace['planet_amp']):.6f} ± {np.std(trace['planet_amp']):.6f}")
if FIT_ORBITAL:
    print(f"t0          {params['t0']:.6f} ± 0.100000  {np.mean(trace['t0']):.6f} ± {np.std(trace['t0']):.6f}")
    print(f"planet_radius {params['Rp']:.6f} ± 0.020000  {np.mean(trace['planet_radius']):.6f} ± {np.std(trace['planet_radius']):.6f}")
    print(f"semi_major   {params['a']:.6f} ± 0.100000  {np.mean(trace['semi_major']):.6f} ± {np.std(trace['semi_major']):.6f}")
    print(f"inclination  {params['inc']:.6f} ± 0.100000  {np.mean(trace['inclination']):.6f} ± {np.std(trace['inclination']):.6f}")
    print(f"u1          {params['u1']:.6f} ± 0.100000  {np.mean(trace['u1']):.6f} ± {np.std(trace['u1']):.6f}")
    print(f"u2          {params['u2']:.6f} ± 0.100000  {np.mean(trace['u2']):.6f} ± {np.std(trace['u2']):.6f}")
else:
    print(f"t0          {params['t0']:.6f} (fixed)")
    print(f"planet_radius {params['Rp']:.6f} (fixed)")
    print(f"semi_major   {params['a']:.6f} (fixed)")
    print(f"inclination  {params['inc']:.6f} (fixed)")
    print(f"u1          {params['u1']:.6f} (fixed)")
    print(f"u2          {params['u2']:.6f} (fixed)")

# Print map coefficients
y_coeffs_mean = np.mean(trace['planet_y'], axis=0)
y_coeffs_std = np.std(trace['planet_y'], axis=0)
n_coeffs = len(y_coeffs_mean)
for i in range(n_coeffs):
    print(f"Y{i}          0.000000 ± 0.500000  {y_coeffs_mean[i]:.6f} ± {y_coeffs_std[i]:.6f}")

# Print systematic parameters if they were used
if USE_POLYNOMIAL:
    print(f"c0           {params['c0']:.6f} ± 0.001000  {np.mean(trace['c0']):.6f} ± {np.std(trace['c0']):.6f}")
    print(f"c1           {params['c1']:.6f} ± 0.001000  {np.mean(trace['c1']):.6f} ± {np.std(trace['c1']):.6f}")
if USE_RAMP:
    print(f"r0           {params['r0']:.6f} ± 0.001000  {np.mean(trace['r0']):.6f} ± {np.std(trace['r0']):.6f}")
    print(f"r1           {params['r1']:.6f} ± 0.100000  {np.mean(trace['r1']):.6f} ± {np.std(trace['r1']):.6f}")

print(f"error_inflation 1.000000 ± 0.500000  {np.mean(trace['error_inflation']):.6f} ± {np.std(trace['error_inflation']):.6f}")

# Print all orbital parameters being fitted
print("\nOrbital Parameters Being Fitted:")
print("Parameter     Value")
print("-" * 30)
print(f"period       {params['per']:.6f} days")
print(f"t0          {params['t0']:.6f} BJD")
print(f"a/R*        {params['a']:.6f}")
print(f"Rp/R*       {params['Rp']:.6f}")
print(f"inc         {params['inc']:.6f} deg")
print(f"u1          {params['u1']:.6f}")
print(f"u2          {params['u2']:.6f}")

# Calculate chi-squared and get error inflation
mean_model = np.mean(trace["flux"], axis=0)
error_inflation = np.mean(trace['error_inflation'])
print(f"\nError inflation factor: {error_inflation:.6f}")

chi2 = np.sum((flux - mean_model)**2 / (flux_err * error_inflation)**2)
print("\nReduced chi-squared: {:.2f}".format(chi2/len(flux)))

# Plot the light curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err * error_inflation, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot systematic components using fitted parameters
t_norm = time - time[0]  # Time in days
t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation

# Initialize systematic trend
systematic = np.ones_like(t_norm)

# Add polynomial baseline if it was fitted
if USE_POLYNOMIAL:
    baseline = np.mean(trace["c0"]) + np.mean(trace["c1"]) * t_norm
    systematic *= baseline
    plt.plot(t_norm, baseline, 'g--', alpha=0.7, label='Polynomial trend')

# Add ramp if it was fitted
if USE_RAMP:
    r0_fit = np.mean(trace["r0"])
    r1_fit = np.mean(trace["r1"])
    ramp = 1 + r0_fit * np.exp(-t_norm_seconds / (r1_fit * 86400))
    systematic *= ramp
    plt.plot(t_norm, ramp, 'g--', alpha=0.7, label='Ramp trend')

# Plot the mean model fit
mean_model = np.mean(trace["flux"], axis=0)
plt.plot(t_norm, mean_model, 'C0-', linewidth=2, label='Mean model fit')

# Plot a few individual samples with very low alpha to show uncertainty
for i in np.random.choice(range(len(trace["flux"])), 10):
    plt.plot(t_norm, trace["flux"][i], 'C0-', alpha=0.1)

plt.legend(fontsize=10)
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-43b Eclipse Light Curve with l=2 Map")

plt.savefig('wasp43b_fit_map.png')
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
plt.savefig('wasp43b_maps.png')
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
plt.savefig('wasp43b_map_profiles.png')
plt.close()

# Save the fitted parameters
with open('wasp43b_map_params.txt', 'w') as f:
    f.write("Parameter,Mean,Std\n")
    f.write(f"planet_amp,{np.mean(trace['planet_amp']):.6f},{np.std(trace['planet_amp']):.6f}\n")
    if FIT_ORBITAL:
        f.write(f"t0,{np.mean(trace['t0']):.6f},{np.std(trace['t0']):.6f}\n")
        f.write(f"planet_radius,{np.mean(trace['planet_radius']):.6f},{np.std(trace['planet_radius']):.6f}\n")
        f.write(f"semi_major,{np.mean(trace['semi_major']):.6f},{np.std(trace['semi_major']):.6f}\n")
        f.write(f"inclination,{np.mean(trace['inclination']):.6f},{np.std(trace['inclination']):.6f}\n")
        f.write(f"u1,{np.mean(trace['u1']):.6f},{np.std(trace['u1']):.6f}\n")
        f.write(f"u2,{np.mean(trace['u2']):.6f},{np.std(trace['u2']):.6f}\n")
    else:
        f.write(f"t0,{params['t0']:.6f},0.000000\n")
        f.write(f"planet_radius,{params['Rp']:.6f},0.000000\n")
        f.write(f"semi_major,{params['a']:.6f},0.000000\n")
        f.write(f"inclination,{params['inc']:.6f},0.000000\n")
        f.write(f"u1,{params['u1']:.6f},0.000000\n")
        f.write(f"u2,{params['u2']:.6f},0.000000\n")
    for i in range(n_coeffs):
        f.write(f"Y{i},{y_coeffs_mean[i]:.6f},{y_coeffs_std[i]:.6f}\n")
    if USE_POLYNOMIAL:
        f.write(f"c0,{np.mean(trace['c0']):.6f},{np.std(trace['c0']):.6f}\n")
        f.write(f"c1,{np.mean(trace['c1']):.6f},{np.std(trace['c1']):.6f}\n")
    if USE_RAMP:
        f.write(f"r0,{np.mean(trace['r0']):.6f},{np.std(trace['r0']):.6f}\n")
        f.write(f"r1,{np.mean(trace['r1']):.6f},{np.std(trace['r1']):.6f}\n")
    f.write(f"error_inflation,{np.mean(trace['error_inflation']):.6f},{np.std(trace['error_inflation']):.6f}\n")

print("\nWASP-43b analysis complete!")
print("Generated files:")
print("- wasp43b_raw_data.png")
print("- corner_plot_wasp43b_map.png")
print("- wasp43b_fit_map.png")
print("- wasp43b_maps.png")
print("- wasp43b_map_profiles.png")
print("- wasp43b_map_params.txt")
