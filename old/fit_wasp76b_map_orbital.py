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

# Key model parameters
MAP_DEGREE = 1        # Degree of spherical harmonic map
FOURIER_DEGREE = 1    # Degree of Fourier series for phase curve
FIT_ORBITAL = False   # Whether to fit orbital parameters
N_SAMPLES = 100       # Number of MCMC samples
N_TUNE = 100         # Number of tuning steps
TARGET_ACCEPT = 0.9   # Target acceptance rate for MCMC
N_CHAINS = 2          # Number of MCMC chains
N_CORES = 2           # Number of CPU cores to use

# Define orbital parameters (PLACEHOLDER VALUES - TO BE UPDATED)
params = {
    'fpfs': 0.0035,    # Planet-to-star flux ratio (placeholder)
    'per': 	1.80988198,  # Orbital period in days (from literature)
    't0': 60678.05902137451+0.15-0.5*1.80988198,    # Time of inferior conjunction (placeholder)
    'a': 4.1088,         # Semi-major axis in stellar radii (placeholder)
    'Rs': 1.744,        # Stellar radius in solar radii (from literature)
    'Rp': 0.1068,       # Planet radius in stellar radii (1.854 Rj converted)
    'inc': 89.623,      # Orbital inclination in degrees (placeholder)
    'c0': 1.0,        # Polynomial order 0 coefficient (placeholder)
    'c1': 0.0,        # Polynomial order 1 coefficient (placeholder)
    'r0': 0.0,        # Ramp magnitude (placeholder)
    'r1': 0.0         # Ramp timescale in days (placeholder)
}

# Load and clean the data
data = np.genfromtxt('WASP-76b_WhiteLight.csv', delimiter=',', skip_header=1)
time = data[:, 0]
flux = data[:, 3]  # Using systematics-corrected flux
flux_err = data[:, 2]  # Keep original error bars
sys_corr = data[:, 3]  # Systematics corrected flux

# Remove NaNs
valid_mask = ~np.isnan(flux)
time = time[valid_mask]
flux = flux[valid_mask]
flux_err = flux_err[valid_mask]
sys_corr = sys_corr[valid_mask]


# Convert to float64 for Theano compatibility
time = np.asarray(time, dtype=np.float64)
flux = np.asarray(flux, dtype=np.float64)
flux_err = np.asarray(flux_err, dtype=np.float64)
sys_corr = np.asarray(sys_corr, dtype=np.float64)

# Find the closest transit and eclipse times to our data
n_orbits = np.round((time[0] - params['t0']) / params['per'])
transit_time = params['t0'] + n_orbits * params['per']
eclipse_time = transit_time + params['per']/2

print(f"Data start time: {time[0]:.2f}")
print(f"Closest transit time: {transit_time:.2f}")
print(f"Closest eclipse time: {eclipse_time:.2f}")

# Check for NaN, inf, and negative values
print("\nData quality check:")
print(f"Number of NaN values in time: {np.sum(np.isnan(time))}")
print(f"Number of NaN values in flux: {np.sum(np.isnan(flux))}")
print(f"Number of NaN values in flux_err: {np.sum(np.isnan(flux_err))}")
print(f"Number of inf values in flux: {np.sum(np.isinf(flux))}")
print(f"Number of negative values in flux: {np.sum(flux < 0)}")
print(f"Number of negative values in flux_err: {np.sum(flux_err < 0)}")

# Plot the raw data
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='Systematics-corrected data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-76b Systematics-Corrected Data")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('wasp76b_raw_data.png')
plt.close()

# Define the PyMC3 model
with pm.Model() as model:
    # Create a star with uniform disk (no limb darkening needed for eclipse)
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=0, amp=1.0),
        m=0,  # Reference frame centered on star
        r=params['Rs'],  # Fixed stellar radius
        prot=1.0
    )

    # Fit for the planet amplitude
    planet_amp = pm.Normal("planet_amp", mu=params['fpfs'], sd=0.001, testval=params['fpfs'])
    
    # Fit for t0
    t0 = pm.Normal("t0", mu=params['t0'], sd=0.1, testval=params['t0'])
    
    # Handle orbital parameters based on FIT_ORBITAL flag
    if FIT_ORBITAL:
        # Fit orbital parameters
        planet_radius = pm.Normal("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
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
    planet_cov = 0.1**2 * np.eye(ncoeff)  # 0.1 standard deviation
    planet.map[1:, :] = pm.MvNormal("planet_y", planet_mu, planet_cov, shape=(ncoeff,))

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Total system flux is star + planet
    astro_flux = star_flux + planet_flux
    
    # Fit for systematic parameters
    # Polynomial baseline parameters
    c0 = pm.Normal("c0", mu=params['c0'], sd=0.001, testval=params['c0'])
    c1 = pm.Normal("c1", mu=params['c1'], sd=0.001, testval=params['c1'])
    
    # Ramp parameters
    r0 = pm.Normal("r0", mu=params['r0'], sd=0.001, testval=params['r0'])
    r1 = pm.Normal("r1", mu=params['r1'], sd=0.1, testval=params['r1'])
    
    # Add systematic effects
    t_norm = (time - time[0]) * 86400  # Convert days to seconds
    poly_baseline = c0 + c1 * t_norm / 86400
    ramp = 1 + r0 * pm.math.exp(-t_norm / (r1 * 86400))
    
    # Add error inflation parameter
    error_inflation = pm.HalfNormal("error_inflation", sd=0.5, testval=1.0)
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
    
    # Save the map as a deterministic for plotting
    map_vals = pm.Deterministic("map", planet.map.render(projection="rect"))
    
    # Define the likelihood with error inflation
    pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)

    # Plot initial model guess
    print("\nPlotting initial model guess...")
    with model:
        # Get initial values for all parameters
        initial_values = {
            'planet_amp': params['fpfs'],
            't0': params['t0'],
            'planet_radius': params['Rp'],
            'semi_major': params['a'],
            'inclination': params['inc'],
            'planet_y': np.zeros(ncoeff),  # Initial Y coefficients
            'c0': params['c0'],
            'c1': params['c1'],
            'r0': params['r0'],
            'r1': params['r1'],
            'error_inflation': 1.0
        }
        
        # Create initial model
        initial_star = starry.Primary(
            starry.Map(ydeg=0, udeg=0, amp=1.0),
            m=0,
            r=params['Rs'],
            prot=1.0
        )
        
        initial_planet = starry.Secondary(
            starry.Map(ydeg=MAP_DEGREE, udeg=0, amp=initial_values['planet_amp']),
            m=system_mass,
            r=initial_values['planet_radius'],
            inc=initial_values['inclination'],
            a=initial_values['semi_major'],
            t0=initial_values['t0']
        )
        
        initial_planet.porb = params['per']
        initial_planet.prot = params['per']
        initial_planet.theta0 = 180.0
        
        # Set initial Y coefficients
        initial_planet.map[1:, :] = initial_values['planet_y']
        
        # Create initial system
        initial_system = starry.System(initial_star, initial_planet)
        
        # Compute initial flux
        initial_star_flux, initial_planet_flux = initial_system.flux(time, total=False)
        initial_astro_flux = initial_star_flux + initial_planet_flux
        
        # Add systematic effects
        initial_t_norm = (time - time[0]) * 86400
        initial_baseline = initial_values['c0'] + initial_values['c1'] * initial_t_norm / 86400
        initial_ramp = 1 + initial_values['r0'] * np.exp(-initial_t_norm / (initial_values['r1'] * 86400))
        initial_flux = initial_astro_flux * initial_baseline * initial_ramp
        
        # Evaluate the initial flux
        initial_flux_eval = initial_flux.eval()
        
        # Plot initial model
        plt.figure(figsize=(12, 5))
        plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='Data')
        plt.plot(time - time[0], initial_flux_eval, 'r-', linewidth=2, label='Initial model')
        plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')
        plt.xlabel("Time [days from start]", fontsize=12)
        plt.ylabel("Relative Flux", fontsize=12)
        plt.title("WASP-76b Initial Model Guess")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig('wasp76b_initial_guess.png')
        plt.close()
        
        # Print initial parameter values
        print("\nInitial parameter values:")
        print(f"planet_amp: {initial_values['planet_amp']:.6f}")
        print(f"t0: {initial_values['t0']:.6f}")
        print(f"planet_radius: {initial_values['planet_radius']:.6f}")
        print(f"semi_major: {initial_values['semi_major']:.6f}")
        print(f"inclination: {initial_values['inclination']:.6f}")
        print(f"Y coefficients: {initial_values['planet_y']}")
        print(f"c0: {initial_values['c0']:.6f}")
        print(f"c1: {initial_values['c1']:.6f}")
        print(f"r0: {initial_values['r0']:.6f}")
        print(f"r1: {initial_values['r1']:.6f}")
        print(f"error_inflation: {initial_values['error_inflation']:.6f}")
        
        # Calculate initial chi-squared
        initial_chi2 = np.sum((flux - initial_flux_eval)**2 / (flux_err * initial_values['error_inflation'])**2)
        print(f"\nInitial chi-squared: {initial_chi2:.2f}")
        print(f"Initial reduced chi-squared: {initial_chi2/len(flux):.2f}")

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
        samples = np.vstack([
            trace['planet_amp'],
            trace['t0'],
            trace['planet_radius'],
            trace['semi_major'],
            trace['inclination'],
            trace['planet_y'].T,  # All y coefficients
            trace['c0'],
            trace['c1'],
            trace['r0'],
            trace['r1'],
            trace['error_inflation']
        ]).T
        labels = ['planet_amp', 't0', 'Rp/R*', 'a/R*', 'inc', 'Y10', 'Y11c', 'Y11s', 'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s',
                  'c0', 'c1', 'r0', 'r1', 'error_inflation']
    else:
        samples = np.vstack([
            trace['planet_amp'],
            trace['t0'],
            trace['planet_y'].T,  # All y coefficients
            trace['c0'],
            trace['c1'],
            trace['r0'],
            trace['r1'],
            trace['error_inflation']
        ]).T
        labels = ['planet_amp', 't0', 'Y10', 'Y11c', 'Y11s', 'Y20', 'Y21c', 'Y21s', 'Y22c', 'Y22s',
                  'c0', 'c1', 'r0', 'r1', 'error_inflation']
    fig = corner_module.corner(samples, labels=labels, 
                       show_titles=True,
                       title_fmt='.3f',
                       use_math_text=True,
                       quiet=True)
    plt.savefig('wasp76b_corner_plot_map_orbital.png')
    plt.close()

# Print parameter comparison
print("\nParameter Comparison:")
print("Parameter     Prior Mean ± SD      Fitted Mean ± SD")
print("-" * 55)
print(f"planet_amp   {params['fpfs']:.6f} ± 0.001000  {np.mean(trace['planet_amp']):.6f} ± {np.std(trace['planet_amp']):.6f}")
print(f"t0           {params['t0']:.6f} ± 0.100000  {np.mean(trace['t0']):.6f} ± {np.std(trace['t0']):.6f}")
if FIT_ORBITAL:
    print(f"planet_radius {params['Rp']:.6f} ± 0.020000  {np.mean(trace['planet_radius']):.6f} ± {np.std(trace['planet_radius']):.6f}")
    print(f"semi_major   {params['a']:.6f} ± 0.100000  {np.mean(trace['semi_major']):.6f} ± {np.std(trace['semi_major']):.6f}")
    print(f"inclination  {params['inc']:.6f} ± 0.100000  {np.mean(trace['inclination']):.6f} ± {np.std(trace['inclination']):.6f}")
else:
    print(f"planet_radius {params['Rp']:.6f} (fixed)")
    print(f"semi_major   {params['a']:.6f} (fixed)")
    print(f"inclination  {params['inc']:.6f} (fixed)")

# Print map coefficients
y_coeffs_mean = np.mean(trace['planet_y'], axis=0)
y_coeffs_std = np.std(trace['planet_y'], axis=0)
n_coeffs = len(y_coeffs_mean)
for i in range(n_coeffs):
    print(f"Y{i}          0.000000 ± 0.100000  {y_coeffs_mean[i]:.6f} ± {y_coeffs_std[i]:.6f}")

print(f"c0           {params['c0']:.6f} ± 0.001000  {np.mean(trace['c0']):.6f} ± {np.std(trace['c0']):.6f}")
print(f"c1           {params['c1']:.6f} ± 0.001000  {np.mean(trace['c1']):.6f} ± {np.std(trace['c1']):.6f}")
print(f"r0           {params['r0']:.6f} ± 0.001000  {np.mean(trace['r0']):.6f} ± {np.std(trace['r0']):.6f}")
print(f"r1           {params['r1']:.6f} ± 0.100000  {np.mean(trace['r1']):.6f} ± {np.std(trace['r1']):.6f}")
print(f"error_inflation 1.000000 ± 0.500000  {np.mean(trace['error_inflation']):.6f} ± {np.std(trace['error_inflation']):.6f}")

# Calculate chi-squared and get error inflation
mean_model = np.mean(trace["flux"], axis=0)
error_inflation = np.mean(trace['error_inflation'])  # Already a numpy float64
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
baseline = np.mean(trace["c0"]) + np.mean(trace["c1"]) * t_norm / 86400
r0_fit = np.mean(trace["r0"])
r1_fit = np.mean(trace["r1"])
ramp = 1 + r0_fit * np.exp(-t_norm / (r1_fit * 86400))
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
plt.title("WASP-76b Eclipse Light Curve with l=2 Map")

plt.savefig('wasp76b_fit_map_orbital.png')
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
plt.savefig('wasp76b_maps_orbital.png')
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
plt.savefig('wasp76b_map_profiles_orbital.png')
plt.close()

# Save the fitted parameters
with open('wasp76b_map_orbital_params.txt', 'w') as f:
    f.write("Parameter,Mean,Std\n")
    f.write(f"planet_amp,{np.mean(trace['planet_amp']):.6f},{np.std(trace['planet_amp']):.6f}\n")
    f.write(f"t0,{np.mean(trace['t0']):.6f},{np.std(trace['t0']):.6f}\n")
    if FIT_ORBITAL:
        f.write(f"planet_radius,{np.mean(trace['planet_radius']):.6f},{np.std(trace['planet_radius']):.6f}\n")
        f.write(f"semi_major,{np.mean(trace['semi_major']):.6f},{np.std(trace['semi_major']):.6f}\n")
        f.write(f"inclination,{np.mean(trace['inclination']):.6f},{np.std(trace['inclination']):.6f}\n")
    else:
        f.write(f"planet_radius,{params['Rp']:.6f},0.000000\n")
        f.write(f"semi_major,{params['a']:.6f},0.000000\n")
        f.write(f"inclination,{params['inc']:.6f},0.000000\n")
    for i in range(n_coeffs):
        f.write(f"Y{i},{y_coeffs_mean[i]:.6f},{y_coeffs_std[i]:.6f}\n")
    f.write(f"c0,{np.mean(trace['c0']):.6f},{np.std(trace['c0']):.6f}\n")
    f.write(f"c1,{np.mean(trace['c1']):.6f},{np.std(trace['c1']):.6f}\n")
    f.write(f"r0,{np.mean(trace['r0']):.6f},{np.std(trace['r0']):.6f}\n")
    f.write(f"r1,{np.mean(trace['r1']):.6f},{np.std(trace['r1']):.6f}\n")
    f.write(f"error_inflation,{np.mean(trace['error_inflation']):.6f},{np.std(trace['error_inflation']):.6f}\n")

# Fit a phase curve model without a map
print("\nFitting phase curve model...")

# Use fitted orbital parameters if FIT_ORBITAL is True, otherwise use initial parameters
if FIT_ORBITAL:
    pc_params = {
        'fpfs': np.mean(trace['planet_amp']),
        'per': params['per'],
        't0': np.mean(trace['t0']),
        'a': np.mean(trace['semi_major']),
        'Rs': params['Rs'],
        'Rp': np.mean(trace['planet_radius']),
        'inc': np.mean(trace['inclination']),
        'c0': np.mean(trace['c0']),
        'c1': np.mean(trace['c1']),
        'r0': np.mean(trace['r0']),
        'r1': np.mean(trace['r1'])
    }
else:
    pc_params = params.copy()
    pc_params['t0'] = np.mean(trace['t0'])

# Define the phase curve model
with pm.Model() as pc_model:
    # Create a star with uniform disk
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=0, amp=1.0),
        m=0,  # Reference frame centered on star
        r=pc_params['Rs'],  # Fixed stellar radius
        prot=1.0
    )

    # Fit for the planet amplitude
    planet_amp = pm.Normal("planet_amp", mu=pc_params['fpfs'], sd=pc_params['fpfs']/3.0, testval=pc_params['fpfs'])

    # Fit for t0
    t0 = pm.Normal("t0", mu=pc_params['t0'], sd=0.1, testval=pc_params['t0'])

    # Set up Fourier series coefficients for phase curve
    # For each degree, we have a cosine and sine term
    n_fourier = FOURIER_DEGREE
    fourier_coeffs = []
    for n in range(1, n_fourier + 1):
        # Cosine term
        fourier_coeffs.append(pm.Normal(f"a{n}", mu=0.0, sd=1, testval=0.0))
        # Sine term
        fourier_coeffs.append(pm.Normal(f"b{n}", mu=0.0, sd=1, testval=0.0))

    # Create a planet with uniform disk
    planet = starry.Secondary(
        starry.Map(ydeg=0, udeg=0, amp=planet_amp),
        m=system_mass,  # Use system mass from map fit
        r=pc_params['Rp'],
        inc=pc_params['inc'],
        a=pc_params['a'],
        t0=t0
    )
    
    # Set additional orbital parameters
    planet.porb = pc_params['per']
    planet.prot = pc_params['per']
    planet.theta0 = 180.0

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Calculate phase angle
    phase = 2 * np.pi * (time - t0) / pc_params['per']
    
    # Build the Fourier series for phase curve modulation
    phase_modulation = 1.0  # Start with constant term
    for n in range(1, n_fourier + 1):
        phase_modulation += fourier_coeffs[2*(n-1)] * pm.math.cos(n * phase)  # Cosine term
        phase_modulation += fourier_coeffs[2*(n-1)+1] * pm.math.sin(n * phase)  # Sine term
    
    # Apply phase curve modulation to planet flux
    planet_flux_modulated = planet_flux * phase_modulation
    
    # Total system flux is star + modulated planet
    astro_flux = star_flux + planet_flux_modulated
    
    # Use systematic parameters from map fit
    t_norm = (time - time[0]) * 86400  # Convert days to seconds
    poly_baseline = pc_params['c0'] + pc_params['c1'] * t_norm / 86400
    ramp = 1 + pc_params['r0'] * pm.math.exp(-t_norm / (pc_params['r1'] * 86400))
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
    
    # Define the likelihood with error inflation from map fit
    pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)

# Run the MCMC sampling for phase curve model
with pc_model:
    pc_trace = pmx.sample(
        tune=N_SAMPLES,
        draws=N_SAMPLES,
        target_accept=TARGET_ACCEPT,
        return_inferencedata=False,
        cores=N_CORES,
        chains=N_CHAINS
    )

# Plot the phase curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err * error_inflation, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot systematic components using fitted parameters
t_norm = time - time[0]  # Time in days
t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
baseline = pc_params['c0'] + pc_params['c1'] * t_norm
ramp = 1 + pc_params['r0'] * np.exp(-t_norm_seconds / (pc_params['r1'] * 86400))
systematic = baseline * ramp

plt.plot(t_norm, systematic, 'g--', alpha=0.7, label='Systematic trend')

# Plot the mean phase curve model fit
mean_pc_model = np.mean(pc_trace["flux"], axis=0)
plt.plot(t_norm, mean_pc_model, 'C1-', linewidth=2, label='Phase curve model')

# Plot a few individual samples with very low alpha to show uncertainty
for i in np.random.choice(range(len(pc_trace["flux"])), 10):
    plt.plot(t_norm, pc_trace["flux"][i], 'C1-', alpha=0.1)

plt.legend(fontsize=10)
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-76b Eclipse Light Curve with Phase Curve Model")

plt.savefig('wasp76b_fit_pc.png')
plt.close()

# Calculate chi-squared for phase curve model
chi2_pc = np.sum((flux - mean_pc_model)**2 / (flux_err * error_inflation)**2)
print("\nPhase curve model reduced chi-squared: {:.2f}".format(chi2_pc/len(flux)))

# Save the mean flux curves for comparison
with open('wasp76b_flux_curves.txt', 'w') as f:
    f.write("time,data,data_err,map_flux,pc_flux\n")
    for i in range(len(time)):
        f.write(f"{time[i]:.6f},{flux[i]:.6f},{flux_err[i]:.6f},{np.mean(trace['flux'][:,i]):.6f},{np.mean(pc_trace['flux'][:,i]):.6f}\n")

# Calculate deviations from mean phase curve model
data_deviation = flux - mean_pc_model
map_deviation = np.mean(trace['flux'], axis=0) - mean_pc_model  # Use mean map flux
map_deviation_std = np.std(trace['flux'], axis=0)  # Use std of map flux

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
    # Inflate the binned errors
    binned_data_dev_err[i] = np.sqrt(np.sum((flux_err[start_idx:end_idx] * error_inflation)**2)) / bin_size

# Plot the deviations
plt.figure(figsize=(12, 5))
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.errorbar((binned_time - eclipse_time)*24*60, binned_data_dev, 
             yerr=binned_data_dev_err, fmt='k.', alpha=0.3, ms=2, 
             label='Data - PC model (binned)')

# Plot map model deviations with inflated errors
plt.fill_between((time - eclipse_time)*24*60, 
                 map_deviation - map_deviation_std * error_inflation,
                 map_deviation + map_deviation_std * error_inflation,
                 alpha=0.3, color='C0', label='Map model - PC model (1σ)')
plt.plot((time - eclipse_time)*24*60, map_deviation, 'C0-', 
         linewidth=2, label='Map model - PC model (mean)')

# Add eclipse time
plt.axvline(x=0, color='b', linestyle='--', label='Eclipse')

plt.legend(fontsize=10)
plt.xlabel("Time from Eclipse [minutes]", fontsize=12)
plt.ylabel("Deviation from PC Model", fontsize=12)
plt.title("WASP-76b Eclipse Deviations from Phase Curve Model")

plt.tight_layout()
plt.savefig('wasp76b_deviations.png')
plt.close()

# Save the phase curve fitted parameters
with open('wasp76b_pc_params.txt', 'w') as f:
    f.write("Parameter,Mean,Std\n")
    f.write(f"planet_amp,{np.mean(pc_trace['planet_amp']):.6f},{np.std(pc_trace['planet_amp']):.6f}\n")
    f.write(f"t0,{np.mean(pc_trace['t0']):.6f},{np.std(pc_trace['t0']):.6f}\n")
    
    # Write Fourier coefficients
    for n in range(1, FOURIER_DEGREE + 1):
        f.write(f"a{n},{np.mean(pc_trace[f'a{n}']):.6f},{np.std(pc_trace[f'a{n}']):.6f}\n")
        f.write(f"b{n},{np.mean(pc_trace[f'b{n}']):.6f},{np.std(pc_trace[f'b{n}']):.6f}\n")
    
    if FIT_ORBITAL:
        f.write(f"planet_radius,{pc_params['Rp']:.6f},0.000000\n")
        f.write(f"semi_major,{pc_params['a']:.6f},0.000000\n")
        f.write(f"inclination,{pc_params['inc']:.6f},0.000000\n")
    else:
        f.write(f"planet_radius,{pc_params['Rp']:.6f},0.000000\n")
        f.write(f"semi_major,{pc_params['a']:.6f},0.000000\n")
        f.write(f"inclination,{pc_params['inc']:.6f},0.000000\n")
    f.write(f"c0,{pc_params['c0']:.6f},0.000000\n")
    f.write(f"c1,{pc_params['c1']:.6f},0.000000\n")
    f.write(f"r0,{pc_params['r0']:.6f},0.000000\n")
    f.write(f"r1,{pc_params['r1']:.6f},0.000000\n")

# Create summary plot with five subplots
fig = plt.figure(figsize=(12, 20))
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1])

# First subplot: Light curves
ax1 = fig.add_subplot(gs[0])
ax1.errorbar((time - eclipse_time)*24*60, flux, yerr=flux_err * error_inflation, 
             fmt='k.', alpha=0.3, ms=2, label='Data')
ax1.plot((time - eclipse_time)*24*60, mean_pc_model, 'C1-', linewidth=2, 
         label='Phase curve model')
ax1.plot((time - eclipse_time)*24*60, mean_model, 'C0-', linewidth=2, 
         label='Map model')
ax1.axvline(x=0, color='b', linestyle='--', label='Eclipse')
ax1.set_xlabel("Time from Eclipse [minutes]", fontsize=12)
ax1.set_ylabel("Relative Flux", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Second subplot: Mean map
ax2 = fig.add_subplot(gs[1])
planet_mu = np.mean(trace["map"], axis=0)
im = ax2.imshow(
    planet_mu,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)
plt.colorbar(im, ax=ax2, label="Temperature")
ax2.set_xlabel("Longitude [degrees]", fontsize=12)
ax2.set_ylabel("Latitude [degrees]", fontsize=12)
ax2.set_title("Mean Map", fontsize=12)

# Third subplot: Deviations
ax3 = fig.add_subplot(gs[2])
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.errorbar((binned_time - eclipse_time)*24*60, binned_data_dev, 
             yerr=binned_data_dev_err, fmt='k.', alpha=0.3, ms=2, 
             label='Data - PC model (binned)')
ax3.fill_between((time - eclipse_time)*24*60, 
                 map_deviation - map_deviation_std * error_inflation,
                 map_deviation + map_deviation_std * error_inflation,
                 alpha=0.3, color='C0', label='Map model - PC model (1σ)')
ax3.plot((time - eclipse_time)*24*60, map_deviation, 'C0-', 
         linewidth=2, label='Map model - PC model (mean)')
ax3.axvline(x=0, color='b', linestyle='--', label='Eclipse')
ax3.set_xlabel("Time from Eclipse [minutes]", fontsize=12)
ax3.set_ylabel("Deviation from PC Model", fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Fourth subplot: Equator profile
ax4 = fig.add_subplot(gs[3])
ax4.fill_between(lon, 
                 equator_mean - equator_std,
                 equator_mean + equator_std,
                 alpha=0.3, color='C0', label='1σ uncertainty')
ax4.plot(lon, equator_mean, 'C0-', linewidth=2, label='Mean')
ax4.set_xlabel("Longitude [degrees]", fontsize=12)
ax4.set_ylabel("Temperature", fontsize=12)
ax4.set_title("Map Profile Along Equator", fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Fifth subplot: Substellar meridian profile
ax5 = fig.add_subplot(gs[4])
ax5.fill_between(lat,
                 substellar_mean - substellar_std,
                 substellar_mean + substellar_std,
                 alpha=0.3, color='C0', label='1σ uncertainty')
ax5.plot(lat, substellar_mean, 'C0-', linewidth=2, label='Mean')
ax5.set_xlabel("Latitude [degrees]", fontsize=12)
ax5.set_ylabel("Temperature", fontsize=12)
ax5.set_title("Map Profile Through Substellar Point", fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wasp76b_summary.png')
plt.close() 