# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import astropy.constants as const

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
starry.config.lazy = True
starry.config.quiet = True

# Key model parameters
FOURIER_DEGREE = 2    # Degree of Fourier series for phase curve
N_SAMPLES = 1000       # Number of MCMC samples
N_TUNE = 1000         # Number of tuning steps
TARGET_ACCEPT = 0.9   # Target acceptance rate for MCMC
N_CHAINS = 2          # Number of MCMC chains
N_CORES = 2           # Number of CPU cores to use

# Data selection parameters
DATA_START = None     # Start index for data slice (None for beginning)
DATA_END = None      # End index for data slice (None for all data)
BIN_SIZE = 20        # Number of points to bin (None for no binning)

# Systematic parameters
USE_POLYNOMIAL = False  # Whether to fit polynomial baseline
USE_RAMP = False      # Whether to fit exponential ramp
FIT_ORBITAL = True    # Whether to fit orbital parameters

# Define orbital parameters (PLACEHOLDER VALUES - TO BE UPDATED)
params = {
    'fpfs': 0.001415,    # Planet-to-star flux ratio (placeholder)
    'per': 	1.80988198,  # Orbital period in days (from literature)
    't0': 60677.304758,    # Time of inferior conjunction (placeholder)
    'a': 4.031589,         # Semi-major axis in stellar radii (placeholder)
    'Rs': 1.744,        # Stellar radius in solar radii (from literature)
    'Rp': 0.193357,       # Planet radius in stellar radii
    'inc': 91.508127,      # Orbital inclination in degrees (placeholder)
    'c0': 1.0,        # Polynomial order 0 coefficient (placeholder)
    'c1': 0.0,        # Polynomial order 1 coefficient (placeholder)
    'r0': 0.0,        # Ramp magnitude (placeholder)
    'r1': 0.1,         # Ramp timescale in days (non-zero to avoid division issues)
    'u1': 0.366702,        # Linear limb darkening coefficient
    'u2': -0.348141         # Quadratic limb darkening coefficient
}

# Load and clean the data
data = np.genfromtxt('WASP-76b_WhiteLight.csv', delimiter=',', skip_header=1)

# Select data range
if DATA_START is None:
    DATA_START = 0
if DATA_END is None:
    DATA_END = len(data)
data = data[DATA_START:DATA_END]

# Extract columns
time = data[:, 0]
flux = data[:, 3]  # Using systematics-corrected flux
flux_err = data[:, 2]  # Keep original error bars

# Remove NaNs
valid_mask = ~np.isnan(flux)
time = time[valid_mask]
flux = flux[valid_mask]
flux_err = flux_err[valid_mask]

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
print(f"Number of points after NaN removal: {len(time)}")
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
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='Systematics-corrected data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-76b Systematics-Corrected Data")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('wasp76b_raw_data.png')
plt.close()

# Define the phase curve model
with pm.Model() as model:
    # Create a star with quadratic limb darkening
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=2, amp=1.0),  # udeg=2 for quadratic limb darkening
        m=0,  # Reference frame centered on star
        r=params['Rs'],  # Fixed stellar radius
        prot=1.0
    )
    
    # Fit limb darkening coefficients
    u1 = pm.Normal("u1", mu=params['u1'], sd=0.1, testval=params['u1'])
    u2 = pm.Normal("u2", mu=params['u2'], sd=0.1, testval=params['u2'])
    star.map[1] = u1  # Linear term
    star.map[2] = u2  # Quadratic term

    # Fit for the planet amplitude
    planet_amp = pm.Normal("planet_amp", mu=params['fpfs'], sd=params['fpfs']/3.0, testval=params['fpfs'])

    # Fit for t0 only if FIT_ORBITAL is true
    if FIT_ORBITAL:
        t0 = pm.Normal("t0", mu=params['t0'], sd=0.1, testval=params['t0'])
    else:
        t0 = params['t0']

    # Set up Fourier series coefficients for phase curve
    # For each degree, we have a cosine and sine term
    fourier_coeffs = []
    for n in range(1, FOURIER_DEGREE + 1):
        # Cosine term
        fourier_coeffs.append(pm.Normal(f"a{n}", mu=0.0, sd=1, testval=0.0))
        # Sine term
        fourier_coeffs.append(pm.Normal(f"b{n}", mu=0.0, sd=1, testval=0.0))

    # Fit orbital parameters
    planet_radius = pm.Bound(pm.Normal, lower=0.0)("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
    semi_major = pm.Normal("semi_major", mu=params['a'], sd=0.1, testval=params['a'])
    inclination = pm.Normal("inclination", mu=params['inc'], sd=0.1, testval=params['inc'])

    # Calculate system mass from Kepler's law
    a_physical = semi_major * params['Rs'] * const.R_sun.value  # Convert to meters
    p_physical = params['per'] * 24 * 3600  # Convert to seconds
    system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value

    # Create a planet with uniform disk
    planet = starry.Secondary(
        starry.Map(ydeg=0, udeg=0, amp=planet_amp),
        m=system_mass,  # Use system mass from Kepler's law
        r=planet_radius,  # Fitted planet radius
        inc=inclination,
        a=semi_major,
        t0=t0            # Fitted t0
    )
    
    # Set additional orbital parameters
    planet.porb = params['per']
    planet.prot = params['per']
    planet.theta0 = 180.0

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Calculate phase angle
    phase = 2 * np.pi * (time - t0) / params['per']
    
    # Build the Fourier series for phase curve modulation
    phase_modulation = 1.0  # Start with constant term
    for n in range(1, FOURIER_DEGREE + 1):
        phase_modulation += fourier_coeffs[2*(n-1)] * pm.math.cos(n * phase)  # Cosine term
        phase_modulation += fourier_coeffs[2*(n-1)+1] * pm.math.sin(n * phase)  # Sine term
    
    # Apply phase curve modulation to planet flux
    planet_flux_modulated = planet_flux * phase_modulation
    
    # Total system flux is star + modulated planet
    astro_flux = star_flux + planet_flux_modulated
    
    # Initialize systematic effects
    systematic = 1.0
    
    # Add polynomial baseline if enabled
    if USE_POLYNOMIAL:
        c0 = pm.Normal("c0", mu=params['c0'], sd=0.001, testval=params['c0'])
        c1 = pm.Normal("c1", mu=params['c1'], sd=0.001, testval=params['c1'])
        systematic *= (c0 + c1 * (time - time[0]))
    
    # Add ramp if enabled
    if USE_RAMP:
        r0 = pm.Normal("r0", mu=params['r0'], sd=0.001, testval=params['r0'])
        r1 = pm.Normal("r1", mu=params['r1'], sd=0.1, testval=params['r1'])
        systematic *= (1 + r0 * pm.math.exp(-(time - time[0]) / r1))
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * systematic)
    
    # Add error inflation parameter
    error_inflation = pm.HalfNormal("error_inflation", sd=0.5, testval=1.0)
    
    # Define the likelihood with error inflation
    pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)

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

# Print parameter comparison
print("\nParameter Comparison:")
print("Parameter     Prior Mean ± SD      Fitted Mean ± SD")
print("-" * 55)
print(f"planet_amp   {params['fpfs']:.6f} ± {params['fpfs']/3.0:.6f}  {np.mean(trace['planet_amp']):.6f} ± {np.std(trace['planet_amp']):.6f}")
if FIT_ORBITAL:
    print(f"t0          {params['t0']:.6f} ± 0.100000  {np.mean(trace['t0']):.6f} ± {np.std(trace['t0']):.6f}")
else:
    print(f"t0          {params['t0']:.6f} (fixed)")
print(f"planet_radius {params['Rp']:.6f} ± 0.020000  {np.mean(trace['planet_radius']):.6f} ± {np.std(trace['planet_radius']):.6f}")
print(f"semi_major   {params['a']:.6f} ± 0.100000  {np.mean(trace['semi_major']):.6f} ± {np.std(trace['semi_major']):.6f}")
print(f"inclination  {params['inc']:.6f} ± 0.100000  {np.mean(trace['inclination']):.6f} ± {np.std(trace['inclination']):.6f}")
print(f"u1          {params['u1']:.6f} ± 0.100000  {np.mean(trace['u1']):.6f} ± {np.std(trace['u1']):.6f}")
print(f"u2          {params['u2']:.6f} ± 0.100000  {np.mean(trace['u2']):.6f} ± {np.std(trace['u2']):.6f}")

# Print Fourier coefficients
for n in range(1, FOURIER_DEGREE + 1):
    print(f"a{n}          0.000000 ± 1.000000  {np.mean(trace[f'a{n}']):.6f} ± {np.std(trace[f'a{n}']):.6f}")
    print(f"b{n}          0.000000 ± 1.000000  {np.mean(trace[f'b{n}']):.6f} ± {np.std(trace[f'b{n}']):.6f}")

# Print systematic parameters if they were used
if USE_POLYNOMIAL:
    print(f"c0           {params['c0']:.6f} ± 0.001000  {np.mean(trace['c0']):.6f} ± {np.std(trace['c0']):.6f}")
    print(f"c1           {params['c1']:.6f} ± 0.001000  {np.mean(trace['c1']):.6f} ± {np.std(trace['c1']):.6f}")
if USE_RAMP:
    print(f"r0           {params['r0']:.6f} ± 0.001000  {np.mean(trace['r0']):.6f} ± {np.std(trace['r0']):.6f}")
    print(f"r1           {params['r1']:.6f} ± 0.100000  {np.mean(trace['r1']):.6f} ± {np.std(trace['r1']):.6f}")

print(f"error_inflation 1.000000 ± 0.500000  {np.mean(trace['error_inflation']):.6f} ± {np.std(trace['error_inflation']):.6f}")

# Calculate chi-squared and get error inflation
mean_model = np.mean(trace["flux"], axis=0)
error_inflation = np.mean(trace['error_inflation'])
chi2 = np.sum((flux - mean_model)**2 / (flux_err * error_inflation)**2)
print(f"\nError inflation factor: {error_inflation:.6f}")
print(f"Reduced chi-squared: {chi2/len(flux):.2f}")

# Plot the light curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err * error_inflation, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot systematic components using fitted parameters
t_norm = time - time[0]  # Time in days

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
    ramp = 1 + r0_fit * np.exp(-t_norm / r1_fit)
    systematic *= ramp
    plt.plot(t_norm, ramp, 'g--', alpha=0.7, label='Ramp trend')

# Plot the mean model fit
plt.plot(t_norm, mean_model, 'C0-', linewidth=2, label='Mean model fit')

# Plot a few individual samples with very low alpha to show uncertainty
for i in np.random.choice(range(len(trace["flux"])), 10):
    plt.plot(t_norm, trace["flux"][i], 'C0-', alpha=0.1)

plt.legend(fontsize=10)
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-76b Light Curve with Phase Curve Model")
plt.grid(True, alpha=0.3)

plt.savefig('wasp76b_fit_orbital_pc.png')
plt.close()

# Save the fitted parameters
with open('wasp76b_orbital_pc_params.txt', 'w') as f:
    f.write("Parameter,Mean,Std\n")
    f.write(f"planet_amp,{np.mean(trace['planet_amp']):.6f},{np.std(trace['planet_amp']):.6f}\n")
    if FIT_ORBITAL:
        f.write(f"t0,{np.mean(trace['t0']):.6f},{np.std(trace['t0']):.6f}\n")
    else:
        f.write(f"t0,{params['t0']:.6f},0.000000\n")
    f.write(f"planet_radius,{np.mean(trace['planet_radius']):.6f},{np.std(trace['planet_radius']):.6f}\n")
    f.write(f"semi_major,{np.mean(trace['semi_major']):.6f},{np.std(trace['semi_major']):.6f}\n")
    f.write(f"inclination,{np.mean(trace['inclination']):.6f},{np.std(trace['inclination']):.6f}\n")
    f.write(f"u1,{np.mean(trace['u1']):.6f},{np.std(trace['u1']):.6f}\n")
    f.write(f"u2,{np.mean(trace['u2']):.6f},{np.std(trace['u2']):.6f}\n")
    
    # Write Fourier coefficients
    for n in range(1, FOURIER_DEGREE + 1):
        f.write(f"a{n},{np.mean(trace[f'a{n}']):.6f},{np.std(trace[f'a{n}']):.6f}\n")
        f.write(f"b{n},{np.mean(trace[f'b{n}']):.6f},{np.std(trace[f'b{n}']):.6f}\n")
    
    if USE_POLYNOMIAL:
        f.write(f"c0,{np.mean(trace['c0']):.6f},{np.std(trace['c0']):.6f}\n")
        f.write(f"c1,{np.mean(trace['c1']):.6f},{np.std(trace['c1']):.6f}\n")
    if USE_RAMP:
        f.write(f"r0,{np.mean(trace['r0']):.6f},{np.std(trace['r0']):.6f}\n")
        f.write(f"r1,{np.mean(trace['r1']):.6f},{np.std(trace['r1']):.6f}\n")
    f.write(f"error_inflation,{np.mean(trace['error_inflation']):.6f},{np.std(trace['error_inflation']):.6f}\n") 