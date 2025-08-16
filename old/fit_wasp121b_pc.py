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

# Define orbital parameters
params = {
    'fpfs': 0.006189086423180161,  # Planet-to-star flux ratio
    'per': 1.274924838525504,      # Orbital period in days
    't0': 58661.06381351084,       # Time of inferior conjunction
    'a': 3.7955530148421195,       # Semi-major axis in stellar radii
    'Rs': 1.437300832999673,       # Stellar radius in solar radii
    'Rp': 0.176,                   # Planet radius in solar radii
    'inc': 88.4711683469513,       # Orbital inclination in degrees
    'c0': 0.9963231393364322,      # Polynomial order 0 coefficient
    'c1': 0.0,                     # Polynomial order 1 coefficient
    'r0': 0.0012994350195636188,   # Ramp magnitude
    'scatter_mult': 1.4481236402977442  # Error bar inflation
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

# Plot the raw data
plt.figure(figsize=(12, 5))
plt.errorbar(time, flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time, color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [BJD]")
plt.ylabel("Relative Flux")
plt.title("WASP-121b Raw Light Curve")
plt.grid(True)
plt.legend()
plt.savefig('wasp121b_raw.png')
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
    
    # Fit for planet radius and semi-major axis
    planet_radius = pm.Normal("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
    semi_major = pm.Normal("semi_major", mu=params['a'], sd=0.1, testval=params['a'])
    inclination = pm.Normal("inclination", mu=params['inc'], sd=0.1, testval=params['inc'])
    
    # Calculate system mass from Kepler's law using original semi-major axis
    a_physical = semi_major * params['Rs'] * const.R_sun.value  # Convert to meters
    p_physical = params['per'] * 24 * 3600  # Convert to seconds
    system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value
    
    # Create the planet with uniform map
    planet = starry.Secondary(
        starry.Map(ydeg=0, udeg=0, amp=planet_amp),
        m=system_mass,    # Use system mass directly (since star mass is 0)
        r=planet_radius,  # Fitted planet radius
        inc=inclination,  # Fitted inclination
        a=semi_major,     # Use original semi-major axis
        t0=params['t0']
    )
    
    # Set additional orbital parameters
    planet.porb = params['per']  # Orbital period
    planet.prot = params['per']  # Rotation period (synchronous)
    planet.theta0 = 180.0        # Initial phase angle

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux, separated into star and planet components
    star_flux, planet_flux = starry.System(star, planet, light_delay=True).flux(time, total=False)
    
    # Add phase curve modulation
    # First-order Fourier series: 1 + A*cos(phi) + B*sin(phi)
    # Phase is 2pi * (t - t0) / period
    phase = 2 * np.pi * ((time - params['t0']) / params['per'])
    
    # Fit for phase curve coefficients
    cos_coeff = pm.Normal("cos_coeff", mu=0.0, sd=0.5, testval=0.0)
    sin_coeff = pm.Normal("sin_coeff", mu=0.0, sd=0.5, testval=0.0)
    
    # Compute phase curve modulation
    phase_curve = 1.0 + cos_coeff * pm.math.cos(phase) + sin_coeff * pm.math.sin(phase)
    
    # Apply phase curve to planet flux only
    modulated_planet_flux = planet_flux * phase_curve
    
    # Total system flux is star + modulated planet
    astro_flux = star_flux + modulated_planet_flux
    
    # Fit for systematic parameters
    # Polynomial baseline parameters
    c0 = pm.Normal("c0", mu=params['c0'], sd=0.001, testval=params['c0'])
    c1 = pm.Normal("c1", mu=params['c1'], sd=0.001, testval=params['c1'])
    
    # Ramp parameters
    r0 = pm.Normal("r0", mu=params['r0'], sd=0.001, testval=params['r0'])
    r1 = pm.Normal("r1", mu=0.2, sd=0.1, testval=0.2)  # Timescale in days
    
    # Add systematic effects
    # Polynomial baseline: c0 + c1*(t-t[0])
    t_norm = (time - time[0]) * 86400  # Convert days to seconds
    poly_baseline = c0 + c1 * t_norm / 86400  # Keep polynomial baseline in days
    
    # Ramp: r0*exp(-(t-t[0])/r1)
    ramp = 1 + r0 * pm.math.exp(-t_norm / (r1 * 86400))  # Convert r1 to seconds to match t_norm
    
    # Combine all effects
    flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
    
    # Define the likelihood with inflated error bars
    pm.Normal("obs", mu=flux_model, sd=flux_err * params['scatter_mult'], observed=flux)

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
    
    # Save the trace for use in the map fitting
    pm.save_trace(trace, 'fit_wasp121b_pc_trace', overwrite=True)

    # Create corner plot
    samples = np.vstack([
        trace['planet_amp'],
        trace['planet_radius'],
        trace['semi_major'],
        trace['inclination'],
        trace['c0'],
        trace['c1'],
        trace['r0'],
        trace['r1'],
        trace['cos_coeff'],
        trace['sin_coeff']
    ]).T
    labels = ['planet_amp', 'Rp/R*', 'a/R*', 'inc', 'c0', 'c1', 'r0', 'r1', 
              'cos(phi)', 'sin(phi)']
    fig = corner_module.corner(samples, labels=labels, 
                       show_titles=True,
                       title_fmt='.3f',
                       use_math_text=True,
                       quiet=True)
    plt.savefig('corner_plot.png')
    plt.close()

# Print parameter comparison
print("\nParameter Comparison:")
print("Parameter     Prior Mean ± SD      Fitted Mean ± SD")
print("-" * 55)
print(f"planet_amp   {params['fpfs']:.6f} ± 0.001000  {np.mean(trace['planet_amp']):.6f} ± {np.std(trace['planet_amp']):.6f}")
print(f"planet_radius {params['Rp']:.6f} ± 0.020000  {np.mean(trace['planet_radius']):.6f} ± {np.std(trace['planet_radius']):.6f}")
print(f"semi_major   {params['a']:.6f} ± 0.100000  {np.mean(trace['semi_major']):.6f} ± {np.std(trace['semi_major']):.6f}")
print(f"inclination  {params['inc']:.6f} ± 0.100000  {np.mean(trace['inclination']):.6f} ± {np.std(trace['inclination']):.6f}")
print(f"c0           {params['c0']:.6f} ± 0.001000  {np.mean(trace['c0']):.6f} ± {np.std(trace['c0']):.6f}")
print(f"c1           {params['c1']:.6f} ± 0.001000  {np.mean(trace['c1']):.6f} ± {np.std(trace['c1']):.6f}")
print(f"r0           {params['r0']:.6f} ± 0.001000  {np.mean(trace['r0']):.6f} ± {np.std(trace['r0']):.6f}")
print(f"r1           0.200000 ± 0.100000  {np.mean(trace['r1']):.6f} ± {np.std(trace['r1']):.6f}")
print(f"cos_coeff    0.000000 ± 0.500000  {np.mean(trace['cos_coeff']):.6f} ± {np.std(trace['cos_coeff']):.6f}")
print(f"sin_coeff    0.000000 ± 0.500000  {np.mean(trace['sin_coeff']):.6f} ± {np.std(trace['sin_coeff']):.6f}")

# Plot the light curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot systematic components using fitted parameters
t_norm = time - time[0]  # Time in days
t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
baseline = np.mean(trace["c0"]) + np.mean(trace["c1"]) * t_norm
r0_fit = np.mean(trace["r0"])
r1_fit = np.mean(trace["r1"])
ramp = 1 + r0_fit * np.exp(-t_norm_seconds / (r1_fit * 86400))  # Convert r1 to seconds
systematic = baseline * ramp

plt.plot(t_norm, systematic, 'g--', alpha=0.7, label='Systematic trend')

# Add vertical lines for ramp visualization
ramp_start = 0  # Since we normalized time
ramp_timescale = r1_fit  # r1 is in days
plt.axvline(x=ramp_start, color='purple', linestyle=':', label='Ramp start')
plt.axvline(x=ramp_timescale, color='orange', linestyle=':', label=f'Ramp timescale (r1={r1_fit:.3f} days)')

# Plot the mean model fit
mean_model = np.mean(trace["flux"], axis=0)
plt.plot(t_norm, mean_model, 'C0-', linewidth=2, label='Mean model fit')

# Plot a few individual samples with very low alpha to show uncertainty
for i in np.random.choice(range(len(trace["flux"])), 10):
    plt.plot(t_norm, trace["flux"][i], 'C0-', alpha=0.1)

plt.legend(fontsize=10)
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-121b Eclipse Light Curve")

# Add text box with ramp parameters
plt.text(0.02, 0.02, 
         f'Ramp parameters:\nr0={r0_fit:.6f}\nr1={r1_fit:.6f} days', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig('wasp121b_fit.png')
plt.close()

# Calculate chi-squared
mean_model = np.mean(trace["flux"], axis=0)
chi2 = np.sum((flux - mean_model)**2 / flux_err**2)
print("\nReduced chi-squared: {:.2f}".format(chi2/len(flux)))

# Save the average fit
np.savetxt('wasp121b_pc_fit.txt', 
           np.column_stack([time, mean_model]),
           header='time flux')

# Save the fitted parameters
with open('wasp121b_pc_params.txt', 'w') as f:
    f.write("Parameter,Mean,Std\n")
    f.write(f"planet_amp,{np.mean(trace['planet_amp']):.6f},{np.std(trace['planet_amp']):.6f}\n")
    f.write(f"planet_radius,{np.mean(trace['planet_radius']):.6f},{np.std(trace['planet_radius']):.6f}\n")
    f.write(f"semi_major,{np.mean(trace['semi_major']):.6f},{np.std(trace['semi_major']):.6f}\n")
    f.write(f"inclination,{np.mean(trace['inclination']):.6f},{np.std(trace['inclination']):.6f}\n")
    f.write(f"c0,{np.mean(trace['c0']):.6f},{np.std(trace['c0']):.6f}\n")
    f.write(f"c1,{np.mean(trace['c1']):.6f},{np.std(trace['c1']):.6f}\n")
    f.write(f"r0,{np.mean(trace['r0']):.6f},{np.std(trace['r0']):.6f}\n")
    f.write(f"r1,{np.mean(trace['r1']):.6f},{np.std(trace['r1']):.6f}\n")
    f.write(f"cos_coeff,{np.mean(trace['cos_coeff']):.6f},{np.std(trace['cos_coeff']):.6f}\n")
    f.write(f"sin_coeff,{np.mean(trace['sin_coeff']):.6f},{np.std(trace['sin_coeff']):.6f}\n") 