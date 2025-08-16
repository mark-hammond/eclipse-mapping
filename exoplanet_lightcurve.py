# -*- coding: utf-8 -*-
import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import os
from corner import corner
import h5py
from astropy import constants as const

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
starry.config.lazy = True
starry.config.quiet = True

# Define orbital parameters (WASP-121b example)
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
    time = f['time'][:]
    flux = f['data'][:].squeeze()
    flux_err = f['err'][:].squeeze()
    
    # Handle masked values
    mask = ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]
    
    # Normalize
    flux = flux / np.median(flux)
    flux_err = flux_err / np.median(flux)

# Create a star with limb darkening (following working example)
star = starry.Primary(
    starry.Map(ydeg=0, udeg=2, amp=1.0),  # Add .eval()
    m=1.0,
    r=params['Rs'],
    prot=1.0
)

# Set limb darkening coefficients
star.map[1] = params['c0']
star.map[2] = params['c1']

# Create a planet with l=2 map
planet = starry.Secondary(
    starry.Map(ydeg=2, udeg=0, amp=params['fpfs']).eval(),  # Add .eval()
    m=0.0,
    r=np.sqrt(params['fpfs']),
    inc=params['inc'],
    porb=params['per'],
    a=params['a'],
    t0=params['t0']
)

# Create the system
system = starry.System(star, planet)

# Define the PyMC3 model
with pm.Model() as model:
    # The Ylm coefficients of the planet
    ncoeff = planet.map.Ny - 1
    planet_mu = np.zeros(ncoeff)
    planet_cov = 1e-2 * np.eye(ncoeff)
    planet_y = pm.MvNormal("planet_y", planet_mu, planet_cov, shape=(ncoeff,))
    
    # Create a new planet map
    planet_map = starry.Map(ydeg=2, udeg=0).eval()  # Add .eval()
    
    # Set the Ylm coefficients
    def set_planet_map(y):
        planet_map.amp = params['fpfs']
        planet_map[1:, :] = y
        return planet_map
    
    # Create the model planet
    model_planet = starry.Secondary(
        set_planet_map(planet_y),
        m=0.0,
        r=np.sqrt(params['fpfs']),
        inc=params['inc'],
        porb=params['per'],
        a=params['a'],
        t0=params['t0']
    )
    
    # Create the model system
    model_system = starry.System(star, model_planet)
    
    # Compute the model flux
    flux_model = model_system.flux(time)
    pm.Deterministic("flux_model", flux_model)
    
    # Save the planet map render
    planet_render = planet_map.render(projection="rect").eval()
    pm.Deterministic("planet_render", planet_render)
    
    # Define the likelihood
    pm.Normal("obs", mu=flux_model, sd=flux_err, observed=flux)

# Run the MCMC sampling
with model:
    trace = pmx.sample(
        tune=300,
        draws=300,
        target_accept=0.95,
        return_inferencedata=False,
        cores=2
    )

# Plot using the saved deterministics
plt.figure(figsize=(12, 5))
plt.plot(time, flux, 'k.', alpha=0.3, ms=3, label='Noisy data')
plt.plot(time, flux_model, 'b-', label='True model')
plt.xlabel('Time [days]')
plt.ylabel('Flux [normalized]')
plt.title('Exoplanet Light Curve')
plt.legend()
plt.grid(True)
plt.savefig('exoplanet_lightcurve.png')
plt.close()

# Plot the maps using the saved renders
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Create true map render (do this before PyMC3 model)
true_map = starry.Map(ydeg=2)
true_map.amp = params['fpfs']
true_map[1, 0] = 0.5
planet_true = true_map.render(projection="rect").eval()

# Mean and random sample from posterior
planet_mu = np.mean(trace["planet_render"], axis=0)
i = np.random.randint(len(trace["planet_render"]))
planet_draw = trace["planet_render"][i]

# Plot maps with colorbars
im0 = ax[0].imshow(
    planet_true,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)
im1 = ax[1].imshow(
    planet_mu,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)
im2 = ax[2].imshow(
    planet_draw,
    origin="lower",
    extent=(-180, 180, -90, 90),
    cmap="plasma",
)

# Add colorbars
plt.colorbar(im0, ax=ax[0])
plt.colorbar(im1, ax=ax[1])
plt.colorbar(im2, ax=ax[2])

ax[0].set_title("True Map")
ax[1].set_title("Mean Map")
ax[2].set_title("Random Sample")
plt.tight_layout()
plt.savefig('map_comparison.png')
plt.close()

# Create a corner plot for the planet parameters
samps = np.hstack((trace["planet_y"]))
nparams = samps.shape[0]  # Get number of parameters
fig, ax = plt.subplots(nparams, nparams, figsize=(12, 12))
labels = [
    r"$Y_{%d,%d}$" % (l, m)
    for l in range(2)  # ydeg=2 for planet
    for m in range(-l, l + 1)
]
corner(samps, fig=fig, labels=labels)
plt.tight_layout()
plt.savefig('corner_plot.png')
plt.close()

# Add diagnostics
print("\nFit diagnostics:")
print("\nPlanet Y_lm coefficients:")
for i, coef in enumerate(np.mean(trace["planet_y"], axis=0)):
    std_val = np.std(trace['planet_y'][:, i])
    print("Y_{}: {:.3f} +/- {:.3f}".format(i+1, coef, std_val))

# Calculate chi-squared
mean_model = np.mean(trace["flux_model"], axis=0)
chi2 = np.sum((flux - mean_model)**2 / flux_err**2)
print("\nReduced chi-squared: {:.2f}".format(chi2/len(flux)))

print("MCMC sampling completed. Results saved as images.")

# Add new plot for posterior map distributions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Get map samples
nsamples = len(trace["planet_render"])
lon = np.linspace(-180, 180, trace["planet_render"].shape[2])
lat = np.linspace(-90, 90, trace["planet_render"].shape[1])

# Extract equatorial slice (latitude = 0)
eq_idx = len(lat) // 2
equator_samples = trace["planet_render"][:, eq_idx, :]
equator_mean = np.mean(equator_samples, axis=0)
equator_std = np.std(equator_samples, axis=0)

# Extract north-south slice through substellar point (longitude = 0)
ss_idx = len(lon) // 2
ns_samples = trace["planet_render"][:, :, ss_idx]
ns_mean = np.mean(ns_samples, axis=0)
ns_std = np.std(ns_samples, axis=0)

# Plot equatorial distribution
ax1.fill_between(lon, 
                 equator_mean - equator_std, 
                 equator_mean + equator_std, 
                 alpha=0.3, color='C0', label='1σ')
ax1.plot(lon, equator_mean, 'C0-', label='Mean')

# Get true map equatorial slice
true_eq = planet_true[eq_idx, :]
ax1.plot(lon, true_eq, 'k--', label='True')

ax1.set_xlabel('Longitude [deg]')
ax1.set_ylabel('Intensity')
ax1.set_title('Equatorial Map Distribution')
ax1.legend()
ax1.grid(True)

# Plot north-south distribution
ax2.fill_between(lat, 
                 ns_mean - ns_std, 
                 ns_mean + ns_std, 
                 alpha=0.3, color='C0', label='1σ')
ax2.plot(lat, ns_mean, 'C0-', label='Mean')

# Get true map north-south slice
true_ns = planet_true[:, ss_idx]
ax2.plot(lat, true_ns, 'k--', label='True')

ax2.set_xlabel('Latitude [deg]')
ax2.set_ylabel('Intensity')
ax2.set_title('North-South Distribution (Through Substellar Point)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('map_distributions.png')
plt.close()