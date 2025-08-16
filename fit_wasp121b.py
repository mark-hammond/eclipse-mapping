# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import h5py
import exoplanet as xo
import os
from corner import corner


# inital input parameters
# Parameter,Mean,-1sigma,+1sigma,16th,50th,84th
# fpfs,0.006189086423180161,-3.445040615506975e-05,3.321301710860949e-05,0.006155488373510247,0.0061899387796653165,0.006223151796773926
# per,1.274924838525504,-3.882474008953807e-08,3.486776312122686e-08,1.2749248006952827,1.2749248395200228,1.274924874387786
# t0,58661.06381351084,-2.4621309421490878e-05,2.5654306227806956e-05,58661.0637888187,58661.06381344001,58661.06383909431
# a,3.7955530148421195,-0.004373382590542985,0.0043785193594478855,3.791141627207597,3.79551500979814,3.799893529157588
# Rs,1.437300832999673,-0.018244513004626484,0.016186378020627723,1.4198262489460889,1.4380707619507154,1.454257139971343
# inc,88.4711683469513,-0.10317280531594974,0.12146766086721072,88.36060564903086,88.46377845434681,88.58524611521402
# c0,0.9963231393364322,-7.403316795195902e-05,2.9161158723423064e-05,0.996264451626676,0.9963384847946279,0.9963676459533514
# c1,0.00016232055659861063,-0.0006766751715671023,0.001183006548659933,-0.00068682763315613,-1.015246158902783e-05,0.001172854087070905
# r0,0.0012994350195636188,-0.00019473739273036098,0.0002574941828148091,0.001079890127473456,0.001274627520203817,0.0015321217030186262
# r1,27.962962028357737,-6.976363775569208,6.021049637849028,20.74476716350513,27.72113093907434,33.74218057692337
# scatter_mult,1.4481236402977442,-0.02242393139408927,0.025507152286899926,1.4243265772386526,1.446750508632742,1.4722576609196418

# parameters fitted with phase curve
# Parameter Comparison:
# Parameter     Prior Mean ± SD      Fitted Mean ± SD
# -------------------------------------------------------
# planet_amp   0.006189 ± 0.001000  0.003273 ± 0.000297
# planet_radius 0.176000 ± 0.020000  0.188597 ± 0.045394
# semi_major   3.795553 ± 0.100000  3.980538 ± 0.302592
# inclination  88.471168 ± 0.100000  84.166258 ± 1.066889
# c0           0.996323 ± 0.001000  0.994728 ± 0.000113
# c1           0.000000 ± 0.001000  -0.001614 ± 0.000543
# r0           0.001299 ± 0.001000  0.002071 ± 0.000105
# r1           0.200000 ± 0.100000  0.015276 ± 0.002534
# cos_coeff    0.000000 ± 0.500000  -0.958021 ± 0.189769
# sin_coeff    0.000000 ± 0.500000  0.015102 ± 0.030315

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
    'Rp': 0.176,                   # Planet radius in solar radii (1.753 * 0.10045)
    'inc': 88.4711683469513,       # Orbital inclination in degrees
    'c0': 0.9963231393364322,      # Polynomial order 0 coefficient
    'c1': 0.00016232055659861063,  # Polynomial order 1 coefficient
    'r0': 0.0012994350195636188,   # Ramp magnitude
    'r1': 27.962962028357737,      # Ramp timescale
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

# Plot the raw data with more informative labels
plt.figure(figsize=(12, 5))
plt.errorbar(time, flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=transit_time, color='r', linestyle='--', label='Transit')
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
    # Create a star with limb darkening
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=2, amp=1.0),
        m=1.0,  # Fixed stellar mass
        r=params['Rs'],  # Fixed stellar radius
        prot=1.0
    )

    # Set limb darkening coefficients
    star.map[1] = params['c0']
    star.map[2] = params['c1']

    # Fit for the planet amplitude
    planet_amp = pm.Normal("planet_amp", mu=params['fpfs'], sd=0.001, testval=params['fpfs'])
    
    # Fit for planet radius and semi-major axis
    planet_radius = pm.Normal("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
    semi_major = pm.Normal("semi_major", mu=params['a'], sd=0.1, testval=params['a'])
    inclination = pm.Normal("inclination", mu=params['inc'], sd=0.1, testval=params['inc'])
    
    # Create the planet with some fitted parameters
    planet = starry.Secondary(
        starry.Map(ydeg=1, udeg=0, amp=planet_amp),
        m=0.0,
        r=planet_radius,  # Fitted planet radius
        inc=inclination,  # Fitted inclination
        porb=params['per'],
        a=semi_major,  # Fitted semi-major axis
        t0=params['t0']
    )

    # Add spherical harmonics coefficients
    ncoeff = planet.map.Ny - 1
    planet_mu = np.zeros(ncoeff)
    planet_cov = 1e-2 * np.eye(ncoeff)
    planet.map[1:, :] = pm.MvNormal("planet_y", planet_mu, planet_cov, shape=(ncoeff,))

    # Create the system
    system = starry.System(star, planet)
    
    # Compute the astrophysical model flux
    astro_flux = system.flux(time)
    
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
    
    # Save the map as a deterministic for plotting
    map_vals = pm.Deterministic("map", planet.map.render(projection="rect"))
    
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
        trace['planet_y'].reshape(len(trace['planet_y']), -1)
    ]).T
    labels = ['planet_amp', 'Rp/R*', 'a/R*', 'inc', 'c0', 'c1', 'r0', 'r1', 
              'Y₁₀', 'Y₁₁ᶜ', 'Y₁₁ˢ']
    fig = corner.corner(samples, labels=labels, 
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



# Plot the light curve fit
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')

# Calculate and plot the systematic components using fitted parameters
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

# Generate and plot full-orbit light curve
t_full = np.linspace(transit_time, transit_time + params['per'], 1000)
t_norm_full = t_full - t_full[0]

# Create a new system for the full orbit
star_full = starry.Primary(
    starry.Map(ydeg=0, udeg=2, amp=1.0),  # Fixed amplitude
    m=1.0,
    r=params['Rs'],
    prot=1.0
)
star_full.map[1] = params['c0']
star_full.map[2] = params['c1']

planet_full = starry.Secondary(
    starry.Map(ydeg=0, udeg=0, amp=np.mean(trace["planet_amp"])),
    m=0.0,
    r=np.mean(trace["planet_radius"]),
    inc=np.mean(trace["inclination"]),  # Use fitted inclination
    porb=params['per'],
    a=np.mean(trace["semi_major"]),
    t0=params['t0']
)

system_full = starry.System(star_full, planet_full, light_delay=True)

# Compute the full-orbit model with fitted parameters
astro_flux_full = system_full.flux(t_full)
baseline_full = np.mean(trace["c0"]) + np.mean(trace["c1"]) * t_norm_full
ramp_full = 1 + np.mean(trace["r0"]) * np.exp(-t_norm_full / np.mean(trace["r1"]))
flux_model_full = astro_flux_full * baseline_full * ramp_full

# Plot the full-orbit light curve
plt.figure(figsize=(12, 5))
plt.plot(t_full, flux_model_full.eval(), 'C0-', label='Full-orbit model')
plt.axvline(x=transit_time, color='r', linestyle='--', label='Transit')
plt.axvline(x=eclipse_time, color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [BJD]")
plt.ylabel("Relative Flux")
plt.title("WASP-121b Full-orbit Light Curve")
plt.grid(True)
plt.legend()
plt.savefig('wasp121b_full_orbit.png')
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

# Plot map distributions
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Get map samples
nsamples = len(trace["map"])
lon = np.linspace(-180, 180, trace["map"].shape[2])
lat = np.linspace(-90, 90, trace["map"].shape[1])

# Extract equatorial slice
eq_idx = len(lat) // 2
equator_samples = trace["map"][:, eq_idx, :]
equator_mean = np.mean(equator_samples, axis=0)
equator_std = np.std(equator_samples, axis=0)

# Extract north-south slice through substellar point
ss_idx = len(lon) // 2
ns_samples = trace["map"][:, :, ss_idx]
ns_mean = np.mean(ns_samples, axis=0)
ns_std = np.std(ns_samples, axis=0)

# Plot distributions
ax1.fill_between(lon, 
                 equator_mean - equator_std, 
                 equator_mean + equator_std, 
                 alpha=0.3, color='C0', label='1-sigma')
ax1.plot(lon, equator_mean, 'C0-', label='Mean')
ax1.set_xlabel('Longitude [deg]')
ax1.set_ylabel('Intensity')
ax1.set_title('Equatorial Map Distribution')
ax1.legend()
ax1.grid(True)

ax2.fill_between(lat, 
                 ns_mean - ns_std, 
                 ns_mean + ns_std, 
                 alpha=0.3, color='C0', label='1-sigma')
ax2.plot(lat, ns_mean, 'C0-', label='Mean')
ax2.set_xlabel('Latitude [deg]')
ax2.set_ylabel('Intensity')
ax2.set_title('North-South Distribution (Through Substellar Point)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('wasp121b_distributions.png')
plt.close()

# Print diagnostics
print("\nMap coefficient summary:")
for i, coef in enumerate(np.mean(trace["planet_y"], axis=0)):
    print("Y_{}: {:.3f} +/- {:.3f}".format(
        i+1, 
        coef, 
        np.std(trace["planet_y"][:, i])
    ))

# Calculate chi-squared
mean_model = np.mean(trace["flux"], axis=0)
chi2 = np.sum((flux - mean_model)**2 / flux_err**2)
print("\nReduced chi-squared: {:.2f}".format(chi2/len(flux)))

