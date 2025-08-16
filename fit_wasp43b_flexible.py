# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import corner as corner_module
import astropy.constants as const
import os

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
starry.config.lazy = True
starry.config.quiet = True

# Key model parameters
MAP_DEGREE = 2        # Degree of spherical harmonic map
FOURIER_DEGREE = 2    # Degree of Fourier series for phase curve
FIT_ORBITAL = False   # Whether to fit orbital parameters
FIT_LIMB_DARK = True  # Whether to fit limb darkening parameters
NORMALIZE = False      # Whether to normalize the data to have mean = 1.0
N_SAMPLES = 300       # Number of MCMC samples
N_TUNE = 300         # Number of tuning steps
TARGET_ACCEPT = 0.95  # Higher acceptance rate for better mixing
N_CHAINS = 2          # Number of MCMC chains
N_CORES = 2           # Number of CPU cores to use

# Data selection parameters
DATA_START = None     # Start index for data slice (None for beginning)
DATA_END = None      # End index for data slice (None for all data)
BIN_SIZE = 10      # Number of points to bin (None for no binning)
DEVIATION_BIN_SIZE = 1  # Number of points to bin for deviation plots

# Systematic parameters
USE_POLYNOMIAL = True   # Whether to fit polynomial baseline
USE_RAMP = True       # Whether to fit exponential ramp

# Create output directory
OUTPUT_DIR = "wasp43b_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

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
    'fpfs': 0.005,        # Planet-to-star flux ratio (realistic eclipse depth)
    'per': 0.8134740621723353,  # Orbital period in days (from literature)
    't0': 55934.292283,   # Transit time (BMJD) from Eclipse Map Fit
    'a': 4.859,           # Semi-major axis in stellar radii from Eclipse Map Fit
    'Rs': 1.0,            # Stellar radius = 1.0 (reference unit)
    'Rp': 0.15839,        # Planet radius in stellar radii from Eclipse Map Fit
    'inc': 82.106,        # Orbital inclination in degrees from Eclipse Map Fit
    'c0': 0.0,      # Constant baseline deviation from 1.0
    'c1': 0.0,      # Linear trend deviation from 1.0
    'r0': 0.0,      # Ramp magnitude deviation from 1.0
    'r1': 3.7,            # Ramp time constant (1/day) from Eclipse Map Fit
    'u1': 0.0182,         # Limb darkening parameter q1 from Eclipse Map Fit
    'u2': 0.595           # Limb darkening parameter q2 from Eclipse Map Fit
}

print(f"\nSystem parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# Calculate expected eclipse depth
expected_eclipse_depth = (params['Rp'] / params['Rs'])**2
print(f"\nExpected eclipse depth (Rp/Rs)²: {expected_eclipse_depth:.6f} = {expected_eclipse_depth*1e6:.1f} ppm")

# Data preprocessing function
def preprocess_data(time, flux, flux_err, data_start=None, data_end=None, bin_size=None, NORMALIZE=False):
    """Preprocess the data: select range, bin, normalize"""
    
    # Select data range
    if data_start is None:
        data_start = 0
    if data_end is None:
        data_end = len(time)
    time = time[data_start:data_end]
    flux = flux[data_start:data_end]
    flux_err = flux_err[data_start:data_end]

    # Bin the data if requested
    if bin_size is not None and bin_size > 1:
        n_bins = len(time) // bin_size
        binned_time = np.zeros(n_bins)
        binned_flux = np.zeros(n_bins)
        binned_flux_err = np.zeros(n_bins)
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size
            binned_time[i] = np.mean(time[start_idx:end_idx])
            binned_flux[i] = np.mean(flux[start_idx:end_idx])
            # Error propagation for binning
            binned_flux_err[i] = np.sqrt(np.sum(flux_err[start_idx:end_idx]**2)) / bin_size
        
        time = binned_time
        flux = binned_flux
        flux_err = binned_flux_err

    print(f"\nData selection summary:")
    print(f"Using data points {data_start} to {data_end}")
    print(f"Number of points after selection: {len(time)}")
    if bin_size is not None:
        print(f"Binned every {bin_size} points")
        print(f"Final number of points: {len(time)}")

    # Convert to float64 for Theano compatibility
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # Normalize the data to have mean around 1.0
    if NORMALIZE:
        flux_mean = np.mean(flux)
        flux = flux / flux_mean
        flux_err = flux_err / flux_mean
        print(f"Normalized flux to mean = 1.0")
        print(f"Normalized flux range: {np.min(flux):.6f} to {np.max(flux):.6f}")
    
    return time, flux, flux_err

# Flexible fitting function
def fit_exoplanet_model(time, flux, flux_err, params, model_type="map", 
                       map_degree=2, fourier_degree=2, 
                       fit_orbital=False, fit_limb_dark=True,
                       use_polynomial=True, use_ramp=True,
                       n_samples=500, n_tune=500, target_accept=0.95,
                       n_chains=2, n_cores=2):
    """
    Flexible function to fit either spherical harmonic maps or phase curves
    
    Parameters:
    -----------
    model_type : str
        Either "map" for spherical harmonic mapping or "phase_curve" for Fourier series
    map_degree : int
        Degree of spherical harmonic map (only used if model_type="map")
    fourier_degree : int
        Degree of Fourier series (only used if model_type="phase_curve")
    """
    
    print(f"\nFitting {model_type} model...")
    
    # Find eclipse time
    n_orbits = np.round((time[0] - params['t0']) / params['per'])
    transit_time = params['t0'] + n_orbits * params['per']
    eclipse_time = transit_time + params['per']/2
    
    with pm.Model() as model:
        # Create a star with quadratic limb darkening
        star = starry.Primary(
            starry.Map(ydeg=0, udeg=2, amp=1.0),
            m=0,
            r=params['Rs'],
            prot=1.0
        )
        
        # Set the limb darkening coefficients
        if fit_limb_dark:
            u1 = pm.Normal("u1", mu=params['u1'], sd=0.1, testval=params['u1'])
            u2 = pm.Normal("u2", mu=params['u2'], sd=0.1, testval=params['u2'])
            star.map[1] = u1
            star.map[2] = u2
        else:
            star.map[1] = params['u1']
            star.map[2] = params['u2']

        # Fit for the planet amplitude - use more conservative prior
        planet_amp = pm.Normal("planet_amp", mu=params['fpfs'], sd=0.003, testval=params['fpfs'])
        
        # Handle orbital parameters
        if fit_orbital:
            t0 = pm.Normal("t0", mu=params['t0'], sd=0.1, testval=params['t0'])
            planet_radius = pm.Bound(pm.Normal, lower=0.0)("planet_radius", mu=params['Rp'], sd=0.02, testval=params['Rp'])
            semi_major = pm.Normal("semi_major", mu=params['a'], sd=0.1, testval=params['a'])
            inclination = pm.Normal("inclination", mu=params['inc'], sd=0.1, testval=params['inc'])
        else:
            t0 = params['t0']
            planet_radius = params['Rp']
            semi_major = params['a']
            inclination = params['inc']
        
        # Calculate system mass
        a_physical = semi_major * params['Rs'] * const.R_sun.value
        p_physical = params['per'] * 24 * 3600
        system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value
        
        if model_type == "map":
            # Create planet with spherical harmonic map
            planet = starry.Secondary(
                starry.Map(ydeg=map_degree, udeg=0, amp=planet_amp),
                m=system_mass,
                r=planet_radius,
                inc=inclination,
                a=semi_major,
                t0=t0
            )
            
            # Set orbital parameters
            planet.porb = params['per']
            planet.prot = params['per']
            planet.theta0 = 180.0
            
            # Fit Ylm coefficients - more conservative prior
            ncoeff = planet.map.Ny - 1
            planet_mu = np.zeros(ncoeff)
            planet_cov = np.eye(ncoeff)  # Much smaller prior
            planet.map[1:, :] = pm.MvNormal("planet_y", planet_mu, planet_cov, shape=(ncoeff,))
            
            # Create system and compute flux
            system = starry.System(star, planet)
            star_flux, planet_flux = system.flux(time, total=False)
            astro_flux = star_flux + planet_flux
            
            # Save map for plotting
            map_vals = pm.Deterministic("map", planet.map.render(projection="rect"))
            
        elif model_type == "phase_curve":
            # Create planet with uniform disk
            planet = starry.Secondary(
                starry.Map(ydeg=0, udeg=0, amp=planet_amp),
                m=system_mass,
                r=planet_radius,
                inc=inclination,
                a=semi_major,
                t0=t0
            )
            
            # Set orbital parameters
            planet.porb = params['per']
            planet.prot = params['per']
            planet.theta0 = 180.0
            
            # Create system and compute base flux
            system = starry.System(star, planet)
            star_flux, planet_flux = system.flux(time, total=False)
            
            # Calculate phase angle
            phase = 2 * np.pi * (time - t0) / params['per']
            
            # Set up Fourier series coefficients - more conservative priors
            fourier_coeffs = []
            for n in range(1, fourier_degree + 1):
                fourier_coeffs.append(pm.Normal(f"a{n}", mu=0.0, sd=1.0, testval=0.0))  # Cosine
                fourier_coeffs.append(pm.Normal(f"b{n}", mu=0.0, sd=1.0, testval=0.0))  # Sine
            
            # Build Fourier series modulation
            phase_modulation = 1.0
            for n in range(1, fourier_degree + 1):
                phase_modulation += fourier_coeffs[2*(n-1)] * pm.math.cos(n * phase)
                phase_modulation += fourier_coeffs[2*(n-1)+1] * pm.math.sin(n * phase)
            
            # Apply modulation to planet flux
            planet_flux_modulated = planet_flux * phase_modulation
            astro_flux = star_flux + planet_flux_modulated
        
        # Add systematic effects
        t_norm = (time - time[0]) * 86400
        
        if use_polynomial:
            c0 = pm.Normal("c0", mu=0.0, sd=0.001, testval=0.0)  # Deviation from 1.0
            c1 = pm.Normal("c1", mu=0.0, sd=0.001, testval=0.0)  # Deviation from 1.0
            poly_baseline = 1.0 + c0 + c1 * t_norm / 86400  # Start at 1.0, add deviations
        else:
            poly_baseline = 1.0
        
        if use_ramp:
            r0 = pm.Normal("r0", mu=0.0, sd=0.001, testval=0.0)  # Deviation from 1.0
            r1 = pm.Normal("r1", mu=1.0, sd=0.1, testval=1.0)    # Time constant
            ramp = 1.0 + r0 * pm.math.exp(-t_norm / (r1 * 86400))  # Start at 1.0, add ramp
        else:
            ramp = 1.0
        
        # Error inflation - more conservative prior
        error_inflation = pm.HalfNormal("error_inflation", sd=0.1, testval=1.0)
        
        # Final model
        flux_model = pm.Deterministic("flux", astro_flux * poly_baseline * ramp)
        
        # Likelihood
        pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)
    
    # First optimize to get good starting point
    print(f"Optimizing {model_type} model...")
    with model:
        map_soln = pmx.optimize()
        
    # Run MCMC with better settings for convergence
    print(f"Running MCMC for {model_type} model...")
    print(f"Parameters to fit: {[var.name for var in model.free_RVs]}")
    
    with model:
        trace = pmx.sample(
            tune=n_tune,
            draws=n_samples,
            target_accept=target_accept,
            return_inferencedata=False,
            cores=n_cores,
            chains=n_chains,
            start=map_soln,  # Start from optimized solution
            random_seed=42
        )
    
    print(f"MCMC complete. Trace shape: {trace['planet_amp'].shape}")
    print(f"Available parameters: {list(trace.varnames)}")
    
    return trace, model

# Preprocess data
time, flux, flux_err = preprocess_data(time, flux, flux_err, 
                                     DATA_START, DATA_END, BIN_SIZE, NORMALIZE)

# Find eclipse time for plotting
n_orbits = np.round((time[0] - params['t0']) / params['per'])
transit_time = params['t0'] + n_orbits * params['per']
eclipse_time = transit_time + params['per']/2

print(f"Data start time: {time[0]:.2f}")
print(f"Closest transit time: {transit_time:.2f}")
print(f"Closest eclipse time: {eclipse_time:.2f}")

# Plot raw data
plt.figure(figsize=(12, 5))
plt.errorbar(time - time[0], flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='WASP-43b data')
plt.axvline(x=eclipse_time - time[0], color='b', linestyle='--', label='Eclipse')
plt.xlabel("Time [days from start]", fontsize=12)
plt.ylabel("Relative Flux", fontsize=12)
plt.title("WASP-43b Light Curve Data")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_raw_data.png'))
plt.close()


print("\n" + "="*60)
print("FITTING PHASE CURVE MODEL")
print("="*60)

pc_trace, pc_model = fit_exoplanet_model(
    time, flux, flux_err, params,
    model_type="phase_curve",
    fourier_degree=FOURIER_DEGREE,
    fit_orbital=FIT_ORBITAL,
    fit_limb_dark=FIT_LIMB_DARK,
    use_polynomial=USE_POLYNOMIAL,
    use_ramp=USE_RAMP,
    n_samples=N_SAMPLES,
    n_tune=N_TUNE,
    target_accept=TARGET_ACCEPT,
    n_chains=N_CHAINS,
    n_cores=N_CORES
)

pc_mean_model = np.mean(pc_trace["flux"], axis=0)
pc_error_inflation = np.mean(pc_trace['error_inflation'])

# Plot comparison - simple version
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

t_plot = time - time[0]
ax.errorbar(t_plot, flux, yerr=flux_err * pc_error_inflation, fmt='k.', alpha=0.3, ms=2, label='Data')
ax.fill_between(t_plot, np.quantile(pc_trace["flux"], 0.16, axis=0), np.quantile(pc_trace["flux"], 0.84, axis=0), color='C1', alpha=0.2)
ax.plot(t_plot, pc_mean_model, 'C1-', linewidth=2, label='Phase curve model')

# Add systematic trends if they were fitted
if USE_POLYNOMIAL or USE_RAMP:
    t_norm = time - time[0]  # Time in days
    t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
    
    systematic_total = np.ones_like(t_norm)
    
    if USE_POLYNOMIAL:
        c0_fit = np.mean(pc_trace["c0"])
        c1_fit = np.mean(pc_trace["c1"])
        poly_baseline = 1.0 + c0_fit + c1_fit * t_norm
        systematic_total *= poly_baseline
        ax.plot(t_plot, poly_baseline, 'g--', alpha=0.7, linewidth=1.5, label=f'Polynomial (c0={c0_fit:.4f}, c1={c1_fit:.4f})')
    
    if USE_RAMP:
        r0_fit = np.mean(pc_trace["r0"])
        r1_fit = np.mean(pc_trace["r1"])
        ramp = 1.0 + r0_fit * np.exp(-t_norm_seconds / (r1_fit * 86400))
        systematic_total *= ramp
        ax.plot(t_plot, ramp, 'm--', alpha=0.7, linewidth=1.5, label=f'Ramp (r0={r0_fit:.4f}, r1={r1_fit:.2f})')
    
    # Plot total systematic effect
    ax.plot(t_plot, systematic_total, 'r--', alpha=0.8, linewidth=2, label='Total systematic')

ax.axvline(x=eclipse_time - time[0], color='b', linestyle='--', alpha=0.5, label='Eclipse')
ax.set_ylabel("Relative Flux", fontsize=12)
ax.set_title("WASP-43b: Phase Curve Model")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_fit_pc.png'))

# Fit both models
print("\n" + "="*60)
print("FITTING SPHERICAL HARMONIC MAP MODEL")
print("="*60)

map_trace, map_model = fit_exoplanet_model(
    time, flux, flux_err, params,
    model_type="map",
    map_degree=MAP_DEGREE,
    fit_orbital=FIT_ORBITAL,
    fit_limb_dark=FIT_LIMB_DARK,
    use_polynomial=USE_POLYNOMIAL,
    use_ramp=USE_RAMP,
    n_samples=N_SAMPLES,
    n_tune=N_TUNE,
    target_accept=TARGET_ACCEPT,
    n_chains=N_CHAINS,
    n_cores=N_CORES
)

map_mean_model = np.mean(map_trace["flux"], axis=0)
map_error_inflation = np.mean(map_trace['error_inflation'])

# Plot comparison - simple version
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

t_plot = time - time[0]
ax.errorbar(t_plot, flux, yerr=flux_err * map_error_inflation, fmt='k.', alpha=0.3, ms=2, label='Data')
ax.fill_between(t_plot, np.quantile(map_trace["flux"], 0.16, axis=0), np.quantile(map_trace["flux"], 0.84, axis=0), color='C0', alpha=0.2)
ax.plot(t_plot, map_mean_model, 'C0-', linewidth=2, label='Map model')

# Add systematic trends if they were fitted
if USE_POLYNOMIAL or USE_RAMP:
    t_norm = time - time[0]  # Time in days
    t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
    
    systematic_total = np.ones_like(t_norm)
    
    if USE_POLYNOMIAL:
        c0_fit = np.mean(map_trace["c0"])
        c1_fit = np.mean(map_trace["c1"])
        poly_baseline = 1.0 + c0_fit + c1_fit * t_norm
        systematic_total *= poly_baseline
        ax.plot(t_plot, poly_baseline, 'g--', alpha=0.7, linewidth=1.5, label=f'Polynomial (c0={c0_fit:.4f}, c1={c1_fit:.4f})')
    
    if USE_RAMP:
        r0_fit = np.mean(map_trace["r0"])
        r1_fit = np.mean(map_trace["r1"])
        ramp = 1.0 + r0_fit * np.exp(-t_norm_seconds / (r1_fit * 86400))
        systematic_total *= ramp
        ax.plot(t_plot, ramp, 'm--', alpha=0.7, linewidth=1.5, label=f'Ramp (r0={r0_fit:.4f}, r1={r1_fit:.2f})')
    
    # Plot total systematic effect
    ax.plot(t_plot, systematic_total, 'r--', alpha=0.8, linewidth=2, label='Total systematic')

ax.axvline(x=eclipse_time - time[0], color='b', linestyle='--', alpha=0.5, label='Eclipse')
ax.set_ylabel("Relative Flux", fontsize=12)
ax.set_title("WASP-43b: Map Model")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_fit_map.png'))


# Save simple diagnostics
def save_model_diagnostics(trace, model_name, params, time, flux, flux_err, output_dir):
    """Save simple model diagnostics to text file"""
    
    filename = os.path.join(output_dir, f'{model_name}_diagnostics.txt')
    
    print(f"\nSaving {model_name} diagnostics...")
    print(f"Trace varnames: {list(trace.varnames)}")
    
    with open(filename, 'w') as f:
        f.write(f"WASP-43b {model_name.upper()} MODEL RESULTS\n")
        f.write("="*50 + "\n\n")
        
        # Fitted parameters
        f.write("FITTED PARAMETERS:\n")
        f.write("-"*30 + "\n")
        
        param_count = 0
        for param in trace.varnames:
            print(f"Checking parameter: {param}")
            # Only exclude flux and map, include everything else
            if param in trace and param not in ['flux', 'map']:
                samples = trace[param]
                mean_val = np.mean(samples)
                std_val = np.std(samples)
                f.write(f"{param}: {mean_val:.6f} ± {std_val:.6f}\n")
                print(f"  {param}: {mean_val:.6f} ± {std_val:.6f}")
                param_count += 1
        
        print(f"Found {param_count} parameters")
        
        # Model fit
        if 'flux' in trace:
            model_flux = np.mean(trace['flux'], axis=0)
            error_inflation = np.mean(trace['error_inflation'])
            chi2 = np.sum((flux - model_flux)**2 / (flux_err * error_inflation)**2)
            f.write(f"\nChi-squared: {chi2:.2f}\n")
            f.write(f"Reduced chi-squared: {chi2/len(flux):.2f}\n")
            f.write(f"Error inflation: {error_inflation:.3f}\n")
            print(f"Chi-squared: {chi2:.2f}")
            print(f"Reduced chi-squared: {chi2/len(flux):.2f}")
            print(f"Error inflation: {error_inflation:.3f}")
        
    print(f"Saved {model_name} results to {filename}")

# Save diagnostics for both models
save_model_diagnostics(map_trace, "map", params, time, flux, flux_err, OUTPUT_DIR)
save_model_diagnostics(pc_trace, "phase_curve", params, time, flux, flux_err, OUTPUT_DIR)

# Compare models
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Calculate chi-squared for both models
map_mean_model = np.mean(map_trace["flux"], axis=0)
pc_mean_model = np.mean(pc_trace["flux"], axis=0)

map_error_inflation = np.mean(map_trace['error_inflation'])
pc_error_inflation = np.mean(pc_trace['error_inflation'])

map_chi2 = np.sum((flux - map_mean_model)**2 / (flux_err * map_error_inflation)**2)
pc_chi2 = np.sum((flux - pc_mean_model)**2 / (flux_err * pc_error_inflation)**2)

print(f"Map Model:")
print(f"  Error inflation: {map_error_inflation:.3f}")
print(f"  Reduced chi-squared: {map_chi2/len(flux):.2f}")

print(f"\nPhase Curve Model:")
print(f"  Error inflation: {pc_error_inflation:.3f}")
print(f"  Reduced chi-squared: {pc_chi2/len(flux):.2f}")

# Save comparison summary
comparison_file = os.path.join(OUTPUT_DIR, 'model_comparison_summary.txt')
with open(comparison_file, 'w') as f:
    f.write("WASP-43b MODEL COMPARISON SUMMARY\n")
    f.write("="*40 + "\n\n")
    f.write(f"Map Model:\n")
    f.write(f"  Error inflation: {map_error_inflation:.3f}\n")
    f.write(f"  Reduced chi-squared: {map_chi2/len(flux):.2f}\n")
    f.write(f"  Planet amplitude: {np.mean(map_trace['planet_amp']):.6f} ± {np.std(map_trace['planet_amp']):.6f}\n\n")
    f.write(f"Phase Curve Model:\n")
    f.write(f"  Error inflation: {pc_error_inflation:.3f}\n")
    f.write(f"  Reduced chi-squared: {pc_chi2/len(flux):.2f}\n")
    f.write(f"  Planet amplitude: {np.mean(pc_trace['planet_amp']):.6f} ± {np.std(pc_trace['planet_amp']):.6f}\n\n")
    
    # Determine which model is better
    if map_chi2 < pc_chi2:
        f.write("CONCLUSION: Map model fits better (lower chi-squared)\n")
    else:
        f.write("CONCLUSION: Phase curve model fits better (lower chi-squared)\n")

print(f"Saved model comparison summary to {comparison_file}")

# Plot comparison - simple version
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top panel: Both models
t_plot = time - time[0]
ax1.errorbar(t_plot, flux, yerr=flux_err * map_error_inflation, fmt='k.', alpha=0.3, ms=2, label='Data')
ax1.plot(t_plot, map_mean_model, 'C0-', linewidth=2, label='Map model')
ax1.plot(t_plot, pc_mean_model, 'C1-', linewidth=2, label='Phase curve model')
ax1.fill_between(t_plot, np.quantile(map_trace["flux"], 0.16, axis=0), np.quantile(map_trace["flux"], 0.84, axis=0), color='C0', alpha=0.2)
ax1.fill_between(t_plot, np.quantile(pc_trace["flux"], 0.16, axis=0), np.quantile(pc_trace["flux"], 0.84, axis=0), color='C1', alpha=0.2)

# Add systematic trends if they were fitted
if USE_POLYNOMIAL or USE_RAMP:
    t_norm = time - time[0]  # Time in days
    t_norm_seconds = t_norm * 86400  # Convert to seconds for ramp calculation
    
    systematic_total = np.ones_like(t_norm)
    
    if USE_POLYNOMIAL:
        c0_fit = np.mean(map_trace["c0"])
        c1_fit = np.mean(map_trace["c1"])
        poly_baseline = 1.0 + c0_fit + c1_fit * t_norm
        systematic_total *= poly_baseline
        ax1.plot(t_plot, poly_baseline, 'g--', alpha=0.7, linewidth=1.5, label=f'Polynomial (c0={c0_fit:.4f}, c1={c1_fit:.4f})')
    
    if USE_RAMP:
        r0_fit = np.mean(map_trace["r0"])
        r1_fit = np.mean(map_trace["r1"])
        ramp = 1.0 + r0_fit * np.exp(-t_norm_seconds / (r1_fit * 86400))
        systematic_total *= ramp
        ax1.plot(t_plot, ramp, 'm--', alpha=0.7, linewidth=1.5, label=f'Ramp (r0={r0_fit:.4f}, r1={r1_fit:.2f})')
    
    # Plot total systematic effect
    ax1.plot(t_plot, systematic_total, 'r--', alpha=0.8, linewidth=2, label='Total systematic')

ax1.axvline(x=eclipse_time - time[0], color='b', linestyle='--', alpha=0.5, label='Eclipse')
ax1.set_ylabel("Relative Flux", fontsize=12)
ax1.set_title("WASP-43b: Map vs Phase Curve Models")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom panel: Residuals
map_residuals = flux - map_mean_model
pc_residuals = flux - pc_mean_model

ax2.errorbar(t_plot, map_residuals, yerr=flux_err * map_error_inflation, fmt='C0.', alpha=0.5, ms=2, label='Map residuals')
ax2.errorbar(t_plot, pc_residuals, yerr=flux_err * pc_error_inflation, fmt='C1.', alpha=0.5, ms=2, label='PC residuals')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=eclipse_time - time[0], color='b', linestyle='--', alpha=0.5)
ax2.set_xlabel("Time [days from start]", fontsize=12)
ax2.set_ylabel("Residuals", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_model_comparison.png'))
plt.close()

# Plot maps if available
if 'map' in map_trace.varnames:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean map
    planet_mu = np.mean(map_trace["map"], axis=0)
    im0 = ax[0].imshow(planet_mu, origin="lower", extent=(-180, 180, -90, 90), cmap="plasma")
    plt.colorbar(im0, ax=ax[0])
    ax[0].set_title("Mean Temperature Map")
    ax[0].set_xlabel("Longitude [degrees]")
    ax[0].set_ylabel("Latitude [degrees]")
    
    # Uncertainty map
    planet_std = np.std(map_trace["map"], axis=0)
    im1 = ax[1].imshow(planet_std, origin="lower", extent=(-180, 180, -90, 90), cmap="plasma")
    plt.colorbar(im1, ax=ax[1])
    ax[1].set_title("Uncertainty Map")
    ax[1].set_xlabel("Longitude [degrees]")
    ax[1].set_ylabel("Latitude [degrees]")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_temperature_maps.png'))
    plt.close()


# Plot map posterior along equator and substellar meridian
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Get the map dimensions
n_lat, n_lon = map_trace["map"].shape[1:]

# Create longitude and latitude arrays matching map dimensions
lon = np.linspace(-180, 180, n_lon)
lat = np.linspace(-90, 90, n_lat)

# Get the map values along the equator (lat=0)
equator_idx = n_lat // 2  # Index corresponding to lat=0
equator_maps = map_trace["map"][:, equator_idx, :]
equator_mean = np.mean(equator_maps, axis=0)
equator_std = np.std(equator_maps, axis=0)

# Get the map values along the substellar meridian (lon=0)
substellar_idx = n_lon // 2  # Index corresponding to lon=0
substellar_maps = map_trace["map"][:, :, substellar_idx]
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
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_map_profiles.png'))
plt.close()

# Plot residuals: Map model - Phase curve model
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

t_plot = time - time[0]

# Calculate residuals
data_minus_pc = flux - pc_mean_model
map_minus_pc = map_mean_model - pc_mean_model

# Get 1-sigma range for map model
map_16 = np.quantile(map_trace["flux"], 0.16, axis=0)
map_84 = np.quantile(map_trace["flux"], 0.84, axis=0)
map_minus_pc_16 = map_16 - pc_mean_model
map_minus_pc_84 = map_84 - pc_mean_model

# Plot data residuals
ax.errorbar(t_plot, data_minus_pc, yerr=flux_err * map_error_inflation, 
           fmt='k.', alpha=0.5, ms=3, label='Data - Phase curve model')

# Plot map model residuals with uncertainty
ax.fill_between(t_plot, map_minus_pc_16, map_minus_pc_84, 
               color='C0', alpha=0.3, label='Map - Phase curve model (1σ)')
ax.plot(t_plot, map_minus_pc, 'C0-', linewidth=2, label='Map - Phase curve model (mean)')

# Add eclipse line
ax.axvline(x=eclipse_time - time[0], color='b', linestyle='--', alpha=0.7, label='Eclipse')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

ax.set_xlabel("Time [days from start]", fontsize=12)
ax.set_ylabel("Residual Flux", fontsize=12)
ax.set_title("WASP-43b: Map Model Residuals Relative to Phase Curve Model")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wasp43b_map_residuals.png'))
plt.close()

print(f"\nAnalysis complete! Results saved in {OUTPUT_DIR}/")
print("Generated files:")
print("- wasp43b_raw_data.png")
print("- wasp43b_model_comparison.png")
print("- wasp43b_temperature_maps.png")
print("- wasp43b_map_residuals.png")
print("- map_diagnostics.txt")
print("- phase_curve_diagnostics.txt")
print("- model_comparison_summary.txt")
