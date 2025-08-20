# -*- coding: utf-8 -*-

import starry
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import os

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for starry
starry.config.lazy = False
starry.config.quiet = True

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Map parameters
MAP_DEGREE = 2        # Degree of spherical harmonic map
MAP_TYPE = "hotspot"  # Type of map: "hotspot", "uniform", "custom", or "coefficients"

# Phase coverage parameters
START_PHASE = -0.5    # Start phase relative to eclipse (0 = eclipse center)
END_PHASE = 1.5       # End phase relative to eclipse (1.5 - (-0.5) = 2 orbital periods)

# Time sampling parameters
CADENCE_MINUTES = 1.5  # Observing cadence in minutes (20x faster than original 30 min)
TIME_STEP_MINUTES = CADENCE_MINUTES  # Time step for simulation

# Noise parameters
PHOTOMETRIC_PRECISION_PPM = 100  # Photometric precision in parts per million
ADD_SYSTEMATICS = True           # Whether to add systematic effects
SYSTEMATIC_AMPLITUDE = 0.001     # Amplitude of systematic effects

# Map specification (used if MAP_TYPE = "hotspot")
HOTSPOT_LONGITUDE = 30    # Hotspot longitude offset from substellar point (degrees)
DAY_NIGHT_CONTRAST = 0.8  # Contrast between day and night sides
HOTSPOT_WIDTH = 60        # Width of hotspot in degrees

# Custom spherical harmonic coefficients (used if MAP_TYPE = "coefficients")
# Format: [Y_1,-1, Y_1,0, Y_1,1, Y_2,-2, Y_2,-1, Y_2,0, Y_2,1, Y_2,2, ...]
CUSTOM_YLM_COEFFS = [0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example coefficients

# Output parameters
OUTPUT_DIR = "wasp43b_simulation"
SAVE_NUMPY_FILES = True   # Save time, flux, flux_err as .npy files
GENERATE_PLOTS = True     # Generate diagnostic plots

# ============================================================================
# SYSTEM PARAMETERS (same format as fit_wasp43b.py)
# ============================================================================

params = {
    'fpfs': 0.005,        # Planet-to-star flux ratio (realistic eclipse depth)
    'per': 0.8134740621723353,  # Orbital period in days (from literature)
    't0': 55934.292283,   # Transit time (BMJD) from Eclipse Map Fit
    'a': 4.859,           # Semi-major axis in stellar radii from Eclipse Map Fit
    'Rs': 1.0,            # Stellar radius = 1.0 (reference unit)
    'Rp': 0.15839,        # Planet radius in stellar radii from Eclipse Map Fit
    'inc': 82.106,        # Orbital inclination in degrees from Eclipse Map Fit
    'c0': 0.0,            # Constant baseline deviation from 1.0
    'c1': 0.0,            # Linear trend deviation from 1.0
    'r0': 0.001,          # Ramp magnitude deviation from 1.0
    'r1': 3.7,            # Ramp time constant (1/day) from Eclipse Map Fit
    'u1': 0.0182,         # Limb darkening parameter q1 from Eclipse Map Fit
    'u2': 0.595           # Limb darkening parameter q2 from Eclipse Map Fit
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def create_temperature_map(map_type, map_degree, **kwargs):
    """
    Create a temperature map based on the specified type
    
    Parameters:
    -----------
    map_type : str
        Type of map to create: "hotspot", "uniform", "custom", or "coefficients"
    map_degree : int
        Degree of spherical harmonic expansion
    **kwargs : dict
        Additional parameters for map creation
    
    Returns:
    --------
    ylm_coeffs : array
        Spherical harmonic coefficients
    """
    
    # Create a temporary map to get the coefficient structure
    temp_map = starry.Map(ydeg=map_degree)
    ncoeff = temp_map.Ny - 1  # Exclude Y_0,0 coefficient
    ylm_coeffs = np.zeros(ncoeff)
    
    if map_type == "uniform":
        # Uniform temperature (only Y_0,0 contributes, which is handled by amplitude)
        pass
        
    elif map_type == "hotspot":
        # Create a simple hotspot model
        hotspot_lon = kwargs.get('hotspot_longitude', 30)  # degrees
        day_night = kwargs.get('day_night_contrast', 0.8)
        
        # Simple hotspot: mainly Y_1,1 and Y_1,0 terms
        if map_degree >= 1:
            # Y_1,0 term (day-night contrast)
            ylm_coeffs[1] = day_night * 0.3  # Index 1 corresponds to Y_1,0
            # Y_1,1 term (hotspot offset)
            ylm_coeffs[2] = day_night * 0.2 * np.cos(np.radians(hotspot_lon))  # Y_1,1
            if ncoeff > 3:
                ylm_coeffs[0] = day_night * 0.2 * np.sin(np.radians(hotspot_lon))  # Y_1,-1
        
    elif map_type == "coefficients":
        # Use custom coefficients
        custom_coeffs = kwargs.get('custom_ylm_coeffs', [])
        n_custom = min(len(custom_coeffs), ncoeff)
        ylm_coeffs[:n_custom] = custom_coeffs[:n_custom]
        
    elif map_type == "custom":
        # For custom 2D maps (not implemented in this streamlined version)
        print("Custom 2D map input not implemented. Using uniform map.")
        
    return ylm_coeffs

def generate_time_array(params, start_phase, end_phase, cadence_minutes):
    """
    Generate time array for simulation
    
    Parameters:
    -----------
    params : dict
        System parameters
    start_phase : float
        Starting phase relative to eclipse
    end_phase : float
        Ending phase relative to eclipse  
    cadence_minutes : float
        Observing cadence in minutes
        
    Returns:
    --------
    time : array
        Time array in same units as params['t0']
    """
    
    # Calculate eclipse time
    eclipse_time = params['t0'] + params['per']/2
    
    # Convert cadence to days
    cadence_days = cadence_minutes / (24 * 60)
    
    # Total duration based on phase coverage
    phase_duration = end_phase - start_phase  # Phase range (in units of orbital periods)
    total_duration = phase_duration * params['per']  # Convert to days
    
    # Generate time array
    n_points = int(total_duration / cadence_days)
    time = np.linspace(0, total_duration, n_points)
    
    # Shift to start at the correct phase relative to first eclipse
    time_start = eclipse_time + start_phase * params['per']
    time = time + time_start
    
    return time

def simulate_light_curve(time, params, ylm_coeffs, map_degree, add_systematics=True):
    """
    Simulate a light curve using starry
    
    Parameters:
    -----------
    time : array
        Time array
    params : dict
        System parameters
    ylm_coeffs : array
        Spherical harmonic coefficients
    map_degree : int
        Degree of spherical harmonic map
    add_systematics : bool
        Whether to add systematic effects
        
    Returns:
    --------
    flux_clean : array
        Clean astrophysical flux
    flux_total : array
        Total flux including systematics
    """
    
    # Create a star with quadratic limb darkening
    star = starry.Primary(
        starry.Map(ydeg=0, udeg=2, amp=1.0),
        m=0,
        r=params['Rs'],
        prot=1.0
    )
    
    # Set limb darkening coefficients
    star.map[1] = params['u1']
    star.map[2] = params['u2']
    
    # Calculate system mass
    a_physical = params['a'] * params['Rs'] * const.R_sun.value
    p_physical = params['per'] * 24 * 3600
    system_mass = ((2 * np.pi * a_physical**(3/2)) / p_physical)**2 / const.G.value / const.M_sun.value
    
    # Create planet with temperature map
    planet = starry.Secondary(
        starry.Map(ydeg=map_degree, udeg=0, amp=params['fpfs']),
        m=system_mass,
        r=params['Rp'],
        inc=params['inc'],
        a=params['a'],
        t0=params['t0']
    )
    
    # Set orbital parameters
    planet.porb = params['per']
    planet.prot = params['per']  # Tidally locked
    planet.theta0 = 180.0        # Start with night side facing observer
    
    # Set the spherical harmonic coefficients
    if len(ylm_coeffs) > 0:
        planet.map[1:, :] = ylm_coeffs
    
    # Create system and compute flux
    system = starry.System(star, planet)
    star_flux, planet_flux = system.flux(time, total=False)
    flux_clean = star_flux + planet_flux
    
    # Add systematic effects if requested
    if add_systematics:
        t_norm = (time - time[0]) * 86400  # Time in seconds from start
        
        # Polynomial baseline
        poly_baseline = 1.0 + params['c0'] + params['c1'] * (time - time[0])
        
        # Exponential ramp
        ramp = 1.0 + params['r0'] * np.exp(-t_norm / (params['r1'] * 86400))
        
        flux_total = flux_clean * poly_baseline * ramp
    else:
        flux_total = flux_clean.copy()
    
    return flux_clean, flux_total

def add_noise(flux, precision_ppm):
    """
    Add Gaussian noise to flux
    
    Parameters:
    -----------
    flux : array
        Input flux
    precision_ppm : float
        Photometric precision in parts per million
        
    Returns:
    --------
    flux_noisy : array
        Flux with added noise
    flux_err : array
        Flux uncertainties
    """
    
    # Convert precision to fractional uncertainty
    fractional_precision = precision_ppm * 1e-6
    
    # Calculate flux errors (assume constant fractional precision)
    flux_err = flux * fractional_precision
    
    # Add Gaussian noise
    noise = np.random.normal(0, flux_err)
    flux_noisy = flux + noise
    
    return flux_noisy, flux_err

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def main():
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    print("WASP-43b Light Curve Simulation")
    print("=" * 50)
    
    # Print simulation parameters
    print(f"Map type: {MAP_TYPE}")
    print(f"Map degree: {MAP_DEGREE}")
    print(f"Phase coverage: {START_PHASE} to {END_PHASE} ({END_PHASE - START_PHASE:.1f} orbital periods)")
    print(f"Cadence: {CADENCE_MINUTES} minutes")
    print(f"Photometric precision: {PHOTOMETRIC_PRECISION_PPM} ppm")
    print(f"Add systematics: {ADD_SYSTEMATICS}")
    
    # Create temperature map
    print(f"\nCreating {MAP_TYPE} temperature map...")
    ylm_coeffs = create_temperature_map(
        MAP_TYPE, MAP_DEGREE,
        hotspot_longitude=HOTSPOT_LONGITUDE,
        day_night_contrast=DAY_NIGHT_CONTRAST,
        custom_ylm_coeffs=CUSTOM_YLM_COEFFS
    )
    
    print(f"Spherical harmonic coefficients: {ylm_coeffs}")
    
    # Generate time array
    print(f"\nGenerating time array...")
    time = generate_time_array(params, START_PHASE, END_PHASE, CADENCE_MINUTES)
    print(f"Generated {len(time)} time points")
    print(f"Time range: {time[0]:.6f} to {time[-1]:.6f}")
    
    # Simulate light curve
    print(f"\nSimulating light curve...")
    flux_clean, flux_total = simulate_light_curve(
        time, params, ylm_coeffs, MAP_DEGREE, ADD_SYSTEMATICS
    )
    
    # Add noise
    print(f"Adding noise...")
    flux_noisy, flux_err = add_noise(flux_total, PHOTOMETRIC_PRECISION_PPM)
    
    print(f"Simulation complete!")
    print(f"Clean flux range: {np.min(flux_clean):.6f} to {np.max(flux_clean):.6f}")
    print(f"Noisy flux range: {np.min(flux_noisy):.6f} to {np.max(flux_noisy):.6f}")
    
    # Save data
    if SAVE_NUMPY_FILES:
        print(f"\nSaving data to {OUTPUT_DIR}/...")
        # Save in format compatible with fit_wasp43b.py
        np.save(os.path.join(OUTPUT_DIR, 'w43b_time.npy'), time)
        np.save(os.path.join(OUTPUT_DIR, 'w43b_flux.npy'), flux_noisy)
        np.save(os.path.join(OUTPUT_DIR, 'w43b_error.npy'), flux_err)
        # Also save additional simulation data
        np.save(os.path.join(OUTPUT_DIR, 'sim_flux_clean.npy'), flux_clean)
        np.save(os.path.join(OUTPUT_DIR, 'sim_flux_total.npy'), flux_total)
        np.save(os.path.join(OUTPUT_DIR, 'sim_ylm_coeffs.npy'), ylm_coeffs)
        print("Saved: w43b_time.npy, w43b_flux.npy, w43b_error.npy (fit_wasp43b.py compatible)")
        print("       sim_flux_clean.npy, sim_flux_total.npy, sim_ylm_coeffs.npy (simulation extras)")
    
    # Generate plots
    if GENERATE_PLOTS:
        print(f"\nGenerating plots...")
        
        # Calculate eclipse times for plotting
        eclipse_time = params['t0'] + params['per']/2
        
        # Plot 1: Full light curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        t_plot = time - time[0]
        
        # Top panel: Light curves
        ax1.plot(t_plot, flux_clean, 'b-', linewidth=2, label='Clean astrophysical', alpha=0.8)
        if ADD_SYSTEMATICS:
            ax1.plot(t_plot, flux_total, 'g-', linewidth=2, label='With systematics', alpha=0.8)
        ax1.errorbar(t_plot, flux_noisy, yerr=flux_err, fmt='k.', alpha=0.5, ms=1, 
                    label=f'Observed ({PHOTOMETRIC_PRECISION_PPM} ppm)')
        
        # Mark eclipses
        n_eclipses = int(np.ceil(END_PHASE - START_PHASE)) + 1
        for i in range(n_eclipses):
            eclipse_t = eclipse_time + i * params['per'] - time[0]
            if eclipse_t >= t_plot[0] and eclipse_t <= t_plot[-1]:
                ax1.axvline(x=eclipse_t, color='r', linestyle='--', alpha=0.5)
        
        ax1.set_ylabel("Relative Flux", fontsize=12)
        ax1.set_title(f"WASP-43b Simulated Light Curve ({MAP_TYPE} map)")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Residuals
        residuals = flux_noisy - flux_clean
        ax2.errorbar(t_plot, residuals, yerr=flux_err, fmt='k.', alpha=0.5, ms=1)
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        
        # Mark eclipses
        n_eclipses = int(np.ceil(END_PHASE - START_PHASE)) + 1
        for i in range(n_eclipses):
            eclipse_t = eclipse_time + i * params['per'] - time[0]
            if eclipse_t >= t_plot[0] and eclipse_t <= t_plot[-1]:
                ax2.axvline(x=eclipse_t, color='r', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel("Time [days from start]", fontsize=12)
        ax2.set_ylabel("Residuals", fontsize=12)
        ax2.set_title("Observed - Clean Model")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'simulated_light_curve.png'), dpi=150)
        plt.close()
        
        # Plot 2: Temperature map (if map degree > 0)
        if MAP_DEGREE > 0 and np.any(ylm_coeffs != 0):
            # Create a map object to render the temperature map
            temp_map = starry.Map(ydeg=MAP_DEGREE)
            temp_map[1:, :] = ylm_coeffs
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            rendered_map = temp_map.render(projection="rect")
            im = ax.imshow(rendered_map, origin="lower", extent=(-180, 180, -90, 90), 
                          cmap="plasma", aspect='auto')
            plt.colorbar(im, ax=ax, label='Relative Temperature')
            ax.set_xlabel("Longitude [degrees]", fontsize=12)
            ax.set_ylabel("Latitude [degrees]", fontsize=12)
            ax.set_title(f"Input Temperature Map ({MAP_TYPE})")
            
            # Mark substellar point
            ax.plot(0, 0, 'w*', markersize=15, label='Substellar point')
            if MAP_TYPE == "hotspot":
                ax.plot(HOTSPOT_LONGITUDE, 0, 'r*', markersize=15, label='Hotspot')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'input_temperature_map.png'), dpi=150)
            plt.close()
        
        print("Generated: simulated_light_curve.png")
        if MAP_DEGREE > 0:
            print("           input_temperature_map.png")
    
    print(f"\nSimulation complete! Results saved in {OUTPUT_DIR}/")
    print(f"\nTo fit this simulated data with fit_wasp43b.py:")
    print(f"1. Copy the files w43b_time.npy, w43b_flux.npy, w43b_error.npy to the main directory")
    print(f"2. Or modify fit_wasp43b.py to load from '{OUTPUT_DIR}/' instead")
    print(f"3. The true input map coefficients are saved in sim_ylm_coeffs.npy for comparison")

if __name__ == "__main__":
    main()
