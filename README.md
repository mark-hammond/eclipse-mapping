# Exoplanet Light Curve Mapping

This repository contains code for analyzing exoplanet light curves using spherical harmonic mapping with the `starry` package. The project focuses on eclipse mapping to recover temperature distributions on exoplanet surfaces.

## Overview

The code performs Bayesian inference using PyMC3 to fit spherical harmonic maps to exoplanet eclipse light curves. This allows us to:

- Recover 2D temperature maps of exoplanet surfaces
- Fit orbital and systematic parameters
- Generate corner plots and uncertainty maps
- Compare map models with simpler phase curve models

## Current Analysis

### WASP-43b Eclipse Mapping
- **Data**: MIRI observations from JWST
- **Model**: l=2 spherical harmonic map
- **Parameters**: Using published "Eclipse Map Fit" parameters from literature
- **Systematics**: Polynomial baseline and exponential ramp included

## Files

### Core Analysis Scripts
- `fit_wasp43b.py` - Main analysis script for WASP-43b
- `manual_fit_wasp76b.py` - Original WASP-76b analysis (reference)
- `exoplanet_lightcurve.py` - Example light curve analysis

### Data Processing
- `extract_pickle_data.py` - Extract data from pickle files
- `inspect_pickle_files.py` - Inspect pickle file structure

### Configuration
- `Dockerfile` - Docker environment setup
- `requirements.txt` - Python dependencies
- `docker_installation_instructions.txt` - Setup guide

## Setup

### Using Docker (Recommended)
```bash
# Build the Docker image
docker build --platform linux/amd64 -t exoplanet-mapper .

# Run the container
docker run -it --name exoplanet_container -v $(pwd):/app exoplanet-mapper

# Inside container, run analysis
python3 fit_wasp43b.py
```

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python fit_wasp43b.py
```

## Output

The analysis generates several output files in the `wasp43b_results/` directory:

- `wasp43b_raw_data.png` - Raw light curve data
- `corner_plot_wasp43b_map.png` - Parameter correlation plot
- `wasp43b_fit_map.png` - Light curve fit with map model
- `wasp43b_maps.png` - Mean and uncertainty temperature maps
- `wasp43b_map_profiles.png` - Temperature profiles along equator and meridian
- `wasp43b_map_params.txt` - Fitted parameters and uncertainties

## Key Parameters

### WASP-43b System Parameters (Eclipse Map Fit)
- **Period**: 0.813474 days
- **Transit Time**: 55934.292283 BMJD
- **Semi-major Axis**: 4.859 R*
- **Planet Radius**: 0.15839 R*
- **Inclination**: 82.106°
- **Limb Darkening**: u1=0.0182, u2=0.595

### Systematic Parameters
- **Constant Baseline**: -2881 ppm
- **Linear Trend**: -240 ppm/day
- **Ramp Magnitude**: 1319 ppm
- **Ramp Time Constant**: 3.7 day^-1

## Model Details

The analysis uses a two-stage approach:

1. **Map Model**: Fits spherical harmonic coefficients (Ylm) to capture spatial temperature variations
2. **Systematic Model**: Accounts for instrumental effects (baseline, ramp)

The likelihood function includes error inflation to account for underestimated uncertainties.

## Dependencies

- `starry==1.0.0` - Spherical harmonic mapping
- `pymc3==3.9.3` - Bayesian inference
- `numpy==1.20.3` - Numerical computing
- `matplotlib==3.3.4` - Plotting
- `corner==2.2.1` - Corner plots
- `astropy==4.2.1` - Astronomical utilities

## References

This work builds on the eclipse mapping techniques developed in the exoplanet community, particularly using the `starry` package for spherical harmonic modeling.

## License

[Add your license here]
