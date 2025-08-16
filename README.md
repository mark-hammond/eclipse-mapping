# Exoplanet Light Curve Mapping

Bayesian inference for exoplanet surface mapping using spherical harmonics with `starry` and PyMC3.

## Quick Start

### Docker Setup

1. **Install Docker Desktop** from https://www.docker.com/products/docker-desktop

2. **Build the image** (5-10 minutes):
```bash
docker build --platform linux/amd64 -t exoplanet-mapper .
```

3. **Run the container**:
```bash
docker run -it --name exoplanet_container -v $(pwd):/app exoplanet-mapper
```

4. **Verify installation**:
```bash
python3 -c 'import numpy, starry, pymc3; print("Success!")'
```

### Run Analysis

```bash
python3 fit_wasp43b.py
```

Results are saved in `wasp43b_results/` directory.

## Script Usage

`fit_wasp43b.py` performs two types of fits:

1. **Spherical Harmonic Map**: Recovers 2D temperature distribution
2. **Phase Curve Model**: Fits temporal variations with Fourier series

### Key Parameters

```python
MAP_DEGREE = 2        # Spherical harmonic degree
FOURIER_DEGREE = 2    # Fourier series degree
N_SAMPLES = 300       # MCMC samples
USE_POLYNOMIAL = False # Fit polynomial baseline
USE_RAMP = False      # Fit exponential ramp
```

### Output Files

- `wasp43b_raw_data.png` - Raw light curve
- `wasp43b_fit_map.png` - Map model fit
- `wasp43b_fit_pc.png` - Phase curve model fit
- `wasp43b_model_comparison.png` - Both models compared
- `wasp43b_map_residuals.png` - Map minus phase curve residuals
- `wasp43b_temperature_maps.png` - 2D temperature maps
- `map_diagnostics.txt` - Fitted parameters
- `phase_curve_diagnostics.txt` - Fitted parameters

## Adapting for Other Planets

### 1. Data Format
Replace the data loading section:
```python
# Current: WASP-43b numpy files
time = np.load('w43b_time.npy')
flux = np.load('w43b_flux.npy')
flux_err = np.load('w43b_error.npy')

# For other data:
time = your_time_data
flux = your_flux_data  
flux_err = your_error_data
```

### 2. System Parameters
Update the `params` dictionary:
```python
params = {
    'fpfs': 0.005,        # Planet-to-star flux ratio
    'per': 0.813,         # Orbital period (days)
    't0': 55934.292,      # Transit time (BMJD)
    'a': 4.859,           # Semi-major axis (R*)
    'Rs': 1.0,            # Stellar radius (reference unit)
    'Rp': 0.158,          # Planet radius (R*)
    'inc': 82.106,        # Inclination (degrees)
    'u1': 0.0182,         # Linear limb darkening
    'u2': 0.595           # Quadratic limb darkening
}
```

### 3. Model Configuration
Adjust model complexity:
```python
MAP_DEGREE = 1        # Start with l=1, increase if needed
FOURIER_DEGREE = 1    # Start with 1st order, increase if needed
USE_POLYNOMIAL = True # Enable if data shows trends
USE_RAMP = True       # Enable if data shows ramps
```

### 4. Data Preprocessing
Modify preprocessing as needed:
```python
BIN_SIZE = 10         # Bin data if too noisy
NORMALIZE = True      # Normalize to mean=1.0 if needed
```

## Container Management

```bash
# Re-enter container
docker start exoplanet_container
docker exec -it exoplanet_container /bin/bash

# Exit container (keeps it running)
exit

# Stop and remove container
docker stop exoplanet_container
docker rm exoplanet_container

```
