#!/usr/bin/env python3
"""
Generic Light Curve Mapper

A tool for fitting exoplanet light curves with spherical harmonic maps.
This version works without starry to test basic functionality on ARM Macs.
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3_ext as pmx
import pickle
import argparse
import os
from pathlib import Path
import corner
import arviz as az

# Set random seed for reproducibility
np.random.seed(42)

# Enable lazy evaluation for PyMC3
pm.config.lazy = True

class LightCurveMapper:
    """Generic light curve mapper for exoplanet data."""
    
    def __init__(self, map_degree=2, fourier_degree=2, fit_orbital=False):
        """
        Initialize the light curve mapper.
        
        Parameters:
        -----------
        map_degree : int
            Degree of spherical harmonic map (default: 2)
        fourier_degree : int
            Degree of Fourier series for phase curve (default: 2)
        fit_orbital : bool
            Whether to fit orbital parameters (default: False)
        """
        self.map_degree = map_degree
        self.fourier_degree = fourier_degree
        self.fit_orbital = fit_orbital
        
        # MCMC parameters
        self.n_samples = 1000
        self.n_tune = 1000
        self.n_chains = 2
        self.n_cores = 2
        self.target_accept = 0.9
        
    def load_pickle_data(self, filepath):
        """Load data from pickle file."""
        print(f"Loading data from: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Available keys: {list(data.keys())}")
            return data
        else:
            print(f"Data is not a dictionary: {type(data)}")
            return None
    
    def extract_light_curve_data(self, data):
        """
        Extract light curve data from the loaded pickle file.
        This is a placeholder - you'll need to adapt this based on your data structure.
        """
        # This is a placeholder - you'll need to implement based on your data structure
        if isinstance(data, dict):
            # Try to find time, flux, and error data
            possible_time_keys = ['time', 't', 'jd', 'bjd']
            possible_flux_keys = ['flux', 'f', 'relative_flux', 'normalized_flux']
            possible_error_keys = ['error', 'err', 'sigma', 'flux_err']
            
            time = None
            flux = None
            flux_err = None
            
            # Find time data
            for key in possible_time_keys:
                if key in data:
                    time = data[key]
                    print(f"Found time data in key: {key}")
                    break
            
            # Find flux data
            for key in possible_flux_keys:
                if key in data:
                    flux = data[key]
                    print(f"Found flux data in key: {key}")
                    break
            
            # Find error data
            for key in possible_error_keys:
                if key in data:
                    flux_err = data[key]
                    print(f"Found error data in key: {key}")
                    break
            
            if time is None or flux is None:
                print("Could not find time or flux data in expected keys")
                return None, None, None
            
            # Convert to numpy arrays
            time = np.asarray(time, dtype=np.float64)
            flux = np.asarray(flux, dtype=np.float64)
            
            if flux_err is not None:
                flux_err = np.asarray(flux_err, dtype=np.float64)
            else:
                # Create dummy errors if not provided
                flux_err = np.ones_like(flux) * 0.001
                print("No error data found, using dummy errors")
            
            return time, flux, flux_err
        
        return None, None, None
    
    def fit_simple_model(self, time, flux, flux_err, output_dir="output"):
        """
        Fit a simple model to the light curve data.
        This is a placeholder that doesn't use starry.
        """
        print("Fitting simple model to light curve data...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Simple polynomial fit as a placeholder
        with pm.Model() as model:
            # Polynomial coefficients
            c0 = pm.Normal("c0", mu=1.0, sd=0.1)
            c1 = pm.Normal("c1", mu=0.0, sd=0.1)
            c2 = pm.Normal("c2", mu=0.0, sd=0.1)
            
            # Time normalized to [0, 1]
            t_norm = (time - time.min()) / (time.max() - time.min())
            
            # Simple polynomial model
            flux_model = c0 + c1 * t_norm + c2 * t_norm**2
            
            # Error inflation
            error_inflation = pm.HalfNormal("error_inflation", sd=1.0)
            
            # Likelihood
            pm.Normal("obs", mu=flux_model, sd=flux_err * error_inflation, observed=flux)
        
        # Sample
        with model:
            trace = pmx.sample(
                tune=self.n_tune,
                draws=self.n_samples,
                target_accept=self.target_accept,
                return_inferencedata=False,
                cores=self.n_cores,
                chains=self.n_chains
            )
        
        # Plot results
        self.plot_results(time, flux, flux_err, trace, output_dir)
        
        return trace
    
    def plot_results(self, time, flux, flux_err, trace, output_dir):
        """Plot the fitting results."""
        print("Creating plots...")
        
        # Light curve fit
        plt.figure(figsize=(12, 6))
        
        # Data
        plt.errorbar(time, flux, yerr=flux_err, fmt='k.', alpha=0.3, ms=2, label='Data')
        
        # Model
        t_norm = (time - time.min()) / (time.max() - time.min())
        c0_mean = np.mean(trace['c0'])
        c1_mean = np.mean(trace['c1'])
        c2_mean = np.mean(trace['c2'])
        
        flux_model = c0_mean + c1_mean * t_norm + c2_mean * t_norm**2
        plt.plot(time, flux_model, 'r-', linewidth=2, label='Model')
        
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Light Curve Fit")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/light_curve_fit.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Corner plot
        samples = np.vstack([trace['c0'], trace['c1'], trace['c2'], trace['error_inflation']]).T
        labels = ['c0', 'c1', 'c2', 'error_inflation']
        
        fig = corner.corner(samples, labels=labels, show_titles=True)
        plt.savefig(f"{output_dir}/corner_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def run_analysis(self, data_file, output_dir="output"):
        """Run the complete analysis pipeline."""
        print(f"Starting analysis of: {data_file}")
        
        # Load data
        data = self.load_pickle_data(data_file)
        if data is None:
            print("Failed to load data")
            return
        
        # Extract light curve data
        time, flux, flux_err = self.extract_light_curve_data(data)
        if time is None:
            print("Failed to extract light curve data")
            return
        
        print(f"Data shape: time={time.shape}, flux={flux.shape}, error={flux_err.shape}")
        print(f"Time range: {time.min():.3f} to {time.max():.3f}")
        print(f"Flux range: {flux.min():.6f} to {flux.max():.6f}")
        
        # Fit model
        trace = self.fit_simple_model(time, flux, flux_err, output_dir)
        
        print("Analysis complete!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generic Light Curve Mapper")
    parser.add_argument("data_file", help="Path to pickle data file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--map-degree", type=int, default=2, help="Map degree")
    parser.add_argument("--fourier-degree", type=int, default=2, help="Fourier degree")
    parser.add_argument("--fit-orbital", action="store_true", help="Fit orbital parameters")
    
    args = parser.parse_args()
    
    # Create mapper
    mapper = LightCurveMapper(
        map_degree=args.map_degree,
        fourier_degree=args.fourier_degree,
        fit_orbital=args.fit_orbital
    )
    
    # Run analysis
    mapper.run_analysis(args.data_file, args.output)

if __name__ == "__main__":
    main()
