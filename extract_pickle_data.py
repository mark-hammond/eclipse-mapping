# -*- coding: utf-8 -*-
import pickle
import numpy as np
from pathlib import Path

# Define a minimal class to handle the pickle data
class lc_dataset:
    def __init__(self):
        self.time = None
        self.t = None  # Alternative time attribute
        self.flux = None
        self.error = None
        self.sigma = None  # Alternative error attribute
        self.params = None
        self.system_params = None
        self.data = None
        self.name = None

def extract_pickle_data(filepath):
    """Extract data from pickle file using minimal class definition"""
    print(f"============================================================")
    print(f"Extracting: {filepath}")
    print(f"============================================================")
    
    try:
        # Add our class to the global namespace
        globals()['lc_dataset'] = lc_dataset
        
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        
        print(f"Successfully loaded object of type: {type(obj)}")
        
        # Try to access attributes
        if hasattr(obj, '__dict__'):
            print(f"Object attributes: {list(obj.__dict__.keys())}")
            
            # Extract key data
            for attr in ['time', 't', 'flux', 'error', 'sigma', 'params', 'system_params', 'data', 'name']:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    print(f"\n{attr}:")
                    print(f"  Type: {type(val)}")
                    
                    if isinstance(val, np.ndarray):
                        print(f"  Shape: {val.shape}")
                        print(f"  Dtype: {val.dtype}")
                        if len(val) > 0:
                            print(f"  First 5 values: {val[:5]}")
                            if not np.all(np.isnan(val)):
                                print(f"  Min: {np.nanmin(val):.6f}, Max: {np.nanmax(val):.6f}")
                                print(f"  Number of NaN values: {np.sum(np.isnan(val))}")
                            else:
                                print(f"  All values are NaN!")
                    elif isinstance(val, dict):
                        print(f"  Keys: {list(val.keys())}")
                        for key, subval in val.items():
                            print(f"    {key}: {type(subval)}")
                            if isinstance(subval, (int, float)):
                                print(f"      Value: {subval}")
                            elif isinstance(subval, np.ndarray):
                                print(f"      Shape: {subval.shape}")
                                if len(subval) > 0:
                                    print(f"      First 3 values: {subval[:3]}")
                    elif isinstance(val, (int, float, str)):
                        print(f"  Value: {val}")
        
        # Try to get the actual data arrays
        time_data = None
        if hasattr(obj, 'time') and obj.time is not None:
            time_data = obj.time
        elif hasattr(obj, 't') and obj.t is not None:
            time_data = obj.t
            print(f"\nUsing 't' attribute for time data")
        
        if time_data is not None:
            print(f"Time data shape: {time_data.shape}")
            if not np.all(np.isnan(time_data)):
                print(f"Time range: {np.nanmin(time_data):.6f} to {np.nanmax(time_data):.6f}")
            else:
                print(f"Time data contains all NaN values!")
        
        if hasattr(obj, 'flux') and obj.flux is not None:
            flux_data = obj.flux
            print(f"Flux data shape: {flux_data.shape}")
            if not np.all(np.isnan(flux_data)):
                print(f"Flux range: {np.nanmin(flux_data):.6f} to {np.nanmax(flux_data):.6f}")
                print(f"Number of valid flux points: {np.sum(~np.isnan(flux_data))}")
            else:
                print(f"Flux data contains all NaN values!")
        
        error_data = None
        if hasattr(obj, 'error') and obj.error is not None:
            error_data = obj.error
        elif hasattr(obj, 'sigma') and obj.sigma is not None:
            error_data = obj.sigma
            print(f"\nUsing 'sigma' attribute for error data")
        
        if error_data is not None:
            print(f"Error data shape: {error_data.shape}")
            if not np.all(np.isnan(error_data)):
                print(f"Error range: {np.nanmin(error_data):.6f} to {np.nanmax(error_data):.6f}")
            else:
                print(f"Error data contains all NaN values!")
        
        # Extract system parameters if available
        if hasattr(obj, 'system_params') and obj.system_params is not None:
            sys_params = obj.system_params
            print(f"\nSystem parameters:")
            if isinstance(sys_params, dict):
                for key, value in sys_params.items():
                    print(f"  {key}: {value}")
        
        return obj
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def clean_and_save_data(obj, output_prefix="w43b"):
    """Clean the data and save to numpy files"""
    print(f"\nCleaning and saving data...")
    
    # Get time data
    time_data = None
    if hasattr(obj, 'time') and obj.time is not None:
        time_data = obj.time
    elif hasattr(obj, 't') and obj.t is not None:
        time_data = obj.t
    
    # Get flux data
    flux_data = None
    if hasattr(obj, 'flux') and obj.flux is not None:
        flux_data = obj.flux
    
    # Get error data
    error_data = None
    if hasattr(obj, 'error') and obj.error is not None:
        error_data = obj.error
    elif hasattr(obj, 'sigma') and obj.sigma is not None:
        error_data = obj.sigma
    
    # Clean the data by removing NaN values
    if time_data is not None and flux_data is not None:
        # Create mask for valid data
        valid_mask = ~(np.isnan(time_data) | np.isnan(flux_data))
        if error_data is not None:
            valid_mask &= ~np.isnan(error_data)
        
        print(f"Original data points: {len(time_data)}")
        print(f"Valid data points: {np.sum(valid_mask)}")
        
        # Apply mask
        clean_time = time_data[valid_mask]
        clean_flux = flux_data[valid_mask]
        clean_error = error_data[valid_mask] if error_data is not None else None
        
        # Save cleaned data
        np.save(f'{output_prefix}_time.npy', clean_time)
        print(f"Saved cleaned time data to {output_prefix}_time.npy")
        
        np.save(f'{output_prefix}_flux.npy', clean_flux)
        print(f"Saved cleaned flux data to {output_prefix}_flux.npy")
        
        if clean_error is not None:
            np.save(f'{output_prefix}_error.npy', clean_error)
            print(f"Saved cleaned error data to {output_prefix}_error.npy")
        
        # Save system parameters
        if hasattr(obj, 'system_params') and obj.system_params is not None:
            np.save(f'{output_prefix}_system_params.npy', obj.system_params, allow_pickle=True)
            print(f"Saved system parameters to {output_prefix}_system_params.npy")
        
        return clean_time, clean_flux, clean_error
    
    return None, None, None

def main():
    """Main function to extract data from all pickle files"""
    datasets_dir = Path("datasets")
    
    if not datasets_dir.exists():
        print("datasets directory not found!")
        return
    
    # Focus on WASP-43b data
    w43b_file = datasets_dir / "w43b_miri_new.pickle"
    
    if not w43b_file.exists():
        print("WASP-43b pickle file not found!")
        return
    
    print("Extracting WASP-43b data...")
    w43b_data = extract_pickle_data(w43b_file)
    
    if w43b_data is not None:
        print("\n" + "="*60)
        print("WASP-43b DATA EXTRACTION SUCCESSFUL!")
        print("="*60)
        
        # Clean and save the data
        time, flux, error = clean_and_save_data(w43b_data, "w43b")
        
        if time is not None:
            print(f"\nFinal data summary:")
            print(f"Time points: {len(time)}")
            print(f"Flux points: {len(flux)}")
            print(f"Time range: {time[0]:.6f} to {time[-1]:.6f}")
            print(f"Flux range: {np.min(flux):.6f} to {np.max(flux):.6f}")
            if error is not None:
                print(f"Error range: {np.min(error):.6f} to {np.max(error):.6f}")

if __name__ == "__main__":
    main()
