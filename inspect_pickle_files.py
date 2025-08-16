#!/usr/bin/env python3
"""
Test script to inspect the structure of pickle files in the datasets folder.
This will help us understand the data format for the generic light curve mapper.
"""

import pickle
import numpy as np
import os
from pathlib import Path
import sys

def inspect_pickle_file_robust(filepath):
    """Inspect pickle file with robust error handling for unknown classes"""
    print(f"============================================================")
    print(f"Inspecting: {filepath}")
    print(f"============================================================")
    
    try:
        # Try to load with a custom unpickler that can handle unknown classes
        with open(filepath, 'rb') as f:
            # Read the raw pickle data
            data = f.read()
            
        # Try to extract basic information from the pickle data
        print(f"File size: {len(data)} bytes")
        
        # Look for common patterns in the pickle data
        if b'lc_dataset' in data:
            print("Contains 'lc_dataset' object")
        if b'light_curve' in data:
            print("Contains 'light_curve' data")
        if b'time' in data:
            print("Contains 'time' data")
        if b'flux' in data:
            print("Contains 'flux' data")
        if b'error' in data:
            print("Contains 'error' data")
        if b'params' in data:
            print("Contains 'params' data")
        if b'system_params' in data:
            print("Contains 'system_params' data")
            
        # Try to decode some of the pickle structure
        try:
            # Use a more permissive approach
            import dill
            with open(filepath, 'rb') as f:
                obj = dill.load(f)
                print(f"Successfully loaded with dill")
                print(f"Object type: {type(obj)}")
                
                # Try to access common attributes
                if hasattr(obj, '__dict__'):
                    print(f"Object attributes: {list(obj.__dict__.keys())}")
                    
                    # Try to access specific attributes
                    for attr in ['time', 'flux', 'error', 'params', 'system_params', 'data']:
                        if hasattr(obj, attr):
                            val = getattr(obj, attr)
                            print(f"{attr}: {type(val)}")
                            if isinstance(val, (np.ndarray, list)):
                                print(f"  Shape/length: {val.shape if hasattr(val, 'shape') else len(val)}")
                            elif isinstance(val, dict):
                                print(f"  Keys: {list(val.keys())}")
                                
        except Exception as e:
            print(f"Dill loading failed: {e}")
            
        # Try with pickle and ignore errors
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
                print(f"Successfully loaded with pickle")
                print(f"Object type: {type(obj)}")
        except Exception as e:
            print(f"Pickle loading failed: {e}")
            
    except Exception as e:
        print(f"Error reading pickle file: {e}")

def main():
    """Main function to inspect all pickle files"""
    datasets_dir = Path("datasets")
    
    if not datasets_dir.exists():
        print("datasets directory not found!")
        return
    
    pickle_files = list(datasets_dir.glob("*.pickle"))
    
    if not pickle_files:
        print("No pickle files found in datasets directory!")
        return
    
    print(f"Found {len(pickle_files)} pickle files:")
    for file in pickle_files:
        print(f"  - {file.name}")
    
    print()
    
    for file in pickle_files:
        inspect_pickle_file_robust(file)
        print()

if __name__ == "__main__":
    main()
