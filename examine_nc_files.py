import os
from netCDF4 import Dataset
import numpy as np

# Directory containing the NetCDF files
nc_dir = 'k2-141b_timavgs'

# Get all .nc files
nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.nc')]
print(f"Found {len(nc_files)} NetCDF files:")

# Print all file names
for file in nc_files:
    print(f"- {file}")

# Examine the structure of the first file
if nc_files:
    first_file = os.path.join(nc_dir, nc_files[0])
    print(f"\nExamining structure of: {nc_files[0]}")
    
    with Dataset(first_file, 'r') as nc:
        print("\nDimensions:")
        for dim_name, dim in nc.dimensions.items():
            print(f"- {dim_name}: {len(dim)}")
        
        print("\nVariables:")
        for var_name, var in nc.variables.items():
            print(f"\nVariable: {var_name}")
            print(f"  Shape: {var.shape}")
            print(f"  Data type: {var.dtype}")
            print(f"  Attributes:")
            for attr_name in var.ncattrs():
                print(f"    {attr_name}: {var.getncattr(attr_name)}") 