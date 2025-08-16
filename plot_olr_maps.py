import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import starry
import zipfile

# Configure starry
starry.config.lazy = True
starry.config.quiet = True

# Physical constants
h = 6.626e-34  # Planck's constant
c = 3.0e8      # Speed of light
k = 1.38e-23   # Boltzmann constant
sigma = 5.67e-8  # Stefan-Boltzmann constant

# Star parameters
R_sun = 696340e3  # radius of sun in meters
M_sun = 1.989e30  # mass of sun in kg
R_s_norm = 0.681
M_s_norm = 0.708
R_s = R_s_norm*R_sun  # stellar radius
M_s = M_s_norm*M_sun
T_s = 4570  # stellar effective temperature in K

# Planet parameters
M_jup = 1.898e27  # mass of Jupiter in kg
m_planet_norm = 0.0156  # planet mass in Jupiter masses
m = m_planet_norm * M_jup  # planet mass in kg
a_e = 6.371e6  # radius of earth in meters
a = 1.51*a_e  # planet radius in meters
a_planet_norm = a / R_sun  # normalized planet radius

# Orbit parameters
period = 0.2803244  # days
tc = 2457744.07160  # time of conjunction
inc = 86.3
ecc = 0
w_peri = 0
Om_asc = 0

def planck(wav, T):
    """Calculate Planck function for given wavelength and temperature"""
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    return a/((wav**5)*(np.exp(b)-1.0))

def integrate_planck(T, wl_min=6e-6, wl_max=12e-6, npoints=1000):
    """Integrate Planck function over wavelength range"""
    wls = np.logspace(np.log10(wl_min), np.log10(wl_max), npoints)
    intensity = planck(wls, T)
    return np.trapz(intensity, wls)

# Create output directories if they don't exist
for dir_name in ['olr_maps', 'fpfs_maps', 'phase_curves']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Directory containing the NetCDF files
nc_dir = 'k2-141b_timavgs'

# Get all .nc files
nc_files = [f for f in os.listdir(nc_dir) if f.endswith('.nc')]

# Calculate star's integrated emission
star_emission = integrate_planck(T_s)

# Set up starry system
pri = starry.Primary(
    starry.Map(ydeg=1, udeg=2, amp=1.0),
    r=R_s_norm,
    m=M_s_norm,
)

sec = starry.Secondary(
    starry.Map(ydeg=10, udeg=0),
    r=a_planet_norm,
    m=m_planet_norm,
    porb=period,
    prot=period,
    t0=tc,
    inc=inc,
    theta0=180.,
    ecc=ecc,
    w=w_peri,
    Omega=Om_asc
)

# Set limb darkening
pri.map[1] = 0.40
pri.map[2] = 0.26

# Set map orientation
sec.map.inc = sec.inc
sec.map.obl = sec.Omega

# Set time axis for phase curves
tstart = tc + (-0.25)*period
tend = tstart + period*3
npoints = 1000
t = np.linspace(tstart, tend, npoints)

# Loop through each file
flux_dict = {}  # Store fluxes for combined plot
for file in nc_files:
    print(f"Processing {file}...")
    
    # Open the NetCDF file
    ds = xr.open_dataset(os.path.join(nc_dir, file))
    
    # Extract OLR field (flux_lw at phalf=0) and convert to numpy array
    olr = ds.flux_lw.isel(phalf=0, time=0).values
    
    # Convert OLR to brightness temperature
    T_bright = (olr / sigma)**(1/4)
    
    # Calculate planet's integrated emission
    planet_emission = np.zeros_like(T_bright)
    for i in range(T_bright.shape[0]):
        for j in range(T_bright.shape[1]):
            planet_emission[i,j] = integrate_planck(T_bright[i,j])
    
    # Calculate Fp/Fs ratio
    fpfs = (a_planet_norm/R_s_norm)**2 * (planet_emission / star_emission)
    
    # Get coordinates for plotting
    lon = ds.lon.values
    lat = ds.lat.values
    
    # Plot OLR map
    plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(lon, lat, olr, cmap='viridis', shading='auto')
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.1)
    cbar.set_label('Outgoing Longwave Radiation (W/m²)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title(f'OLR Map - {file}')
    plt.savefig(os.path.join('olr_maps', f'olr_{os.path.splitext(file)[0]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Fp/Fs map
    plt.figure(figsize=(12, 6))
    im = plt.pcolormesh(lon, lat, fpfs*1e6, cmap='viridis', shading='auto')  # Convert to ppm for display
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.1)
    cbar.set_label('Fp/Fs (ppm)')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.title(f'Fp/Fs Map - {file}')
    plt.savefig(os.path.join('fpfs_maps', f'fpfs_{os.path.splitext(file)[0]}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load Fp/Fs map into starry (convert to intensity and ppm)
    pri.map.load(np.ones([100,100]))
    sec.map.load(fpfs)
    
    # Calculate phase curve
    sys = starry.System(pri, sec)
    fs, fp = sys.flux(t, total=False)
    fp = fp.eval()
    fs = fs.eval()
    flux = 1e6*(((fp + fs) / fs[0]) - 1)
    flux_dict[file] = flux  # Store for combined plot
    
    # Plot individual phase curve
    plt.figure(figsize=(10, 6))
    plt.plot((t - tc) * 24, flux, 'k-', lw=2)
    plt.axhline(y=0, color='k', ls='--', alpha=0.3)
    plt.xlabel('Time (hours)')
    plt.ylabel('$F_{p}/F_{s}$')
    plt.title(f'Phase Curve - {file}')
    plt.savefig(os.path.join('phase_curves', f'phase_{os.path.splitext(file)[0]}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots for {file}")

# Create combined phase curve plot
plt.figure(figsize=(12, 8))
line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]  # More line styles
colors = ['C1', 'C2']  # Orange for N2, Green for H2

# Sort files by pressure and optical depth
def get_pressure(file):
    if 'ps1bar' in file:
        return 1
    elif 'ps3bar' in file:
        return 3
    elif 'ps5bar' in file:
        return 5
    elif 'ps10bar' in file:
        return 10
    return 0

def get_optical_depth(file):
    if 'od0.8' in file:
        return 0.8
    elif 'od2.4' in file:
        return 2.4
    elif 'od4.0' in file:
        return 4.0
    elif 'od8.0' in file:
        return 8.0
    return 0

# Sort N2 files
n2_files = sorted([f for f in nc_files if 'N2' in f], 
                  key=lambda x: (get_pressure(x), get_optical_depth(x)))

# Sort H2 files
h2_files = sorted([f for f in nc_files if 'H2' in f], 
                  key=lambda x: (get_pressure(x), get_optical_depth(x)))

# Plot N2 cases
for n, file in enumerate(n2_files):
    plt.plot((t - tc) * 24, flux_dict[file], 
             color=colors[0], ls=line_styles[n % len(line_styles)], 
             lw=1.5, label=file)

# Plot H2 cases
for n, file in enumerate(h2_files):
    plt.plot((t - tc) * 24, flux_dict[file], 
             color=colors[1], ls=line_styles[n % len(line_styles)], 
             lw=1.5, label=file)

plt.axhline(y=0, color='k', ls='--', alpha=0.3)
plt.xlabel('Time (hours)')
plt.ylabel('$F_{p}/F_{s}$ (ppm)')
plt.title('Phase Curves Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside plot
plt.tight_layout()  # Adjust layout to make room for legend
plt.savefig(os.path.join('phase_curves', 'all_phase_curves.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been generated!")

# Create directory for npy files if it doesn't exist
npy_dir = 'phase_curve_data'
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)

# Save each phase curve as .npy file
for file in nc_files:
    # Create filename without extension
    base_name = os.path.splitext(file)[0]
    # Combine time and flux into a single array
    time_flux = np.column_stack(((t - tc) * 24, flux_dict[file]))
    # Save combined data
    np.save(os.path.join(npy_dir, f'{base_name}_phase_curve.npy'), time_flux)

# Create zip archive
with zipfile.ZipFile('phase_curves.zip', 'w') as zipf:
    for file in os.listdir(npy_dir):
        if file.endswith('.npy'):
            zipf.write(os.path.join(npy_dir, file), file)

print("All phase curves have been saved as .npy files and zipped!") 