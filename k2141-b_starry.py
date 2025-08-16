import numpy as np 
import starry 
import xarray as xr 
import xesmf as xe
import matplotlib.pyplot as plt 

starry.config.lazy = True
starry.config.quiet = True

# constants 
h = 6.626e-34
c = 3.0e+8
k = 1.38e-23
sigma = 5.67e-8

# star parameters 
R_sun = 7.e8 # radius of sun
M_sun = 2e30 # mass of sun
L_sun = 3.8e26 # luminosity of sun
R_s_norm = 0.681
M_s_norm = 0.708
R_s = R_s_norm*R_sun   # stellar radius 
M_s = M_s_norm*M_sun
L_s = 0.186*L_sun   # stellar luminosity 
T_s = (L_s / (4*np.pi*sigma*R_s**2.))**(1./4.) # stellar effective tempearture 


print(T_s)

# planet parameters 
a_e = 6.371e6 # radius of earth
a = 1.51*a_e # radius 
a_planet_norm = a / R_sun
rho = 7930
m = 4./3. * np.pi * a**3. * rho 
m_planet_norm = m / M_sun 

# orbit 
period = 0.2803244 # days 
tc = 2457744.07160 # time of conjunction
inc = 86.3
ecc = 0 
w_peri = 0 
Om_asc = 0 

def regrid(data):
    
    lat = data.lat.values 
    lon = data.lon.values 
    
    lat_new = np.arange(-90,90,5) + 2.5
    lon_new = np.arange(0,360,5)+2.5 
    
    grid = xr.Dataset({
            "lat": (["lat"], lat, {"units": "degrees_north"}),
            "lon": (["lon"], lon, {"units": "degrees_east"}),})

    grid_out = xr.Dataset({
            "lat": (["lat"], lat_new, {"units": "degrees_north"}),
            "lon": (["lon"], lon_new, {"units": "degrees_east"}),})
    
    regridder = xe.Regridder(grid, grid_out, 'bilinear', periodic=True)
    
    data = regridder(data, keep_attrs=True)
    
    return data 


def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

def calc_miri_emission(olr_T, Tstar=T_s):
    # wavelength bins
    nwls = 10000

    # miri bandpass (not weighted yet)
    wls = np.logspace(5e-6,12e-6,nwls)
    planet_miri_emission = np.zeros([nwls,np.shape(olr_T)[0],np.shape(olr_T)[1]])

    # planck emission at every lat/lon point 
    for ii in range(np.shape(olr_T)[1]):
        for jj in range(np.shape(olr_T)[0]):
            planet_miri_emission[:,jj,ii] = np.pi*planck(wls,olr_T[jj,ii])

    # integrate over wavelengths
    planet_miri_total_emission = np.trapz(planet_miri_emission,wls,axis=0)
    star_miri_total_emission = np.trapz(np.pi*planck(wls,Tstar),wls,axis=0)
    
    return planet_miri_total_emission, star_miri_total_emission






pri = starry.Primary(
    starry.Map(ydeg=1, udeg=2, amp = 1.0),
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

# limb darkening
pri.map[1] = 0.40
pri.map[2] = 0.26

# need to apply inc and obl to map as well as planet
sec.map.inc = sec.inc
sec.map.obl = sec.Omega

# uniform stellar map
pri.map.load(np.ones([100,100]))



# set time axis
tstart = tc + (-0.25)*period - 0*period/2
tend = tstart+period*3.

 # observational cadence and binning
cadence = 10
bin_cad = 1
cadence = cadence * bin_cad
npoints = int((tend-tstart)*86400/cadence)
t = np.linspace(tstart,tend,npoints)
print(npoints)



flux_dict = {}

sims = ['k2-141b_10.0bar_N2_lowdrag','k2-141b_1.0bar_N2_lowdrag','k2-141b_0.1bar_N2_lowdrag', 
        'k2-141b_10.0bar_H2_lowdrag','k2-141b_1.0bar_H2_lowdrag','k2-141b_0.1bar_H2_lowdrag']

for sim in sims: 

    ds = xr.open_dataset('k2-141b/'+sim+'.nc')
    ds = regrid(ds)
    ds = ds.squeeze(dim='time', drop=True)

    olr = ds.flux_lw.isel(phalf=0).values 
    olr_T = (olr / sigma) ** (1./4.)

    planet_emission, star_emission = calc_miri_emission(olr_T)
    sec_map = ((a_planet_norm/R_s_norm)**2)*planet_emission/star_emission

    # load 2D planet map
    sec.map.load(sec_map)

    # calculate flux of system
    sys=starry.System(pri,sec)
    fs,fp = sys.flux(t,total=False)
    fp = fp.eval()
    fs = fs.eval()
    flux = (fp+fs)/fs[0]
    flux_dict[sim] = flux
#print(len(flux))

ls = ['-','--',':']
for n, sim in enumerate(sims[0:3]):
    plt.plot((t-tc)*24,flux_dict[sim],lw=1.5, label=sim, color='C1', ls=ls[n])#1e6*(flux-1)
for n, sim in enumerate(sims[3:]):
    if n == 0:
        plt.plot((t-tc)*24,flux_dict[sim],lw=1.5, label=sim, color='C2', ls=ls[n])#1e6*(flux-1)
    else:
        plt.plot((t-tc)*24,flux_dict[sim],lw=1.5, color='C2', ls=ls[n])#1e6*(flux-1)

plt.ylim(0.9995,1.0003)
#plt.xlim((t[0]-tc)*24,(t[0]-tc)*24 + period*24)
# plt.xlim(-10,10)

plt.xlabel('Time (hours)')
plt.ylabel('$F_{p}/F_{s}$ (ppm)')

#obs_data = np.genfromtxt('obs_data.csv',delimiter=',')
#plt.scatter((obs_data[:,0]-obs_data[0,0])*24-2.03,obs_data[:,1],c='r',s=15,label='Observations')#1e6*(obs_data[:,1]-1)
plt.yticks([0.9995, 0.99975, 1., 1.00025, 1.0005])
plt.axhline(y=1, color='k', ls='--')

plt.legend(loc='best')
#plt.show()
plt.savefig('pc_low.png', dpi=400)

sec.r = a_planet_norm * 3
sys.show(t[::50], cmap='plasma', file='sys.mp4', figsize=(8,8), html5_video=False, window_pad=1.3)