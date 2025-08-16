'''
paper_eclipse_suite.py

Fit eclipse map to simulated or real data

'''

import numpy as np
import arviz as az
import dill  
import theano
import starry
import pickle
import matplotlib.pyplot as plt
import math
import os
import astropy.constants as const
import netCDF4 as nc
import math
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
import time
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import argparse
import eclipse_pixel_sampling

starry.config.lazy = True
starry.config.quiet = False

# great circle distance between two points
def haversine(lon1, lat1, lon2, lat2, r):
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * r

# test_case = 'gcm_highprec'
# test_case = 'gcm_medprec'
# test_case = 'gcm_lowprec'
# test_case = 'spot_highprec'
# test_case = 'spot_medprec'
# test_case = 'obs_w43b'
# test_case = 'obs_w18b'
# test_case = 'obs_hd189'

# parse input test case
parser = argparse.ArgumentParser()
parser.add_argument('case', type=str)
args = parser.parse_args()
test_case = args.case

print(test_case)



class lc_dataset:
    name = 'name'
    bin_t = None
    bin_flux = None
    bin_sigma = None
    t = None
    flux = None
    sigma = None
    system_params = None
    pc_only_idx = None


delta_inc = 0.0
delta_te = 0./86400.

if test_case == 'gcm_highprec':
    load_name = 'w43b_sim'
    map_model = 'THOR_shift'
    ferr0 = 150
elif test_case == 'gcm_medprec':
    load_name = 'w43b_sim'
    map_model = 'THOR_shift'
    ferr0 = 250
elif test_case == 'gcm_lowprec':
    load_name = 'w43b_sim'
    map_model = 'THOR_shift'
    ferr0 = 2000
elif test_case == 'spot_highprec':
    load_name = 'w43b_sim'
    map_model = 'blob_60'
    ferr0 = 150
    bump_deg=60
    bump=np.pi*bump_deg/180.
elif test_case == 'spot_medprec':
    load_name = 'w43b_sim'
    map_model = 'blob_60'
    ferr0 = 250
    bump_deg=60
    bump=np.pi*bump_deg/180.
elif test_case == 'obs_w43b':
    load_name = 'w43b_miri_new'
    map_model = None
    ferr0 = None
elif test_case == 'obs_w18b':
    load_name = 'w18b_niriss'
    map_model = None
    ferr0 = None
elif test_case == 'obs_hd189':
    load_name = 'hd189_spitzer'
    map_model = None
    ferr0 = None

# name of test case
name = test_case + str(int(delta_te*86400))

print(name)


# list of smoothing magnitudes 16
n_alphas = 16
alpha_list = np.concatenate([[-1,0,0],np.logspace(1,6,n_alphas)])

# n=2 fourier series, l=2 eclipse map, l=4 eclipse map, then smoothed l=4 eclipse maps
ydeg_list = np.array(0*alpha_list) + eclipse_pixel_sampling.fit_ydeg
ydeg_list[:2] = 2


# observational data
if load_name != 'w43b_sim':
    
    # load observational dataset
    with open('datasets/'+load_name+'.pickle', 'rb') as file2:
        load_dataset = pickle.load(file2)

    # system parameters
    system_params = load_dataset.system_params

    # load data
    t = load_dataset.t
    flux = load_dataset.flux
    sigma = load_dataset.sigma

    # remove nans
    t = t[~np.isnan(flux)]
    sigma = sigma[~np.isnan(flux)]
    flux = flux[~np.isnan(flux)]

    # default pixel prior mean
    pixel_prior_mean = 5000e-6

    if load_name == 'w18b_niriss':
        pixel_prior_mean = 1000e-6
        system_params['period'] = 0.941452382
        system_params['a'] = 3.483
        system_params['inc'] = 84.39
        system_params['t0'] = system_params['t0'] - system_params['period']*1
        system_params['u1'] = 0.2
        system_params['u2'] = 0.2

    if load_name == 'hd189_spitzer':
        pixel_prior_mean = 3000e-6
        system_params['a'] = 8.863
        system_params['t0'] = system_params['t0'] - system_params['period']*1
        system_params['u1'] = 0.2
        system_params['u2'] = 0.2

    if load_name == 'w43b_miri_new':
        pixel_prior_mean = 5000e-6
        system_params['t0'] = system_params['t0'] + system_params['period']*4892
        system_params['u1'] = 0.160539777
        system_params['u2'] = -0.02563240137

    # calculate consistent planet mass for starry
    a = system_params['a']*system_params['r_s']*const.R_sun.value
    p = system_params['period']*(24.*3600.)
    b_m = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value - system_params['m_s'])
    system_params['m_p'] = b_m

    # true flux is not known for observational data
    flux_true = 0.0 * flux[:]

# simulated data
else:
    # base simulated data off wasp-43b system parameters
    load_name_sys = 'w43b_miri_new'
    with open('datasets/'+load_name_sys+'.pickle', 'rb') as file2:
        load_dataset = pickle.load(file2)
        
    system_params = load_dataset.system_params
    
    # limb darkening is arbitrary
    system_params['u1'] = 0.2
    system_params['u2'] = 0.2

    # default pixel prior mean
    pixel_prior_mean = 5000e-6

    # set up star
    A = starry.Primary(starry.Map(ydeg=0, udeg=2, amp=1.0), m=system_params['m_s'], r=system_params['r_s'])
    A.map[1] = system_params['u1']
    A.map[2] = system_params['u2']

    # calculate self-consistent planet mass
    a = system_params['a']*system_params['r_s']*const.R_sun.value
    p = system_params['period']*(24.*3600.)
    b_m = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value - system_params['m_s'])
    system_params['m_p'] = b_m


    # set up planet
    b = starry.Secondary(
        starry.Map(ydeg=10, udeg=0, amp=5000e-6, inc=system_params['inc']),
        r=system_params['r_p'],
        m=b_m,
        inc=system_params['inc'],
        prot=system_params['period'],
        porb=system_params['period'],
        t0=system_params['t0'],
        theta0 = 180.
    )
    # ensure map inclination is the same as planet inclination
    b.map.inc = b.inc

    # longitude and latitude
    lons = np.linspace(-np.pi,np.pi,90)[None,:]
    lats = np.linspace(-np.pi/2,np.pi/2,45)[:,None]

    # centre of the Gaussian bump if it's used
    lat0 = np.pi/6
    lon0 = np.pi/6
    
    # magnitude of Gaussian bump map
    fmag = 3000e-6

    # load post-processed OLR from THOR WASP-43b simulation from 2018, post-processed in 2018
    thor_olr = np.load('thor_olr.npy')
    thor_olr_roll = np.roll(np.load('thor_olr.npy'),int(np.shape(thor_olr)[-1]/2),axis=1)
    

    # GCM map
    if map_model == 'THOR_shift':
        test_map = thor_olr_roll[:]

    # gaussian bump map
    elif map_model == 'blob_60':
        
        # set up map with same dimensions as THOR map
        test_map = np.zeros_like(thor_olr_roll) + fmag

        # calculate bump using haversine formula
        distance_grid = np.zeros_like(test_map)
        for ii in range(np.shape(test_map)[1]):
            for jj in range(np.shape(test_map)[0]):
                distance_grid[jj,ii] = haversine(lon0,lat0,lons[0,ii],lats[jj,0],1)

        # add bump to flat map
        test_map += 2*fmag*np.exp(-0.5 * (distance_grid**2) / (bump**2))

    # load in simulated map
    b.map.load(test_map/np.pi)

    # set up starry system
    sys = starry.System(A, b,light_delay=True)

    # number of simulated points
    npoints = 7000

    # time series covering one eclipse and one transit
    t = np.linspace(0.4*system_params['period']+system_params['t0'],0.4*system_params['period']+ system_params['t0']+1.0*system_params['period'], int(npoints))

    # calculate true system flux
    flux_true = sys.flux(t).eval()

    # convert uncertainty from ppm
    ferr = ferr0*1e-6

    # add noise to true system flux
    flux = flux_true + ferr * np.random.randn(len(t))
    sigma = ferr + np.zeros_like(flux)
    


# calculate kfold region edges
kfolds,kfold_edges_ingress,kfold_edges_egress,k_size,te,ti1,ti2,te1,te2 =  \
                            eclipse_pixel_sampling.calc_kfolds(system_params,t)

# wasp-43b dataset has two eclipses; measure the number of datapoints in one period
if load_name == 'w43b_miri_new':
    period_n = int(system_params['period']/(t[1]-t[0]))
else:
    period_n = 0


system_params['t0'] = system_params['t0'] + delta_te
system_params['inc'] = system_params['inc'] + delta_inc

# call sampling function
elpd_list_list, elpd_all_list_list, flux_model_samples_list, \
flux_model_full_samples_list, map_grid_samples_list, S_list = \
eclipse_pixel_sampling.sampler(t,flux,sigma,system_params,pixel_prior_mean,alpha_list,ydeg_list,kfolds[:],period_n,k_size)


# output data object
class fit_output:
    name = name
    fit_ydeg = None
    alpha_list = None
    flux_model_samples_list = None
    flux_model_full_samples_list = None
    elpd_list_list = None
    elpd_all_list_list = None
    true_map = None
    kfolds = None
    S_list = None
    
    t = t
    flux = flux
    sigma = sigma
    flux_true = flux_true
    

# populate output data object
fit_output.S_list = S_list
fit_output.fit_ydeg = ydeg_list
fit_output.kfolds = kfolds
fit_output.alpha_list = alpha_list
fit_output.flux_model_samples_list = flux_model_samples_list
fit_output.flux_model_full_samples_list = flux_model_full_samples_list
fit_output.map_grid_samples_list = map_grid_samples_list
fit_output.elpd_list_list = elpd_list_list
fit_output.elpd_all_list_list = elpd_all_list_list

# data output path
out_path = '/network/group/aopp/planetary/RTP010_HAMMOND_55CLOUDY/starry_output/'

# save with label of current time
now = datetime.now()
date_time = now.strftime("%m%d%Y_%H%M%S")
with open(out_path+name+'_'+date_time, 'wb') as f:
    dill.dump(fit_output, f)

    


