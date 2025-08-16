'''
eclipse_pixel_sampling.py

Functions to fit eclipse map with pixel sampling

'''

import numpy as np
import arviz as az
import starry
import matplotlib.pyplot as plt
import math
import astropy.constants as const
import netCDF4 as nc
import math
import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
import time
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# don't evaluate starry explicitly
starry.config.lazy = True

# spherical harmonic order of smoothed maps
fit_ydeg = 4
# pixel oversampling factor for maps
oversample = 3
# oversample = 6
# width of log-normal pixel prior
tau = 0.1

# number of kfolds
n_k = 10
# number of chains to sample 8
n_chains = 8
# number of samples per chain 250
n_sample = 250
# factor to thin the posterior
thin = 4

# light curve dataset object
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


# likelihood function for data y, model theta, and uncertainty sigma
def likelihood_p(y_i,theta_i,sigma_i):
    return (1/np.sqrt(2*np.pi)) * (1/sigma_i) * \
            np.exp(-0.5 * ((y_i - theta_i)**2) / (sigma_i**2))


# calculate elpd_i for a single point i in a particular kfold
def calc_elpd_i(y_i,theta_samples,sigma_i):

    # number of samples in posterior
    n_samples = len(theta_samples)

    # calculate likelihood of each point given data
    p_samples = np.zeros(n_samples)
    for ii in range(n_samples):
        p_samples[ii] = likelihood_p(y_i,theta_samples[ii],sigma_i)

    # calculate elpd_i
    elpd_i = np.log(np.sum(p_samples) / n_samples)
    return elpd_i





# calculate indexes of kfolds over eclipse and either side
def calc_kfolds(system_params,t):
    aRs = system_params['a']
    bp = np.cos(np.pi*system_params['inc']/180) * aRs

    # calculate duration of transit
    t_transit = (system_params['period']/np.pi) * \
                np.arcsin((1/aRs)*np.sqrt((1+system_params['r_p']/system_params['r_s'])**2-bp**2) / \
                np.sin(np.pi*system_params['inc']/180.))
    
    # calculate duration of total coverage
    t_full = (system_params['period']/np.pi) * \
                np.arcsin((1/aRs)*np.sqrt((1-system_params['r_p']/system_params['r_s'])**2-bp**2) / \
                np.sin(np.pi*system_params['inc']/180.))

    # extend k-fold coverage by a factor of eclipse ingress/egress duration
    extra_krange = 1.0

    # eclipse time
    te = system_params['t0']+system_params['period']/2

    # ingress duration
    t_ingress = (t_transit - t_full)/2.0

    # edges of ingress and egress
    ti2 = te-t_full/2
    ti1 = te-t_transit/2 - extra_krange * t_ingress
    te1 = te+t_full/2
    te2 = te+t_transit/2 + extra_krange * t_ingress

    # a grazing eclipse will have te1 = nan; if it's not nan we proceed normally
    if ~np.isnan(te1):
        # indices of ingress and egress
        ingress_idx = np.where(np.logical_and(t>=ti1, t<=ti2))
        egress_idx = np.where(np.logical_and(t>=te1, t<=te2))
        t_ingress = t[ingress_idx]
        t_egress = t[egress_idx]

        # number of points in each kfold
        k_size = int(np.ceil(len(ingress_idx[0])/n_k))
        ingress_folds = math.ceil(len(t_ingress)/k_size)+1
        egress_folds = math.ceil(len(t_egress)/k_size)+1

        # define edges of kfolds
        kfold_edges_ingress = np.arange(ingress_idx[0][0],ingress_idx[0][0] + \
                                        k_size*ingress_folds,k_size)
        kfold_edges_egress = np.arange(egress_idx[0][-1] \
                                       -k_size*(egress_folds-1),egress_idx[0][-1]+k_size,k_size)

        # populate kfold list
        kfolds = []
        for jj in range(len(kfold_edges_ingress)-1):
            kfolds.append([kfold_edges_ingress[jj],kfold_edges_ingress[jj+1]])

        for jj in range(len(kfold_edges_egress)-1):
            kfolds.append([kfold_edges_egress[jj],kfold_edges_egress[jj+1]])

    # grazing eclipse; we distribute kfolds over the whole eclipse
    else:
        # eclipse indices
        eclipse_idx = np.where(np.logical_and(t>=ti1, t<=te2))
        t_eclipse = t[eclipse_idx]

        # number of points in each kfold
        k_size = int(np.ceil(len(eclipse_idx[0])/(2*n_k)))
        eclipse_folds = math.ceil(len(t_eclipse)/k_size)+1

        # define edges of kfolds
        kfold_edges_eclipse = np.arange(eclipse_idx[0][0],eclipse_idx[0][0] + \
                                        k_size*eclipse_folds,k_size)

        # populate kfold list
        kfolds = []
        for jj in range(len(kfold_edges_eclipse)-1):
            kfolds.append([kfold_edges_eclipse[jj],kfold_edges_eclipse[jj+1]])

        # ingress, egress, eclipse are the same here
        kfold_edges_ingress = kfold_edges_eclipse[:]
        kfold_edges_egress = kfold_edges_eclipse[:]

    return kfolds,kfold_edges_ingress,kfold_edges_egress,k_size,te,ti1,ti2,te1,te2
        
    

# fit a fourier series model of a phase curve, with the eclipse shape of a uniform disk
def fit_fourier_model(time_load,time_full,flux_load,sigma_load,system_params,pixel_prior_mean):
    starry.config.lazy = True

    # always use n=2 Fourier series
    fit_ndeg = 2

    # star
    pri = starry.Primary(
        starry.Map(ydeg=0, udeg=2),
        r=system_params['r_s'],
        m=system_params['m_s'],
    )     
    pri.map[1] = system_params['u1']
    pri.map[2] = system_params['u2']

    # planet
    sec = starry.Secondary(
        starry.Map(ydeg=0, udeg=0, inc=system_params["inc"]),
        r=system_params["r_p"],
        m=system_params["m_p"],
        porb=system_params["period"],
        prot=system_params["period"],
        t0=system_params["t0"],
        inc=system_params["inc"],
        theta0 = 180.
    )
    sec.map.inc = sec.inc


    with pm.Model() as model:

        # set prior on eclipse depth
        sec.map.amp = pm.Normal("sec_amp", mu=pixel_prior_mean, sd=0.3*pixel_prior_mean)

        # set prior on fitted Fourier coefficients
        ncoeff = 2*fit_ndeg
        sec_mu = np.zeros(ncoeff)
        sec_cov = 1e-1 * np.eye(ncoeff)
        N = pm.MvNormal("N", sec_mu, sec_cov, shape=(ncoeff,))
   
        # instantiate the system
        system = starry.System(pri, sec, light_delay=True)
        
        # compute the flux with the kfold removed
        flux_s, flux_p = system.flux(t=time_load,total=False)

        # compute the flux for the full time series
        flux_s_full, flux_p_full = system.flux(t=time_full,total=False)

        # combine the star and planet flux
        flux_model = flux_s + flux_p
        flux_model_full = flux_s_full + flux_p_full

        # sum up fourier series model
        nn=0
        for n in range(fit_ndeg):
            flux_model += N[nn] * flux_p * tt.sin(2 * np.pi * (n+1) * \
                               (time_load-system_params['t0']) / system_params['period'])
            flux_model_full += N[nn] * flux_p_full * tt.sin(2 * np.pi * (n+1) * \
                               (time_full-system_params['t0']) / system_params['period'])
            nn+=1
            flux_model += N[nn] * flux_p * tt.cos(2 * np.pi * (n+1) * \
                        (time_load-system_params['t0']) / system_params['period'])
            flux_model_full += N[nn] * flux_p_full * tt.cos(2 * np.pi * \
                               (n+1) * (time_full-system_params['t0']) / system_params['period'])
            nn+=1

        
        # track the flux models
        pm.Deterministic("flux_model", flux_model)
        pm.Deterministic("flux_model_full", flux_model_full)

        # likelihood function
        pm.Normal("obs", mu=flux_model, sd=sigma_load, observed=flux_load)

    # find maximum a priori solution
    with model:
        map_soln = pmx.optimize()

    # sample posterior
    with model:
        trace = pmx.sample(tune=n_sample, draws=n_sample, start=map_soln,  \
                           chains=n_chains, cores=n_chains,progressbar=False, target_accept=0.9)
         
    return trace



# fit an eclipse map model by sampling pixels evenly spaced in area
def fit_map_model(alpha,time_load,time_full,flux_load,sigma_load,ydeg,system_params,pixel_prior_mean):

    # spherical harmonic order of map
    fit_ydeg = int(ydeg)

    # Primary
    pri = starry.Primary(
        starry.Map(ydeg=0, udeg=2),
        r=system_params['r_s'],
        m=system_params['m_s'],
    )    

    pri.map[1] = system_params['u1']
    pri.map[2] = system_params['u2']


    # Secondary
    sec = starry.Secondary(
        starry.Map(ydeg=fit_ydeg, udeg=0, inc=system_params["inc"]),
        r=system_params["r_p"],
        m=system_params["m_p"],
        porb=system_params["period"],
        prot=system_params["period"],
        t0=system_params["t0"],
        inc=system_params["inc"],
        theta0 = 180.
    )
    sec.map.inc = sec.inc


    # Get pixel transform matrix and number of pixels
    lat, lon, Y2P, P2Y, Dx, Dy = sec.map.get_pixel_transforms(oversample=oversample)
    npix = P2Y.shape[1]
    mu = np.log(pixel_prior_mean/np.pi)

    with pm.Model() as model:
        pixels = pm.Lognormal("pixels", mu=mu,tau=tau,shape=(npix,))

        # Transform pixels to spherical harmonics
        starry_x = tt.dot(P2Y, pixels)

        # Instantiate the system
        system = starry.System(pri, sec, light_delay=True)

        # Calculate light curve
        starry_X = system.design_matrix(time_load)
        starry_X_full = system.design_matrix(time_full)

        # calculate flux model with kfold removed, then flux model for full time series
        flux_model = starry_X[:, 0] + tt.dot(starry_X[:, 1:],starry_x)
        flux_model_full = starry_X_full[:, 0] + tt.dot(starry_X_full[:, 1:],starry_x)

        # Calculate and record map
        map_plot = starry.Map(ydeg=fit_ydeg)
        map_plot.amp = starry_x[0]
        map_plot[1:, :] = starry_x[1:]/starry_x[0]
        map_grid = np.pi*map_plot.render(projection="rect",res=100)
        pm.Deterministic("map_grid",map_grid)

        # default uniform image D
        D = tt.mean(pixels)
        # entropy S
        S = -tt.sum(pixels*tt.log(pixels/D))

        # record entropy and flux models
        pm.Deterministic("S", S)
        pm.Deterministic("flux_model", flux_model)
        pm.Deterministic("flux_model_full", flux_model_full)

        # The likelihood function assuming known Gaussian uncertainty
        pm.Normal("obs", mu=flux_model, sd=sigma_load, observed=flux_load)
        pm.Potential("reg", 2*S*alpha)

    # find maximum a priori solution
    with model:
        map_soln = pmx.optimize()

    # sample posterior 
    with model:
        trace = pmx.sample(tune=n_sample, draws=n_sample, start=map_soln, \
                           chains=n_chains, cores=n_chains,progressbar=False, target_accept=0.9)
        
    return trace


# fit a model to a dataset with a particular kfold removed
def sample_kfold(t,flux,sigma,system_params,pixel_prior_mean,kfold,alpha,period_n,k_size,ydeg):
    
    # indices covered by removed kfold
    print(kfold)
    kfold_range = np.arange(kfold[0],kfold[1])

    # if the kfold starts at zero, do the special case where nothing is removed
    if kfold[0] == 0:
        time_load = t[:]
        flux_load = flux[:]
        sigma_load = sigma[:]
    # otherwise delete the data in the kfold 
    else:
        time_load = np.delete(t,kfold_range)
        flux_load = np.delete(flux,kfold_range)
        sigma_load = np.delete(sigma,kfold_range)

    # empty array for map grid samples
    map_grid_samples = []

    # fix for when starry refuses to run -- running the same command again a second time fixes it
    try:
        # if smoothing parameter alpha is negative, ignore it and fit the fourier series model
        if alpha < 0:
            trace = fit_fourier_model(time_load,t,flux_load,sigma_load,system_params,pixel_prior_mean)
            flux_model_samples,flux_model_full_samples = trace.flux_model, trace.flux_model_full
        # otherwise, fit the map model
        else:
            trace = fit_map_model(alpha,time_load,t,flux_load,sigma_load,ydeg,system_params,pixel_prior_mean)
            flux_model_samples,flux_model_full_samples,map_grid_samples = \
            trace.flux_model, trace.flux_model_full, trace.map_grid
    except:
        if alpha < 0:
            trace = fit_fourier_model(time_load,t,flux_load,sigma_load,system_params,pixel_prior_mean)
            flux_model_samples,flux_model_full_samples = trace.flux_model, trace.flux_model_full
        else:
            trace = fit_map_model(alpha,time_load,t,flux_load,sigma_load,ydeg,system_params,pixel_prior_mean)
            flux_model_samples,flux_model_full_samples,map_grid_samples =  \
            trace.flux_model, trace.flux_model_full, trace.map_grid

    # list of elpd_i values for each data point
    elpd_i_list = []

    # for each data point in kfold, calculate the predictive density
    for ii in kfold_range:
        # y is data, sigma is uncertainty
        y_i = flux[ii]
        sigma_i = sigma[ii]

        # get model posterior for each data point
        theta_samples = flux_model_full_samples[:,ii]

        # calculate elpd for each data point given the model
        elpd_i = calc_elpd_i(y_i,theta_samples,sigma_i)
        elpd_i_list.append(elpd_i)

    # sum up the elpd score for each data point
    elpd_i_list = np.array(elpd_i_list)
    elpd_xval = np.sum(elpd_i_list)

    
    return elpd_xval, elpd_i_list, flux_model_samples, flux_model_full_samples, map_grid_samples, trace

# sampler function to fit models over each kfold
def sampler(t,flux,sigma,system_params,pixel_prior_mean,alpha_list,ydeg_list,kfolds,period_n,k_size):  

    # output data lists
    elpd_list_list = []
    elpd_all_list_list = []
    flux_model_samples_list = []
    flux_model_full_samples_list = []
    map_grid_samples_list = []
    S_list = []

    # track number of sampling runs
    ntests=len(alpha_list)*(len(kfolds)+1)
    print(ntests,' tests')
    test_i=0

    # track index of test
    jj=0

    # loop over model, order, and alpha smoothing value tests
    for alpha in alpha_list:
        elpd_list = []
        elpd_all_list = []

        # run through kfolds
        for kfold in kfolds[:]:
            
            # measure time of one test to estimate total runtime
            start = time.time()

            # fit model with kfold data removed
            elpd_xval,elpd_i_list,_,_,_,_ = sample_kfold(t,flux,sigma,system_params, \
                                                         pixel_prior_mean,kfold,alpha,period_n,k_size,ydeg_list[jj])

            # estimate run time remaining
            end = time.time()
            test_i+=1
            print(np.round((1+ntests-test_i)*(end-start)/60,3), ' minutes to finish '+str(test_i))

            # record elpd value for this kfold
            elpd_list.append(elpd_xval)
            elpd_all_list.append(elpd_i_list)

        
        # fit to full dataset without leaving out k-fold
        _,_,flux_model_samples,flux_model_full_samples,map_grid_samples,trace = \
        sample_kfold(t,flux,sigma,system_params,pixel_prior_mean,[0,1],alpha,period_n,k_size,ydeg_list[jj])

        # record posterior
        flux_model_samples_list.append(flux_model_samples[::thin])
        flux_model_full_samples_list.append(flux_model_full_samples[::thin])
        map_grid_samples_list.append(map_grid_samples[::thin])

        # record entropy posterior for map models
        if alpha >= 0:
            S_list.append(trace.S[::thin])

        # record summed elpd values
        elpd_list_list.append(np.array(elpd_list))

        # record all elpd values
        elpd_all_list_list.append(np.array(elpd_all_list))
        
        jj+=1

        
    return elpd_list_list, elpd_all_list_list, flux_model_samples_list, \
    flux_model_full_samples_list, map_grid_samples_list, S_list

 


