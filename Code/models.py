# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
import itertools
import copy

import models
import scipy.interpolate as interpolate
import warnings
warnings.filterwarnings("ignore")
import itertools
import multiprocessing
import time


def priors():
    names = ['hsig1', 'b1', 'hsig2', 'b2', 'denv', 'Henv']
    # Need to give it numax and priors parameter
    #x = joblib.load('Priors/Priors/hsig1_priors.pkl')
    pri = np.array([joblib.load('Priors/Priors/'+str(i)+'_priors.pkl') for i in names])
    return pri

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = np.median(f[:m*smoo].reshape((m,smoo)), axis=1)
    return ff

def lorentzian(f, hsig, hfreq, hexp):
    """Zero-centred Lorentzian function used to describe granulation.

    Parameters
    ----------
    f : array-like
        Input frequency array from power spectrum

    hsig : either float or [n_samples]
           Parameter describing the height of the Lorentzian function.

    hfreq: either float or [n_samples]
           Parameter describing the characterstic frequency of the Lorentzian
           function.

    hexp: either float or [n_samples]
           Parameter describing the width of the Gaussian envelope.
           width = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, where sigma is
           FWHM.

    Returns
    -------
    array of shape [n_samples, len(f)] or [len(f)]
    Shape of params dictates shape of returned model. This can either be
    n_samples models or just one model returned.
    """
    return hsig / (1 + (f/hfreq)**hexp)

def gaussian(f, henv, numax, width):
    """ Gaussian function used to describe the oscillations.

    Parameters
    ----------
    f : array-like
        Input frequency array from power spectrum

    henv : either float or [n_samples]
           Parameter describing the height of the Gaussian envelope used to
           describe the oscillations.

    numax: either float or [n_samples]
           Parameter describing the centroid of the Gaussian envelope.

    width: either float or [n_samples]
           Parameter describing the width of the Gaussian envelope.
           width = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, where sigma is
           FWHM.

    Returns
    -------
    array of shape [n_samples, len(f)] or [len(f)]
    Shape of params dictates shape of returned model. This can either be
    n_samples models or just one model returned.
    """
    tmp = width / (2.0 * np.sqrt(2.0 * np.log(2)))
    return henv * np.exp(-(f - numax)**2 / (2.0 * tmp**2))

#@profile
def model_wo(f, params):
    """ Model function for granulation with no oscillations.

    For given frequency array f and parameters array params the function
    returns the model of granulation.

    Parameters
    ----------
    f : array-like
        Input frequency array from power spectrum

    params : array-like of shape [n_samples, n_params] or [n_params]
             Array of parameters for the model. If shape is [n_samples, n_params]
             then numpy broadcasting is used to ensure quick computation of
             n_samples models. Otherwise one model if returned computed using
             params.

    Returns
    -------
    m : array of shape [n_samples, len(f)] or [len(f)]
        Shape of params dictates shape of returned model. This can either be
        n_samples models or just one model returned.
    """
    if len(params) > 13:
        # Granulation
        m = np.asarray(models.lorentzian(f, params[:,0].ravel(),
                                            params[:,1].ravel(),
                                            params[:,2].ravel()))
        # Granulation
        m += np.asarray(models.lorentzian(f, params[:,3].ravel(),
                                             params[:,4].ravel(),
                                             params[:,5].ravel()))
        # Oscillations
        # "Activity"
        m += np.asarray(models.lorentzian(f, params[:,9].ravel(),
                                             params[:,10].ravel(),
                                             params[:,11].ravel()))
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[:,12]
        return m
    else:
        # Granulation
        m = lorentzian(f, params[0], params[1], params[2])
        # Granulation
        m += lorentzian(f, params[3], params[4], params[5])
        # Oscillations
        # "Activity"
        m += lorentzian(f, params[9], params[10], params[11])
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[12]
        return m

#@profile
def model(f, params):
    """ Model function for granulation and oscillations.

    For given frequency array f and parameters array params the function
    returns the model of the oscillations and granulation.

    Parameters
    ----------
    f : array-like
        Input frequency array from power spectrum

    params : array-like of shape [n_samples, n_params] or [n_params]
             Array of parameters for the model. If shape is [n_samples, n_params]
             then numpy broadcasting is used to ensure quick computation of
             n_samples models. Otherwise one model if returned computed using
             params.

    Returns
    -------
    m : array of shape [n_samples, len(f)] or [len(f)]
        Shape of params dictates shape of returned model. This can either be
        n_samples models or just one model returned.
    """

    if len(params) > 13:
        # Granulation
        m = np.asarray(models.lorentzian(f, params[:,0].ravel(),
                                            params[:,1].ravel(),
                                            params[:,2].ravel()))
        # Granulation
        m += np.asarray(models.lorentzian(f, params[:,3].ravel(),
                                             params[:,4].ravel(),
                                             params[:,5].ravel()))
        # Oscillations
        m += np.asarray(models.gaussian(f, params[:,8].ravel(),
                                             params[:,6].ravel(),
                                             params[:,7].ravel()))
        # "Activity"
        m += np.asarray(models.lorentzian(f, params[:,9].ravel(),
                                             params[:,10].ravel(),
                                             params[:,11].ravel()))
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[:,12]
        return m

    else:
        # Granulation
        m = lorentzian(f, params[0], params[1], params[2])
        # Granulation
        m += lorentzian(f, params[3], params[4], params[5])
        # Oscillations
        m += gaussian(f, params[8], params[6], params[7])
        # "Activity"
        m += lorentzian(f, params[9], params[10], params[11])
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[12]
        return m
