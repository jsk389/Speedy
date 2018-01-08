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

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = np.median(f[:m*smoo].reshape((m,smoo)), axis=1)
    return ff

def lorentzian(f, hsig, hfreq):
    """Zero-centred super-Lorentzian function used to describe granulation.

    Parameters
    ----------
    f : array-like
        Input frequency array from power spectrum

    hsig : either float or [n_samples]
           Parameter describing the height of the Lorentzian function.

    hfreq: either float or [n_samples]
           Parameter describing the characterstic frequency of the Lorentzian
           function.

    Returns
    -------
    array of shape [n_samples, len(f)] or [len(f)]
    Shape of params dictates shape of returned model. This can either be
    n_samples models or just one model returned.
    """
    return hsig / (1 + (f/hfreq)**4)

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
    if len(params) > 10:
        # Granulation
        new_f = np.append(f, f+f.max())
        m = np.asarray(models.lorentzian(new_f, params[:,0].ravel(),
                                            params[:,1].ravel()))
        # Granulation
        m += np.asarray(models.lorentzian(new_f, params[:,2].ravel(),
                                             params[:,3].ravel()))
        # Oscillations
        # "Activity"
        m += np.asarray(models.lorentzian(new_f, params[:,7].ravel(),
                                             params[:,8].ravel()))
        # Sampling effect
        m *= np.sinc(f / 2.0 / (284.0*2))**2.0

        # Accounting for effect near Nyquist
        m = m[:len(m)//2,:] + m[len(m)//2:,:][::-1]
        # White Noise
        m += params[:,9]
        return m
    else:
        new_f = np.append(f, f+f.max())
        # Granulation
        m = lorentzian(new_f, params[0], params[1])
        # Granulation
        m += lorentzian(new_f, params[2], params[3])
        # Oscillations
        # "Activity"
        m += lorentzian(new_f, params[7], params[8])
        # Sampling effect
        m *= np.sinc(f / 2.0 / (284.0*2))**2.0
        # Accounting for effect near Nyquist
        m = m[:len(m)//2] + m[len(m)//2:][::-1]
        # White Noise
        m += params[9]
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
    new_f = np.append(f, f+f.max())

    if len(params) > 10:
        # Granulation
        m = np.asarray(models.lorentzian(new_f, params[:,0].ravel(),
                                            params[:,1].ravel()))
        # Granulation
        m += np.asarray(models.lorentzian(new_f, params[:,2].ravel(),
                                             params[:,3].ravel()))
        # Oscillations
        m += np.asarray(models.gaussian(new_f, params[:,6].ravel(),
                                             params[:,4].ravel(),
                                             params[:,5].ravel()))
        # "Activity"
        m += np.asarray(models.lorentzian(new_f, params[:,7].ravel(),
                                             params[:,8].ravel()))
        # Sampling effect
        mm = m[:,:][::-1] * np.sinc(new_f / 2.0 /(284.0))**2.0
        m *= np.sinc(new_f / 2.0 /(284.0))**2.0

        # Accounting for effect near Nyquist
        m += mm
        m = m[:len(m)//2,:]

        # White Noise
        m += params[:,9]
        return m

    else:
        # Granulation
        m = lorentzian(new_f, params[0], params[1])
        # Granulation
        m += lorentzian(new_f, params[2], params[3])
        # Oscillations
        m += gaussian(new_f, params[6], params[4], params[5])
        # "Activity"
        m += lorentzian(fnew_, params[7], params[8])
        # Sampling effect
        mm = m[::-1] * np.sinc(new_f / 2.0 /(284.0))**2.0
        m *= np.sinc(new_f / 2.0 / (284.0))**2.0
        # Accounting for effect near Nyquist
        m += mm
        m = m[::len(m)//2]
        # White Noise
        m += params[9]
        return m
