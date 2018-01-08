# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import mixture
import os

from pylab import *
from matplotlib.colors import colorConverter
import pandas as pd
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

def nonpara_prior(x):

    # Y now equal to all data
    print(x)
    # Only want certain parts of dataframe
    data = x[['hsig1', 'b1',
              'hsig2', 'b2',
              'numax', 'denv', 'Henv',
              'alpha', 'beta']]
    plt.plot(data['numax'], data['hsig2'], '.')
    plt.show()

    data[['b1', 'b2', 'numax', 'denv', 'beta']] = np.log(data[['b1',
                                                               'b2',
                                                               'numax',
                                                               'denv',
                                                               'beta']])
    #data = np.log(x)
    data = data[~np.isnan(data).any(axis=1)]
    print(np.shape(data))
    #x = np.log(x)
    #y = np.log(y)
    #data = np.zeros([len(x), 2])
    #data[:,0] = x
    #data[:,1] = y
    """
    n_comp_max = 30
    n_comp = np.arange(1,n_comp_max+1)
    bics = np.zeros(len(n_comp))
    for idx, i in enumerate(n_comp):
        print("N comps = ", i)
        clf = mixture.GMM(n_components=i, \
                          covariance_type='full')
        #clf = mixture.BayesianGaussianMixture(n_components=i, \
        #                  covariance_type='full')
        clf.fit(data)
        #xx = np.linspace(0.0, x.max(), 1000)
        #yy = np.linspace(y.min(), y.max(), 1000)
        #X, Y = np.meshgrid(xx, yy)
        #XX = np.array([X.ravel(), Y.ravel()]).T
        bics[idx] = clf.bic(data)
        print(clf.bic(data))


    sel = np.where(bics == np.min(bics))
    n_comp = n_comp[sel]

    plt.plot(np.linspace(1, len(bics), len(bics)), bics)
    plt.axvline(n_comp, color='r', linestyle='--')
    plt.show()
    print(n_comp)
    clf = mixture.GMM(n_components=n_comp[0], \
                      covariance_type='full')
    clf.fit(data)
    """
    clf = mixture.BayesianGaussianMixture(n_components=20,
                                          covariance_type='full')
    clf.fit(data)
    print(clf.weights_)

    new_x, Y = clf.sample(int(1e5))
    print(np.shape(new_x))
    new_x = np.exp(new_x)
    #for i in range(10):
    #    #plt.plot(np.exp(data['numax']), np.exp(data['hsig2']), 'x')
    #    plt.plot(new_x[:,4], new_x[:,i], '.')
    #    plt.show()
    joblib.dump(clf, 'James_complete_priors.pkl')
    #xx = np.linspace(x.min(), x.max(), 1000)
    #width = np.linspace(width.min(), width.max(), len(xx))
    #heights = np.linspace(heights.min(), heights.max(), len(yy))

    #yy = np.linspace(y.min(), y.max(), 1000)
    #X, Y = np.meshgrid(xx, yy)
    #XX = np.array([X.ravel(), Y.ravel()]).T
    #if plot:
    #    Z = -clf.score_samples(XX)[0]
    #    Z = Z.reshape(X.shape)

    #    fig, ax = plt.subplots(1)
    #    # Plot data so that it lies on top of lines
    #    levels=np.linspace(0, 10, 10)
        #if (j == 9) and (len(kmag) > 0):
        #    ax.contour(np.exp(X), np.exp(Y), Z, levels, colors='r', alpha=0.4, zorder=4)
        #    CA = ax.scatter(np.exp(x), np.exp(y), s=10, edgecolor='None', c=kmag, cmap='cool', zorder=2)
        #    CB = plt.colorbar(CA, label=r'$K_{p}$')

        #else:
    #    ax.contour(np.exp(X), np.exp(Y), Z, levels, colors='r', alpha=0.4, zorder=4)
    #    CA = ax.scatter(np.exp(x), np.exp(y), s=10, edgecolor='None', c=np.log(white_noise), cmap='cool', zorder=2)
    #    CB = plt.colorbar(CA, label=r'$\ln W$')

    #    ax.set_ylim(np.exp(yy.min()),
    #                       np.exp(yy.max()))
        # Plot priors
    #    ax.set_xlim(np.exp(xx.min()), np.exp(xx.max()))
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    #    ax.set_ylabel(param, fontsize=18)
    #    ax.set_ylabel(param, fontsize=18)
    #    ax.set_xlabel(r"$\nu_{\mathrm{max}}$ ($\mu$Hz)", fontsize=18)
    #    plt.tight_layout()
    #    print("NAMES: ", names)
    #    #plt.savefig(names+'.pdf')
    #    plt.close()
    #for i in np.arange(n_comp):
    #    print(i,clf.means_[i],clf.covars_[i,:,:],clf.weights_[i])
    sys.exit()

def names():
    return ['hsig1', 'b1', 'c1', 'hsig2', 'b2', 'c2',
            'numax', 'denv', 'Henv', 'alpha', 'beta', 'c3',
            'white']#, 'Anu']

def labels():
    return [r"$\sigma_{1}$ (ppm$^{2}\mu$Hz$^{-1}$)",
            r"$b_{1}$ ($\mu$Hz)", \
            r"$\sigma_{2}$ (ppm$^{2}\mu$Hz$^{-1}$)",
            r"$b_{2}$ ($\mu$Hz)", \
            r"$\nu_{\mathrm{max}}$ ($\mu$Hz)",
            r"$\delta\nu_{\mathrm{env}}$ ($\mu$Hz)", \
            r"$H_{\mathrm{env}}$ (ppm$^{2}\mu$Hz$^{-1}$)",
            r"$\alpha$ (ppm$^{2}\mu$Hz$^{-1}$)",
            r"$\beta$ ($\mu$Hz)",
            r"$W$ (ppm$^{2}\mu$Hz$^{-1}$)"]

def plot_data(x, y, xlable="", ylable="", xerr=[], yerr=[]):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', facecolor="None")
    if len(xerr) > 0:
        ax.errorbar(x, y, fmt='k.', xerr=xerr)
    if len(yerr) > 0:
        ax.errorbar(x, y, fmt='k.', yerr=yerr)


def get_data(fitter="James"):
    # Read in file
    if fitter == "Andres":
        sys.exit("Not yet implemented!")
    elif fitter == "James":
        data = pd.read_csv('Processed_summary.csv')
    #data.drop(['KIC', 'Kp'], axis=1, inplace=True)
    labs = ['KIC', 'hsig1', 'b1', 'hsig2', 'b2', 'numax', 'denv', 'Henv', 'alpha', 'beta']
    new_labs = [str(i)+'_perr' for i in labs[1:]]
    new_labs += [str(i)+'_nerr' for i in labs[1:]]
    labs += new_labs
    # Extract labels
    labels = labs #list(data.columns.values)
    print(labels)
    total = len(data)
    data = data[labels]
    print(data.head())
    # Only select "good" data -> numax > 1 and error on numax < 1s
    data = data[data['numax'] > 2.0]
    data = data[(data['numax_perr'] < 1.0) & (data['numax_nerr'] < 1.0)]
    # Don't store errors for the moment!
    print("Good data is {0} out of {1}".format(len(data), total))
    #data[['hsig1', 'hsig2', 'Henv', 'alpha']] = data[['hsig1', 'hsig2', 'Henv', 'alpha']].apply(np.exp)
    #print(data['Henv'])
#    data = data[['hsig1', 'b1', 'hsig2', 'b2', 'numax', 'denv', 'Henv', 'alpha', 'beta']]
    return data, labels


if __name__ == "__main__":

#    data, labels = old_get_data()
    data, labels = get_data()
    #labels = labels()
    #names = names()

    #prior_type = ['normal', 'normal', 'uniform',
    #              'normal', 'normal', 'uniform',
    #              'none', 'normal', 'normal', 'none', 'none', 'uniform', 'none', 'none']

    # Convert amplitudes to log
    #data[sel, 0] = np.log(data[sel, 0])
    #data[sel, 3] = np.log(data[sel, 3])
    #data[sel, 9] = np.log(data[sel, 9])
    # Convert H_env to log
    #data[sel, 8] = np.log(data[sel, 8])


    nonpara_prior(data)
