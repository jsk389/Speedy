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
import scipy.misc as misc

import numdifftools as ndt

def rebin(f, smoo):
    if smoo < 1.5:
        return f
    smoo = int(smoo)
    m = len(f) // smoo
    ff = np.median(f[:m*smoo].reshape((m,smoo)), axis=1)
    return ff

def lorentzian(f, hsig, hfreq):
    return hsig / (1 + (f/hfreq)**4)

def gaussian(f, henv, numax, width):
    tmp = width / (2.0 * np.sqrt(2.0 * np.log(2)))
    return henv * np.exp(-(f - numax)**2 / (2.0 * tmp**2))

#@profile
def model_wo(f, params):

    if len(params) > 10:
        # Granulation
        m = np.asarray(models.lorentzian(f, params[:,0].ravel(),
                                            params[:,1].ravel()))
        # Granulation
        m += np.asarray(models.lorentzian(f, params[:,2].ravel(),
                                             params[:,3].ravel()))
        # Oscillations
        # "Activity"
        m += np.asarray(models.lorentzian(f, params[:,7].ravel(),
                                             params[:,8].ravel()))
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[:,9]
        return m
    else:
        # Granulation
        m = lorentzian(f, params[0], params[1])
        # Granulation
        m += lorentzian(f, params[2], params[3])
        # Oscillations
        # "Activity"
        m += lorentzian(f, params[7], params[8])
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[9]
        return m

#@profile
def model(f, params):

    if len(params) > 10:
        # Granulation
        m = np.asarray(models.lorentzian(f, params[:,0][:,None],
                                            params[:,1][:,None]))
        # Granulation
        m += np.asarray(models.lorentzian(f, params[:,2][:,None],
                                             params[:,3][:,None]))
        # Oscillations
        m += np.asarray(models.gaussian(f, params[:,6][:,None],
                                             params[:,4][:,None],
                                             params[:,5][:,None]))
        # "Activity"
        m += np.asarray(models.lorentzian(f, params[:,7][:,None],
                                             params[:,8][:,None]))
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[:,9][:,None]
        return m

    else:
        # Granulation
        m = lorentzian(f, params[0], params[1])
        # Granulation
        m += lorentzian(f, params[2], params[3])
        # Oscillations
        m += gaussian(f, params[6], params[4], params[5])
        # "Activity"
        m += lorentzian(f, params[7], params[8])
        # Sampling effect
        m *= np.sinc(f / 2.0 / 284.0)**2.0
        # White Noise
        m += params[9]
        return m
    """

    """


def lnlike_wo(f, p, smoo, params):

    if len(params) > 10:
        mod = model_wo(f, params.reshape((np.shape(params)[0], np.shape(params)[1],1)))
        #sinc = np.sinc(f / 2.0 / 284.0)**2.0
        #mod = models.model_wo(f, params, sinc)
        ll = -1.0 * np.sum(np.log(mod) + (p / mod), axis=1) * smoo
        ll = ll[~np.isnan(ll)]
        # Select parameters with max log-like and return those instead
        #idx = (ll== ll.max())
        ind = np.argpartition(ll, -10)[-10:]
        best_params = params[ind,:]

        #return_dict['wo'] = ll
        return (ll.max(), best_params)
    else:
        mod = model_wo(f, params)
    #plt.plot(f, p, 'k')
    #plt.plot(f, mod, 'r')
    #plt.show()
        ll = -1.0 * np.sum(np.log(mod) + (p / mod))
        print(np.shape(mod), np.shape(ll))
        if not np.isfinite(ll):
            return -np.inf
        return -1.0 * np.sum(np.log(mod) + (p / mod)) * smoo

def combo_like(*obj_main):

    f, smoo, p, w, wo = obj_main[0]
    sys.exit()

    ll1 = lnlike(f, p, smoo, w)
    ll2 = lnlike(f, p, smii, wo)
    return (ll1, ll2)

#@profile
def lnlike(params, f, p, smoo):
    #print(len(params), np.shape(params))
    if len(params) > 10:
        mod = model(f, params.reshape((np.shape(params)[0], np.shape(params)[1])))
        #print(np.shape(mod))
        #sinc = np.sinc(f / 2.0 / 284.0)**2.0
        #Y = np.zeros([len(params), len(f)])
        #mod = model(Y, f, params, sinc)
        ll = np.sum(np.log(mod) + (p / mod), axis=1) * smoo
        #ll = ll[~np.isnan(ll)]
        # Select parameters with max log-like and return those instead
        #idx = (ll == ll.max())
        ind = np.argpartition(ll, 100)[:100]
        #ind = (ll == ll.min())
        best_params = params[ind,:].squeeze()
        #return_dict['w'] = ll
        return ll, ind, best_params #(ll.max(), best_params)
    else:
        mod = model(f, params)
    #plt.plot(f, p, 'k')
    #plt.plot(f, mod, 'r')
    #plt.show()
        ll = np.sum(np.log(mod) + (p / mod))
        #print(np.shape(mod), np.shape(ll))
        if not np.isfinite(ll):
            return 1e30 #np.inf
        return np.sum(np.log(mod) + (p / mod)) * smoo


def lnpriors(priors, params):
    lnprior = 0
    for idx, i in enumerate(priors):
        lnprior += i.score_samples([p[0], p[idx+1]])[0]
    return lnprior

def lnprob(f, p, priors, smoo, params):
    likelihood = lnlike(f, p, smoo, params)
    lnp = lnpriors(priors, params)
    if not np.isfinite(lnp):
        return -np.inf
    return likelihood + lnp

def compute_Bayes(BIC, BIC_wo, BIC_white):
    """
    Compute Bayes factor for H1 i.e. there are oscillations
    """
    from scipy.misc import logsumexp
    lnprob = -0.5*BIC - np.logaddexp(-0.5*BIC, -0.5*BIC_wo)
    # BIC of H1 - BIC H0
    # larger value favours H1
    logBayes = 0.5 * (-1.0*BIC + BIC_wo)
    #lnprob = np.log(1./3.) - 0.5*BIC - logsumexp([BIC, BIC_wo, BIC_white])
    #print(np.log(1./3.), - 0.5*BIC, - logsumexp([BIC, BIC_wo, BIC_white]))
    logprob = logBayes - logsumexp([logBayes, 1.])
    #print("2lnK: ", 2.0*logBayes)
    lnprob_w = -0.5 * BIC - logsumexp([-0.5*BIC, -0.5*BIC_wo, -0.5*BIC_white])
    lnprob_wo = -0.5 * BIC_wo - logsumexp([-0.5*BIC, -0.5*BIC_wo, -0.5*BIC_white])
    lnprob_white = -0.5 * BIC_white - logsumexp([-0.5*BIC, -0.5*BIC_wo, -0.5*BIC_white])
    #print(0.5 * (BIC_wo - BIC))
    #prob = np.exp(-0.5*BIC) / (np.exp(-0.5*BIC) + np.exp(-0.5*BIC_wo))
    return np.exp(lnprob_w), np.exp(lnprob_wo), np.exp(lnprob_white)

def flatten(iter_lst):
    return list(itertools.chain(*iter_lst))

def perform_inference(f, p, ff, pp, smoo, epic, plot=[], save=[]):

    # first_point = []
    n_bootstrap = 1
    nsamps = int(1e5)
    params = np.zeros([n_bootstrap, 10])
    white = np.median(pp[-100:]) / 0.702 / 1.2
    print("WHITE: ", white)
    proper_samples_full, _ = x.sample(nsamps*n_bootstrap)
    proper_samples_full = np.exp(proper_samples_full)

    #for i in range(10):
    #    plt.plot(proper_samples_full[:,4], proper_samples_full[:,i], '.')
    #    plt.show()

    #proper_samples_full = proper_samples_full[proper_samples_full[:,4] < 283.0]
    proper_samples_full = np.c_[proper_samples_full,
                                np.ones(len(proper_samples_full))*white]
    #proper_samples_full_wo = proper_samples_full.copy()
    #proper_samples_full_wo[:,8] = 0

    like, ind, best_params = lnlike(proper_samples_full, f, p, smoo)
    plt.plot(proper_samples_full[:,4], like, '.')
    plt.plot(proper_samples_full[ind,4], like[ind], 'r.')
    plt.show()

    #print(np.shape(np.median(best_params, axis=0)))

    #Hfun = ndt.Hessian(lnlike)#, full_output=True)
    Hfun = ndt.Hessdiag(lnlike)
    hessian_ndt = Hfun(np.median(best_params, axis=0), f, p, smoo)
    print(np.diag(hessian_ndt))
    se = np.sqrt(np.diag(np.linalg.inv(np.diag(hessian_ndt))))
    #print(best_params, axis=0)
    print(se)
    print("NUMAX: {} +/- {}".format(np.median(best_params, axis=0)[4],
                                    se[4]))
    print("SHAPE: ", np.shape(hessian_ndt))


    plt.plot(ff, pp, 'k')
    for i in range(np.shape(best_params)[0]):
        plt.plot(ff, model(ff, best_params[i,:]), color='g', alpha=0.2, lw=2)
    plt.plot(ff, model(ff, np.median(best_params, axis=0)), color='r', lw=2)
    print(np.median(best_params, axis=0))
    print(np.std(best_params, axis=0))
    plt.show()
    sys.exit()


    s = time.time()

    jobs = []
    pipe_list = []
    pipe_list_wo = []
    pool = multiprocessing.Pool()
    for j in range(n_bootstrap):
        #print(j)
        recv_end, with_send = multiprocessing.Pipe(False)
        samps = proper_samples_full[j*nsamps:(j+1)*nsamps, :]
        samps_wo = proper_samples_full_wo[j*nsamps:(j+1)*nsamps, :]
        #idx = np.where(samps[:,6] < 300.0)
        #samps = samps[idx]
        #idx = np.where(samps_wo[:,6] < 300.0)
        #samps_wo = samps_wo[idx]

        proc = multiprocessing.Process(target=lnlike, args=(f, p, smoo,
                                       samps,
                                       with_send))

        recv_endo, with_sendo = multiprocessing.Pipe(False)
        proc2 = multiprocessing.Process(target=lnlike_wo, args=(f, p, smoo,
                                       samps_wo,
                                       with_sendo))
        #proc.daemon = True
        jobs.append(proc)
        jobs.append(proc2)
        pipe_list.append(recv_end)
        pipe_list_wo.append(recv_endo)
        proc.start()
        proc2.start()

    # Wait for all worker processes to finish
    for procs in jobs:
        procs.join()

    # Extract all data
    result_list_w = [x.recv() for x in pipe_list]
    result_list_wo = [x.recv() for x in pipe_list_wo]
    like_w = np.array(flatten([result_list_w[i][0] for i in range(len(result_list_w))]))
    params_w = np.array(flatten([result_list_w[i][1] for i in range(len(result_list_w))]))
    like_wo = np.array(flatten([result_list_wo[i][0] for i in range(len(result_list_wo))]))
    params_wo = np.array(flatten([result_list_wo[i][1] for i in range(len(result_list_wo))]))
    #print("Time taken (s): ", time.time()-s)
    print("NUMAX EST: {0} +/- {1}".format(np.mean(params_w[:,6]), np.std(params_w[:,6])))

    #print(np.mean(params_w, axis=0))
    #print(np.std(params_w, axis=0))

    best_params = np.median(params_w, axis=0)
    best_params_wo = np.median(params_wo, axis=0)
    like_w = lnlike(f, p, smoo, best_params, [])
    like_wo = lnlike_wo(f, p, smoo, best_params_wo, [])
    best_params_white = best_params.copy()
    best_params_white[:] = 0
    best_params_white[-1] = np.median(p)# * 1.402
    #print("WHITE EST: ", best_params_white[-1])
    like_white = lnlike(f, p, smoo, best_params_white, [])


    BIC = -2.0 * like_w + 13 * np.log(len(f))
    BIC_wo = -2.0 * like_wo + 10 * np.log(len(f))
    BIC_white = -2.0 * like_white + 1.0 * np.log(len(f))

    #print(like_w, like_wo)
    #print("LIKE RATIO: ", like_w-like_wo)
    #sys.exit()

    #idx = (like_w == like_w.max())
    #paramsw = params_w[idx,:][0]
    #idxw = (like_wo == like_wo.max())
    #params_wo = params_wo[idxw,:][0]


    # Set up flags for detection tests - set to 'N' by default
    FLAG_WHITE = 'N'
    pH1_WHITE = 0
    FLAG_GRAN = 'N'
    pH1_GRAN = 0
    FLAG_OSC = 'N'
    pH1_OSC = 0

    # Convert bic difference to Bayes factor of null hypothesis
    # BIC calculated to find out if H1 favoured over H0 (oscillations over none)
    print(BIC, BIC_wo, BIC_white)
    pH0, pH1, pH2 = compute_Bayes(BIC, BIC_wo, BIC_white)
    print("Prob Osc: ", pH0)
    print("Prob Gran: ", pH1)
    print("Prob White: ", pH2)
    # Check to see which if white noise if favoured
    if (pH2 > pH0) and (pH2 > pH1):
        print("No oscillations! Consistent with white noise")
        final_params = best_params_white
        errors = np.zeros(len(final_params))
        # If it is favoured compare to probability that there are poscillations
        #pH1 = compute_Bayes(BIC_w, BIC_white)
        #if pH2 > 0.5:
        FLAG_WHITE = 'Y'
        pH1_WHITE = pH2
        pH1_OSC = pH0
        pH1_GRAN = pH1

        #print("Posterior prob: ", pH1)
    else:
        # Otherwise compare against presence of granulation only and granulation
        # plus oscillations
        #pH1 = compute_Bayes(BIC, BIC_wo, BIC_white)
        #print("Posterior prob: ", pH1)
        #sys.exit()

        if (pH1 > pH0) and (pH1 > pH2):
            print("No oscillations but granulation!")
            # Set flags and probabilities
            FLAG_GRAN = 'Y'
            FLAG_OSC = 'N'
            pH1_WHITE = pH2
            pH1_OSC = pH0
            pH1_GRAN = pH1
            # Parameters
            final_params = best_params_wo
            errors = np.std(params_wo, axis=0)
            #params = params_wo[like_wo == like_wo.max(), :].reshape((-1,1))
            #params_b = params_w[like_w == like_w.max(), :].reshape((-1,1))
            print("NUMAX BEST GUESS: ", final_params[6])

            #params = best_params_wo
            #params_b = best_params
        elif (pH0 > pH1) and (pH0 > pH2):
            print("Oscillations!")
            # Set flags
            FLAG_GRAN = 'N'
            FLAG_OSC = 'Y'
            pH1_WHITE = pH2
            pH1_OSC = pH0
            pH1_GRAN = pH1
            # Parameters
            final_params = best_params
            errors = np.std(params_w, axis=0)
            #params = params_w[like_w == like_w.max(), :].reshape((-1,1))
            #params_b = params_wo[like_wo == like_wo.max(), :].reshape((-1,1))
            print("NUMAX BEST GUESS: ", final_params[6])
            #params = best_params
            #params_b = best_params_wo
        else:
            print("NUTS!")
    """
    # Check to see which if white noise if favoured
    if (BIC_white < BIC) and (BIC_white < BIC_wo):
        print("No oscillations! Consistent with white noise")
        final_params = best_params_white
        errors = np.zeros(len(final_params))
        # If it is favoured compare to probability that there are poscillations
        pH1 = compute_Bayes(BIC_w, BIC_white)
        if pH1 < 0.5:
            FLAG_WHITE = 'Y'
            pH1_WHITE = 1.0 - pH1
            pH1_OSC = pH1

        print("Posterior prob: ", pH1)
    else:
        # Otherwise compare against presence of granulation only and granulation
        # plus oscillations
        pH1 = compute_Bayes(BIC, BIC_wo, BIC_white)
        print("Posterior prob: ", pH1)
        #sys.exit()

        if pH1 < 0.5:
            print("No oscillations!")
            # Set flags
            FLAG_GRAN = 'Y'
            pH1_GRAN = 1.0 - pH1
            FLAG_OSC = 'N'
            pH1_OSC = pH1
            # Parameters
            final_params = best_params_wo
            errors = np.std(params_wo, axis=0)
            #params = params_wo[like_wo == like_wo.max(), :].reshape((-1,1))
            #params_b = params_w[like_w == like_w.max(), :].reshape((-1,1))
            print("NUMAX BEST GUESS: ", final_params[6])

            #params = best_params_wo
            #params_b = best_params
        else:
            print("Oscillations!")
            # Set flags
            FLAG_GRAN = 'N'
            pH1_GRAN = 1.0 - pH1
            FLAG_OSC = 'Y'
            pH1_OSC = pH1
            # Parameters
            final_params = best_params
            errors = np.std(params_w, axis=0)
            #params = params_w[like_w == like_w.max(), :].reshape((-1,1))
            #params_b = params_wo[like_wo == like_wo.max(), :].reshape((-1,1))
            print("NUMAX BEST GUESS: ", final_params[6])
            #params = best_params
            #params_b = best_params_wo

        #plt.hist(params_w[:,0], histtype='step', normed=True, color='b')
        #plt.axvline(params[0], color='r', lw=2)
        #plt.show()
    """

    if plot:
        plt.plot(ff, pp, 'k')
        plt.plot(f, p, 'b')
        if FLAG_OSC == 'Y':
            #plt.plot(ff, model(ff, params), 'r', lw=2)
            for i in range(len(params_w)):
                plt.plot(ff, model(ff, params_w[i,:]), color='g', alpha=0.2)
            plt.plot(ff, model(ff, final_params), color='r', lw=2)
        elif FLAG_GRAN == 'Y':
            for i in range(len(params_w)):
                plt.plot(ff, model_wo(ff, params_w[i,:]), color='g', alpha=0.2)
            plt.plot(ff, model(ff, np.median(params_w, axis=0)), color='r', linestyle='--', lw=2)
            plt.plot(ff, model_wo(ff, final_params), color='r', lw=2)
        elif FLAG_WHITE == 'Y':
            #plt.plot(ff, model(ff, final_params), color='r', lw=2)
            pass
        plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
        plt.ylabel(r'PSD (ppm$^{2}\mu$Hz$^{-1}$)', fontsize=18)
        plt.xlim(1.0, 283.0)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    if save:
        #plt.savefig('/home/jsk389/Dropbox/Python/K2-GAP/C1-results/'+str(epic)+'_est.png')
        #plt.show()
        plt.close()
        # Setup headers
        labels = ['hsig1', 'hsig1err', 'b1', 'b1err', 'c1', 'c1err',
                  'hsig2', 'hsig2err', 'b2', 'b2err', 'c2', 'c2err',
                  'numax', 'numaxerr', 'denv', 'denverr', 'Henv', 'Henverr',
                  'alpha', 'alphaerr', 'beta', 'betaerr', 'c3', 'c3err',
                  'white', 'whiteerr',
                  'FLAG_OSC', 'pH1_OSC', 'FLAG_GRAN', 'pH1_GRAN', 'FLAG_WHITE', 'pH1_WHITE']
        # Interleave values and error arrays
        values = np.vstack((best_params,errors)).reshape((-1,), order='F')
        values = np.append(values, (FLAG_OSC, pH1_OSC, FLAG_GRAN, pH1_GRAN, FLAG_WHITE, pH1_WHITE))
        outputs = np.vstack([labels, values])
        #np.savetxt('/home/jsk389/Dropbox/Python/K2-GAP/C1-results/'+str(epic)+'_diags.txt', outputs, fmt="%s")

def read_data(fname):
    """
    Read in data
    """
    if fname.endswith(".pow"):
        f, p = np.loadtxt(fname, unpack=True)
    elif fname.endswith(".fits"):
        FITSfile=pyfits.open(fname)
        topheader=FITSfile[0].header
        dataheader=FITSfile[1].header
        kic=topheader['keplerid']
        data=FITSfile[1].data
        f,p=data['frequency'],data['psd']
    return f, p

def prepare_data(f, p):
    """
    Prepare data for analysis
    """
    ff = f.copy()
    pp = p.copy()
    p = p[f > 2.0]
    f = f[f > 2.0]

    smoo = int(0.5 / (f[1]-f[0]))
    p = rebin(p, smoo)
    f = rebin(f, smoo)
    return f, p, ff, pp, smoo

def extract_epic(fname):
    epic = fname.split('/')[-1].split('_')[0].lstrip('ktwo')
    return epic

if __name__ == "__main__":


    x = joblib.load('../Extract-Priors/James_complete_priors.pkl')
    import time

    # Arrange samples into N x num_parameters array
    f, p = np.loadtxt('../../../../KOI-3890/PDCspec8564976.pow', unpack=True)
    f, p = np.loadtxt('../../../../KOI-3890/kplr006448890_kasoc-psd_llc_v3.pow', unpack=True)
    f, p = np.loadtxt('../../../../KOI-3890/kplr010683647_kasoc-psd_llc_v1.pow.txt', unpack=True)
    #f, p = np.loadtxt('../../../../KOI-3890/kplr006144777_kasoc-psd_llc_v1.pow.txt', unpack=True)
    #f, p = np.loadtxt('../../../../KOI-3890/kplr008611114_kasoc-psd_llc_v1.pow.txt', unpack=True)
    #p = p[:len(f)//2]
    #f = f[:len(f)//2]
    #p *= 1e6


    # Prep data
    f, p, ff, pp, smoo = prepare_data(f, p)

    # Perform inference
    perform_inference(f, p, ff, pp, smoo, 0, plot=True, save=True)
    #sys.exit()

    sys.exit()

    #fname = '/home/jsk389/Dropbox/Mike/PSD_for_analysis/ktwo215745876_kasoc-psd_llc_v1.pow'
    #f, p = np.loadtxt('/home/jsk389/Dropbox/Mike/PSD_for_analysis/ktwo215745876_kasoc-psd_llc_v1.pow', unpack=True)
    #f, p = np.loadtxt(fname, unpack=True)

    import pyfits
    import glob
    #fname = '/home/jsk389/Dropbox/Python/Importance-Sampling/ktwo201652583_kasoc-psd_llc_v1.fits'
    #fname = 'ktwo201121245_kasoc-psd_llc_v1.fits'
    #fname = 'ktwo201126368_kasoc-psd_llc_v1.fits'
    #fname = 'ktwo201128834_kasoc-psd_llc_v1.fits'
    #fnames = glob.glob('/home/jsk389/Dropbox/Data/K2/C1-Stello/*.fits')#[55:]
    #fnames = ['/home/jsk389/Dropbox/Data/K2/C1-Stello/ktwo201127270_kasoc-psd_llc_v1.fits']
    #fname = '/home/jsk389/Dropbox/Python/Importance-Sampling/ktwo201690230_kasoc-psd_llc_v1.fits'

    for idx, i in enumerate(fnames):
        print("Analysing star {0} of {1}".format(idx+1, len(fnames)))
        # Read in data
        f, p = read_data(i)

        # Extract epic
        epic = extract_epic(i)
        print("EPIC: ", epic)
        #epic = ''
        # Prep data
        f, p, ff, pp, smoo = prepare_data(f, p)

        # Perform inference
        perform_inference(f, p, ff, pp, smoo, epic, plot=True, save=True)
        #sys.exit()
