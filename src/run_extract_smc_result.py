"""
A code to extract results from .h5 db file.

Authors:
    Seungjae Lee

Date:
    01/05/2017
"""

# import libraries
import numpy as np
import pymc as pm
import pysmc as ps
from scipy.misc import logsumexp
from numba import jit
from joblib import Parallel, delayed

import cPickle as pickle
import sys
import argparse
import os
import cluster_model

@jit
def _help_func(res, tmp, ID):
    N = ID.shape[0]
    for n in xrange(N):
        res[ID[n], :] += tmp[n, :]
    return res

def tensor_func(X=None, Y=None, ID=None, mu=None, sigma=None, xi_cluster=None, xi_shared=None):
    # The number of human subjects
    D = ID.max() + 1
    # The number of samples
    N = X.shape[0]
    # The number of clusters we are looking for
    K = mu.shape[0]
    # The number of distict classes
    C = np.unique(Y).max() + 1
    
    ones = np.ones((N, 1))
    X = np.array([np.hstack((cluster_model.pmv_function(xi_cluster=xi_cluster[k,:],xi_shared=xi_shared, X=X)[:,None]/10., ones)) for k in xrange(K)]) # K x N x Nf

    # The number of features
    Nf = X.shape[2]

    i0 = np.repeat(np.arange(N)[:, None], K, axis=1)
    i1 = np.repeat(np.arange(K)[None, :], N, axis=0)
    i2 = np.repeat(Y[:, None], K, axis=1)
    
    W = np.zeros((K, Nf, C))
    for k in xrange(K):
        mu_k = mu[k,:]
        sigma_k = sigma[k]
        sigma2 = sigma_k ** 2.
        beta = mu_k / sigma2
        gamma = -0.5 * mu_k ** 2 / sigma2
        W[k,:,:] = np.vstack((beta,gamma))
    

    
    XW = np.einsum('knf,kfc->nkc',X,W) # N x K x C
            
#    XW = np.dot(X, W)  # N x K x C
    # Get log of softmax on XW
    XW -= logsumexp(XW, axis=2)[:, :, None]
    # Let's turn this into an N x K tensor by selecting only the observed
    # classes
    tmp = XW[i0, i1, i2]
    # Now, for each person, we sum over his observations
    res = np.zeros((D, K))
    return _help_func(res, tmp, ID)
    
# a helper function for reordering
def loop(i, container, mu_sample, sigma_sample, xi_cluster_sample, xi_shared_sample):#, pi_sample):

    ref_z = container[0]
    ncases = container[1]
    X_training = container[2]
    Y_training = container[3]
    occu_ID = container[4]
    permutation = container[5]
    
    tensor = tensor_func(X=X_training, Y=Y_training, ID=occu_ID,
                         mu=mu_sample, sigma=sigma_sample, xi_cluster=xi_cluster_sample, xi_shared=xi_shared_sample)
    D = occu_ID.max() + 1
    i3 = np.arange(D)
    
    # calculate log-probability for each permutation
    logp = np.empty((ncases,))
    for c in xrange(ncases):
        order = permutation[c,:]
#        pi_new = pi_sample[order]
        loglike = tensor[i3,order[ref_z]].sum()
        logp_z=0.
#        for d in xrange(D):
#            logp_z += pm.categorical_like(x = order[ref_z[d]], p = pi_new)
        logp[c] = loglike + logp_z
    res = np.argmax(logp)

#    print logp
#    print res
#    print np.max(logp)
    return res
    
# a function to reorder the labels
def get_SMC_results(db_name, ncpu, data_train):
    print ''
    print '###################################################'
    print 'load SMC results'
    
    out_name = os.path.splitext(db_name)[0]
    
    try: # try importing resampled results from .npz file
        results_org = np.load(out_name+'.npz')
        w = results_org['w']
        mu_samples = results_org['mu_samples']
        sigma_samples = results_org['sigma_samples']
        xi_cluster_samples =results_org['xi_cluster_samples']
        xi_shared_samples = results_org['xi_shared_samples']
        z_samples = results_org['z_samples']
        log_Zs = results_org['log_Zs']
        gammas = results_org['gammas']
        ncluster = results_org['ncluster']
        nclass = results_org['nclass']
        npeople = results_org['npeople']
        nparticles = results_org['nparticles']
        nxic = results_org['nxic']
        nxis = results_org['nxis']
        alpha0_samples = results_org['alpha0_samples']
        nu_samples = results_org['nu_samples']
        pi_samples = results_org['pi_samples']
        print 'resampled original results were loaded from', out_name+'.npz'
        
        with open(out_name+'.pickle', 'rb') as handle:
          results = pickle.load(handle)        
        print 'resampled get_particle_approximation were loaded from', out_name+'.pickle'

    except: # extract results from .h5 file
        print 'load db from .h5'
        db = ps.HDF5DataBase.load(db_name)
        nsteps = db.num_gammas
        results = db.get_particle_approximation(nsteps-1)
        print 'start resampling'
        results.resample()
    
        gammas = db.gammas
        log_Z2_Z1 = db.fd.root.log_Zs[:]
        log_Zs = np.zeros((log_Z2_Z1.shape[0]))
        for i in xrange(log_Z2_Z1.shape[0]):
            log_Zs[i]=log_Z2_Z1[0:i+1].sum()
    
        w = np.exp(results.log_w[:])
        nparticles = w.shape[0]
        
        try:
            nclass = results.mu_0.shape[1]
        except:
            nclass = results.mu.shape[1]
    
        ncluster = 0
        for i in xrange(50):
            try:
                getattr(results, 'mu_%d' %i)
                ncluster = i+1
            except: pass
        if ncluster==0:
            ncluster=1
            
        npeople = 0
        for d in xrange(3000):
            try:
                getattr(results, 'z_%d' %d)
                npeople = d+1
            except: pass
        
        
        try:
            nxic = results.xi_cluster_0.shape[1]
        except:
            nxic = results.xi_cluster.shape[1]
        nxis = results.xi_shared.shape[1]

        print 'the number of classes: ', nclass
        print 'the number of clusters: ', ncluster
        print 'the number of particles: ', nparticles
        print 'the number of people: ', npeople
        print 'the number of xi_cluster: ', nxic
        print 'the number of xi_cluster: ', nxis
    
        mu_samples = np.zeros((ncluster, nparticles, nclass))
        sigma_samples = np.zeros((ncluster, nparticles))
        z_samples = np.zeros((npeople, nparticles), dtype=int)
        xi_cluster_samples = np.zeros((ncluster, nparticles, nxic))
        xi_shared_samples = np.zeros((ncluster, nparticles, nxis))
        alpha0_samples = np.zeros((nparticles,))
        nu_samples = np.zeros((nparticles, ncluster))
        pi_samples = np.zeros((nparticles, ncluster))
    
        if ncluster > 1:
            for d in xrange(npeople):
                z_samples[d,:] = getattr(results, 'z_%d' %d)
            for k in xrange(ncluster):
                mu_samples[k,:,:] = getattr(results, 'mu_%d' %k)
                sigma_samples[k,:] = getattr(results, 'sigma_%d' %k)
                xi_cluster_samples[k,:,:] = getattr(results, 'xi_cluster_%d' %k)
        else:
            mu_samples[0,:,:] = getattr(results, 'mu')
            sigma_samples[0,:] = getattr(results, 'sigma')
            xi_cluster_samples[0,:,:] = getattr(results, 'xi_cluster')
        xi_shared_samples = getattr(results, 'xi_shared')
                        
        try:
            alpha0_samples = getattr(results, 'alpha0')
            for k in xrange(ncluster):
                nu_samples[:,k] = getattr(results, 'nu_%d' %k)    
            for i in xrange(nparticles):
                pi_samples[i,:] = pi_function(K=ncluster, nu = nu_samples[i,:])
                pi_samples[i,:] = 1. / pi_samples[i,:].sum() * pi_samples[i,:]
            DP=True
        except:
            DP=False 
        results_org = {'w': w,
                       'mu_samples': mu_samples,
                       'sigma_samples': sigma_samples,
                       'xi_cluster_samples': xi_cluster_samples,
                       'xi_shared_samples': xi_shared_samples,
                       'z_samples': z_samples,
                       'alpha0_samples': alpha0_samples,
                       'nu_samples': nu_samples,
                       'pi_samples': pi_samples,
                       'log_Zs': log_Zs,
                       'gammas': gammas,
                       'ncluster': ncluster,
                       'nclass': nclass,
                       'npeople': npeople,
                       'nparticles': nparticles,
                       'nxic': nxic,
                       'nxis': nxis}
        
        np.savez_compressed(out_name,**results_org)
        print 'resampled original results are saved in', out_name+'.npz'
        with open(out_name+'.pickle', 'wb') as handle:
            pickle.dump(results, handle)
        print 'resampled get_particle_approximation are saved in', out_name+'.pickle'
    print '###################################################'
    print ''
    print '###################################################'
    print 'start relabeling'
    
    if ncluster==1:
        print "1 cluster model does not require relabeling"
        return 0
    # import data
    DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', data_train)
    X_training, Y_training, occu_ID = cluster_model.load_training_data(DATA_FILE)
#
    # load the model        
    model = pm.MCMC(cluster_model.make_model_cluster_pmv(X_training, Y_training, occu_ID, num_clusters=ncluster))
    ncases, permutation = permutation_func(ncluster)
    logp = np.empty((nparticles,))
    print 'initial logp:', model.logp
    print 'initial loglike:', sum([model.y_obs[d].logp for d in xrange(npeople)]) #model.y_obs.logp

    for i in xrange(nparticles):
        for a in model.stochastics:
            a.value=getattr(results, a.__name__)[i]
        try:
            logp[i] = model.logp
        except:
            logp[i] = -np.inf
    ref = np.argmax(logp)
    print 'index number of a reference particle:', ref
    for a in model.stochastics:
        a.value=getattr(results, a.__name__)[ref]
    print 'logp of the reference particle:', model.logp
    print 'initial loglike:', sum([model.y_obs[d].logp for d in xrange(npeople)]) #model.y_obs.logp
    
    ref_z = np.zeros((npeople,), dtype=int)
    for a in model.stochastics:
        if a.__name__[:2]=='z_':
            idx = int(a.__name__[2:])
            value = int(a.value)
            ref_z[idx] = value

    if DP == True:
        print "This code does not support relabeling of DP model"
        return 0             
    container = [ref_z, ncases, X_training, Y_training, occu_ID, permutation]
    backend = 'multiprocessing'
    print 'work with multiprocessing mode'

    new_order = Parallel(n_jobs=ncpu, verbose=5, backend=backend)(
                         delayed(loop)(
                         i, container,
                         mu_samples[:,i,:],
                         sigma_samples[:,i],
                         xi_cluster_samples[:,i,:],
                         xi_shared_samples[i,:]) for i in xrange(nparticles))
    new_order = np.array(new_order)
        
    mu_samples_reordered = np.zeros((ncluster, nparticles, nclass))
    sigma_samples_reordered = np.zeros((ncluster, nparticles))
    xi_cluster_samples_reordered = np.zeros((ncluster, nparticles, nxic))
    xi_shared_samples_reordered = xi_shared_samples
    z_samples_reordered = np.zeros((npeople, nparticles))
    nu_samples_reordered = np.zeros((nparticles, ncluster))
    pi_samples_reordered = np.zeros((nparticles, ncluster))
        
    for i in xrange(nparticles):
        for k in xrange(ncluster):
            new = permutation[new_order[i],:][k]
            mu_samples_reordered[k,i,:] = mu_samples[new,i,:]
            sigma_samples_reordered[k,i] = sigma_samples[new,i]
            xi_cluster_samples_reordered[k,i,:] = xi_cluster_samples[new,i,:]
            tmp_idx = z_samples[:,i]==new
            z_samples_reordered[tmp_idx,i] = k
            nu_samples_reordered[i,k] = nu_samples[i,new]
            pi_samples_reordered[i,k] = pi_samples[i,new]
            
    results_relabeled = {'w': w,
                         'mu_samples_reordered': mu_samples_reordered,
                         'sigma_samples_reordered': sigma_samples_reordered,
                         'xi_cluster_samples_reordered': xi_cluster_samples_reordered,
                         'xi_shared_samples_reordered': xi_shared_samples_reordered,
                         'z_samples_reordered': z_samples_reordered,
                         'pi_samples_reordered': pi_samples_reordered,
                         'nu_samples_reordered': nu_samples_reordered,
                         'alpha0_samples': alpha0_samples,
                         'log_Zs': log_Zs,
                         'gammas': gammas,
                         'ncluster': ncluster,
                         'nclass': nclass,
                         'npeople': npeople,
                         'nparticles': nparticles,
                         'new_order': new_order}

    np.savez_compressed(out_name+'_relabled',**results_relabeled)  
    print 'resampled & relabeled results are saved in', out_name+'_relabled.npz'
    print '###################################################'
    return 0

def permutation_func(ncluster):
    print "relabel" + str(ncluster) + "clusters"
    pool = range(ncluster)
    from itertools import permutations
    permutation = np.array(list(permutations(pool,ncluster)))
    print permutation.shape
    ncases = permutation.shape[0]
    return ncases, permutation
    
def pi_function(K=None, nu=None):
    pi = np.zeros((K,))
    tmp = 1.
    for i in xrange(K):
        if i!=0:
            tmp = tmp * (1. - nu[i-1])
            pi[i] = nu[i] * tmp
        else:
            pi[i] = nu[i]
    return pi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=str, default='smc.h5',
                        help='the file where the MCMC chain is written')
    parser.add_argument('--ncpu', type=int, default=1,
                        help='specify the number of cpus')
    parser.add_argument('--data-train', type=str, default='SynData.npz',
                        help='specify the training data used')
    args = parser.parse_args()

    get_SMC_results(args.file, args.ncpu, args.data_train)
    print 'Done..'