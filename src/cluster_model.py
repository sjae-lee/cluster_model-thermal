"""
A model of thermal preference with clusters.

Authors:
    Seungjae Lee
    Ilias Bilionis

Date:
    01/05/2017
"""
import numpy as np
import math
import pymc as pm
import pysmc as ps
from scipy.misc import logsumexp
import os


__all__ = ['DATA_FILE',
           'load_training_data',
           'pmv_function',
           'loglike_of_mlr',
           'make_model_cluster_pmv',
           'make_model_nocluster_pmv']


# The training data file.
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'SynData.npz')

def pmv_function (xi_shared, xi_cluster, X):
    """
    This function calculates thermal load with input 'X' and parameters 'xi_cluster' and 'xi_shared'
    
    :param X: the input data (nd x 6)
         X[i,0] = ta:   air temperature [deg C]
         X[i,1] = tr:   mean radiant temperature [deg C]
         X[i,2] = vel:  relative air velocity [m/s]
         X[i,3] = rh:   relative humidity [%], used only this way to input humidity level
         X[i,4] = met:  metabolic rate [met]
         X[i,5] = clo:  clothing insulation [clo]
    :param xi_shared:  3-dim vector
         xi_shared[0]:    a parameter adjusting the radiative heat tranfer coefficient,          xi_1 in the paper
         xi_shared[1]:    a parameter adjusting the natural convective heat tranfer coefficient, xi_2 in the paper
         xi_shared[2]:    a parameter adjusting the forced convective heat tranfer coefficient,  xi_3 in the paper
    :param xi_cluster: 2-dim vector
         xi_cluster[0]:   a parameter adjusting the clothing surface temperature,                xi_4 in the paper
         xi_cluster[1]:   a parameter adjusting the total heat loss,                             xi_5 in the paper
    :returns: the thermal load (nd-dim vector)
    """
    nd = X.shape[0]
    pmv = np.zeros(nd)
    for i in xrange(nd):
        # assign inputs
        ta = X[i,0]
        tr = X[i,1]
        vel = X[i,2]
        rh = X[i,3]
        met = X[i,4]
        clo = X[i,5]
        #water vapor pressure in Pa
        pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))
        icl = .155 * (clo) #thermal insulation of the clothing in M2K/W
        m = met * 58.15
        mw = m
        if (icl < .078):
            fcl = 1.0 + 1.29 * icl
        else:
            fcl = 1.05 + .645 * icl        
        hcf = xi_shared[2] * 12.1 * (vel)**0.5    
        taa = ta + 273.0 #air temperature in kelvin
        tra = tr + 273.0 #mean radiant temperature in kelvin
        # calculate surface temperature of clothing by iteration
        #first guess for surface temperature of clothing
        tcla = taa + (35.5-ta) / (3.5 * (6.45 * icl + .1))
        p1 = icl * fcl #calculation term
        p2 = p1 * 3.96 * xi_shared[0]
        p3 = p1 * 100.0 #calculation term
        p4 = p1 * taa #calculation term
        p5 = 308.7 - xi_cluster[0] - .028 * mw + p2 * (tra / 100.0)**4.0 #calculation term
        xn = tcla / 100.0
        xf = tcla / 50.0
        n = 0 #N: number of iteration
        eps = .00015 #stop criteria in iteration
        hc = 1.
        while abs(xn-xf) > eps:
            xf = (xf + xn) / 2.
            #heat transf. coeff. by natural convection
            hcn = xi_shared[1] * 2.38 * abs(100. * xf - taa)**.25        
            if (hcf > hcn):
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * xf**4.) / (100. + p3 * hc)
            n += 1
            if n > 150:
                print 'Exceed Max. Iteration!'
                return np.ones(nd)*999
        tcl = 100.0 * xn - 273.0 #surface temperature of clothing
        # heat loss components
        #heat loss diff. through skin
        hl1 = 3.05 * .001 * (5733.0 - 6.99 * mw - pa)
        #heat loss by sweating    
        if (mw > 58.15):
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0.0
        hl3 = 1.7 * .00001 * m * (5867.0 - pa) #latent respiration heat loss
        hl4 = .0014 * m * (34.0 - ta) #dry respiration heat loss
        hl5 = 3.96 * xi_shared[0] * fcl * (xn**4.0 - (tra/100.0)**4.0)    
        hl6 = fcl * hc * (tcl - ta) #heat loss by convection
        # calculate the thermal load
        ts = 1.
        pmv[i] = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6 - (xi_cluster[1] * (mw - 58.15))) #the thermal load
    return pmv
    
def load_training_data(data_file=DATA_FILE):
    """
    Load the training data.

    :param data_file: The file containing the training data.
        ASHRAE_training.npz: the subset of ASHRAE DB, see section 2.6 in the paper
        SynData: the synthetic dataset, see section 2.4 in the paper
        
    The data file should include 'X_training', 'Y_training', and 'occu_ID'.
        X_training: the input data for model training, (N x 6)
        Y_training: the output data for model training, thermal preference votes, (N x 1)
        occu_ID: an array indicating from which occupant each row of X_training and Y_training were collected (N x 1)
    """
    dataset = np.load(data_file)
    X_training = dataset['X_training']
    Y_training = dataset['Y_training'].flatten()
    occu_ID = dataset['occu_ID'].flatten()
    return X_training, Y_training, occu_ID

    
# A function to calculate the log likelihood of multinomial logistic regression for all class labels.
def loglike_of_mlr(X, mu, sigma):
    """
    Returns the log likelihood of multinomial logistic regression for all class labels.
    
    :param X: The observed features  (ns x nf)
    :param W: The weights of all classes (list of nc - 1 arrays of size nf)
    :returns: An array of size ns x nc.
    """
    mu0 = mu[0]
    mu1 = mu[1]
    mu2 = mu[2]
    sigma2 = sigma ** 2.
    gamma0 = -0.5 * mu0 ** 2 / sigma2
    beta0 = mu0 / sigma2
    gamma1 = -0.5 * mu1 ** 2 / sigma2
    beta1 = mu1 / sigma2
    gamma2 = -0.5 * mu2 ** 2 / sigma2 
    beta2 = mu2 / sigma2
    W = np.array([[beta0, gamma0],
         [beta1, gamma1],
         [beta2, gamma2]])
    # Number of samples
    ns = X.shape[0]
    # Number of classes
    nc = 3
    tmp = np.ndarray((ns, nc))
    tmp = np.einsum('ij,kj->ik',X,W)
    tmp -= logsumexp(tmp, axis=1)[:, None]
    return tmp


############################################################################
# cluster model using PMV equations #
############################################################################
def make_model_cluster_pmv(X, Y, ID, num_clusters=3):
    """
    Initialize the model.
    
    :param num_clusters: The number of clusters you desire to use.
    :param X: the input data for model training, (N x 6)
    :param Y: the output data for model training, thermal preference votes, (N x 1)
    :param ID: an array indicating from which occupant each row of X and Y were collected (N x 1)
    """
    # ------- Setting ------- #    
    gamma = 1.
    # The number of occupants
    D = ID.max() + 1
    # The number of samples
    N = X.shape[0]
    # The number of features (including the constant in multinomial logistic regression)
    Nf = 2
    # The number of clusters we are looking for
    K = num_clusters
    # The number of distinct classes
    C = np.unique(Y).max() + 1
    
    # Split the data according to what belongs to each person
    x = np.empty(D,dtype=object)
    y = np.empty(D, dtype=object)
    for d in xrange(D):
        idx = ID == d
        x[d] = X[idx, :]
        y[d] = Y[idx]

    # The hyper-parameter controlling the prior of Mu
    lambda_mean = np.array([35., 0., -35.])/10.
    lambda_tau = np.array([0.1, 100., 0.1]) # inv.lambda_var in the paper
    
    # A helper function to compute the prior of Mu
    def mu_function(value=None, mean=None, tau=None):
        return pm.normal_like(value, mu=mean, tau=tau)
        
    # The Mu
    mu = np.empty(K, dtype=object)
    for i in xrange(K):
        mu[i] = pm.Stochastic(logp = mu_function,
                              parents = {'mean': lambda_mean,
                                         'tau': lambda_tau},
                              doc = 'Prior of Mu',
                              name = 'mu_%d' % i,
                              value=lambda_mean,
                              dtype=np.float,
                              observed=False)
                              
    # The Sigma
    sigma = np.empty(K, dtype=object)
    for i in xrange(K):
        sigma[i] = pm.InverseGamma('sigma_%d' % i,
                                   alpha=1.,
                                   beta=0.5,
                                   value=5.)
                                   
    # xi parameters all the clusters sharing, xi_1,2,3 in the paper
    xi_shared = pm.Normal('xi_shared',
                          mu=np.ones((3,)),
                          tau=100.,
                          value = np.ones((3,)))
                          
    # The hyper-parameters controlling the prior of xi_cluster
    alpha_xi = pm.Exponential('alpha_xi',
                              beta=np.array([1.,1.,1.]),
                              value=[10., 10., 10.])
                              
    # A helper function to compute prior of xi_cluster
    def xi_function(value=None, alpha_xi=None):
        return pm.normal_like(value,
                              mu=np.array([alpha_xi[0], 0.]),
                              tau=1./np.array([alpha_xi[1]], alpha_xi[2]))
                              
    # The cluster specific xi parameters, xi_(k,4) and xi_(k,5) in the paper
    xi_cluster = np.empty(num_clusters, dtype=object)
    for i in xrange(num_clusters):
        xi_cluster[i] = pm.Stochastic(logp = xi_function,
                          parents = {'alpha_xi': alpha_xi},
                          doc = 'Prior of xi_cluster',
                          name = 'xi_cluster_%d' % i,
                          value=np.array([0.,1.]),
                          dtype=np.float,
                          observed=False)
                          
    # The hidden cluster value z_(1:D)
    z = np.empty(D, dtype=object)
    for d in xrange(D):
        z[d] = pm.DiscreteUniform('z_%d' % d,
                                  lower=0,
                                  upper=K-1,
                                  value=d % K)  

    # A helper function to compute the thermal load
    def features_func(x=None, z=None, xi_shared=None, xi_cluster=None):
        """
        Return the value of the thermal load for x.
        """
        ns = x.shape[0]
        pmv = pmv_function(xi_shared=xi_shared, xi_cluster=xi_cluster[z], X=x)
        return np.hstack([pmv[:,None]/10., np.ones((ns, 1))])

    # The thermal load associated with each person, E in the paper
    features = np.empty(D, dtype=object)
    for d in xrange(D):
        features[d] = pm.Deterministic(eval = features_func,
                                  name = 'features_%d' % d,
                                  parents = {'z': z[d],
                                             'xi_shared': xi_shared,
                                             'xi_cluster': xi_cluster,
                                             'x': x[d]},
                                  doc = 'The features for person %d' % d,
                                  trace = False,
                                  verbose = 0,
                                  plot = False)
                                  
    # A helper function to compute the likelihood of each person
    def log_like(value=None, mu=None, sigma=None, features=None, z=None, gamma=None):
        nc = mu.shape[0]
        for i in xrange(nc):
            mud = mu[i]
            if any(mud[:-1] < mud[1:]):
                return -np.inf     
                
        mud = mu[z]
        sigmad = sigma[z]
        ns = features.shape[0]
        logp = loglike_of_mlr(features, mud, sigmad)
        return gamma * logp[np.arange(ns), value.astype(np.int)].sum()
        
    # The log likelihood associated with each person
    y_obs = np.empty(D, dtype=object)
    for d in xrange(D):
        y_obs[d] = pm.Stochastic(logp = log_like,
                                 parents = {'mu': mu,
                                            'sigma': sigma,
                                            'features': features[d],
                                            'z': z[d],
                                            'gamma': gamma},
                                 doc = 'The log likelihood associated with person %d' % d,
                                 name = 'y_obs_%d' % d,
                                 value = y[d],
                                 dtype=np.int,
                                 observed=True,
                                 plot=False)

    return locals()
    
############################################################################
# nocluster model using PMV equations #
############################################################################
def make_model_nocluster_pmv(X, Y, ID, num_clusters=3):
    """
    Initialize the model.
    
    :param num_clusters: The number of clusters you desire to use.
    :param X: the input data for model training, (N x 6)
    :param Y: the output data for model training, thermal preference votes, (N x 1)
    :param ID: an array indicating from which occupant each row of X and Y were collected (N x 1)
    """
    # ------- Setting ------- #    
    gamma = 1.
    # The number of occupants
    D = ID.max() + 1
    # The number of samples
    N = X.shape[0]
    # The number of features (including the constant in multinomial logistic regression)
    Nf = 2
    # The number of clusters we are looking for
    K = num_clusters
    # The number of distinct classes
    C = np.unique(Y).max() + 1
    
    x = X
    y = Y
    
    # The hyper-parameter controlling the prior of Mu
    lambda_mean = np.array([35., 0., -35.])/10.
    lambda_tau = np.array([0.1, 100., 0.1]) # inv.lambda_var in the paper
    
    # A helper function to compute the prior of Mu
    def mu_function(value=None, mean=None, tau=None):
        return pm.normal_like(value, mu=mean, tau=tau)
        
    # The Mu
    mu = pm.Stochastic(logp = mu_function,
                       parents = {'mean': lambda_mean,
                                  'tau': lambda_tau},
                       doc = 'Prior of Mu',
                       name = 'mu',
                       value=lambda_mean,
                       dtype=np.float,
                       observed=False)
                              
    # The Sigma
    sigma = pm.InverseGamma('sigma',
                            alpha=1.,
                            beta=0.5,
                            value=5.)
                                   
    # xi parameters all the clusters sharing, xi_1,2,3 in the paper
    xi_shared = pm.Normal('xi_shared',
                          mu=np.ones((3,)),
                          tau=100.,
                          value = np.ones((3,)))
                          
    # The hyper-parameters controlling the prior of xi_cluster
    alpha_xi = pm.Exponential('alpha_xi',
                              beta=np.array([1.,1.,1.]),
                              value=[10., 10., 10.])
                              
    # A helper function to compute prior of xi_cluster
    def xi_function(value=None, alpha_xi=None):
        return pm.normal_like(value,
                              mu=np.array([alpha_xi[0], 0.]),
                              tau=1./np.array([alpha_xi[1]], alpha_xi[2]))
                              
    # The cluster specific xi parameters, xi_(k,4) and xi_(k,5) in the paper
    xi_cluster = pm.Stochastic(logp = xi_function,
                               parents = {'alpha_xi': alpha_xi},
                               doc = 'Prior of xi_cluster',
                               name = 'xi_cluster',
                               value=np.array([0.,1.]),
                               dtype=np.float,
                               observed=False)

    # A helper function to compute the thermal load
    def features_func(x=None, xi_shared=None, xi_cluster=None):
        """
        Return the value of the thermal load for x.
        """
        ns = x.shape[0]
        pmv = pmv_function(xi_shared=xi_shared, xi_cluster=xi_cluster, X=x)
        return np.hstack([pmv[:,None]/10., np.ones((ns, 1))])

    # The thermal load associated with each person, E in the paper
    features = pm.Deterministic(eval = features_func,
                                name = 'features',
                                parents = {'xi_shared': xi_shared,
                                           'xi_cluster': xi_cluster,
                                           'x': x},
                                doc = 'The features for person',
                                trace = False,
                                verbose = 0,
                                plot = False)
                                  
    # A helper function to compute the likelihood of each person
    def log_like(value=None, mu=None, sigma=None, features=None, gamma=None):
        
        
        mud = mu
        if any(mud[:-1] < mud[1:]):
            return -np.inf     
        sigmad = sigma
        ns = features.shape[0]
        logp = loglike_of_mlr(features, mud, sigmad)
        return gamma * logp[np.arange(ns), value.astype(np.int)].sum()
        
    # The log likelihood associated with each person
    y_obs = pm.Stochastic(logp = log_like,
                          parents = {'mu': mu,
                                     'sigma': sigma,
                                     'features': features,
                                     'gamma': gamma},
                          doc = 'The log likelihood associated with person',
                          name = 'y_obs',
                          value = y,
                          dtype=np.int,
                          observed=True,
                          plot=False)

    return locals()

############################################################################
# cluster model using PMV equations with Dirichlet process prior#
############################################################################
def make_model_cluster_pmv_DP(X, Y, ID, num_clusters=7):
    """
    Initialize the model.
    
    :param num_clusters: The number of clusters you desire to use.
    :param X: the input data for model training, (N x 6)
    :param Y: the output data for model training, thermal preference votes, (N x 1)
    :param ID: an array indicating from which occupant each row of X and Y were collected (N x 1)
    """
    # ------- Setting ------- #    
    gamma = 1.
    # The number of occupants
    D = ID.max() + 1
    # The number of samples
    N = X.shape[0]
    # The number of features (including the constant in multinomial logistic regression)
    Nf = 2
    # The number of clusters we are looking for
    K = num_clusters
    # The number of distinct classes
    C = np.unique(Y).max() + 1
    
    # Split the data according to what belongs to each person
    x = np.empty(D,dtype=object)
    y = np.empty(D, dtype=object)
    for d in xrange(D):
        idx = ID == d
        x[d] = X[idx, :]
        y[d] = Y[idx]

    # The hyper-parameter controlling the prior of Mu
    lambda_mean = np.array([35., 0., -35.])/10.
    lambda_tau = np.array([0.1, 100., 0.1]) # inv.lambda_var in the paper
    
    # A helper function to compute the prior of Mu
    def mu_function(value=None, mean=None, tau=None):
        return pm.normal_like(value, mu=mean, tau=tau)
        
    # The Mu
    mu = np.empty(K, dtype=object)
    for i in xrange(K):
        mu[i] = pm.Stochastic(logp = mu_function,
                              parents = {'mean': lambda_mean,
                                         'tau': lambda_tau},
                              doc = 'Prior of Mu',
                              name = 'mu_%d' % i,
                              value=lambda_mean,
                              dtype=np.float,
                              observed=False)
                              
    # The Sigma
    sigma = np.empty(K, dtype=object)
    for i in xrange(K):
        sigma[i] = pm.InverseGamma('sigma_%d' % i,
                                   alpha=1.,
                                   beta=0.5,
                                   value=5.)
                                   
    # xi parameters all the clusters sharing, xi_1,2,3 in the paper
    xi_shared = pm.Normal('xi_shared',
                          mu=np.ones((3,)),
                          tau=100.,
                          value = np.ones((3,)))
                          
    # The hyper-parameters controlling the prior of xi_cluster
    alpha_xi = pm.Exponential('alpha_xi',
                              beta=np.array([1.,1.,1.]),
                              value=[10., 10., 10.])
                              
    # A helper function to compute prior of xi_cluster
    def xi_function(value=None, alpha_xi=None):
        return pm.normal_like(value,
                              mu=np.array([alpha_xi[0], 0.]),
                              tau=1./np.array([alpha_xi[1]], alpha_xi[2]))
                              
    # The cluster specific xi parameters, xi_(k,4) and xi_(k,5) in the paper
    xi_cluster = np.empty(num_clusters, dtype=object)
    for i in xrange(num_clusters):
        xi_cluster[i] = pm.Stochastic(logp = xi_function,
                          parents = {'alpha_xi': alpha_xi},
                          doc = 'Prior of xi_cluster',
                          name = 'xi_cluster_%d' % i,
                          value=np.array([0.,1.]),
                          dtype=np.float,
                          observed=False)


    alpha0 = pm.Exponential('alpha0', beta = 1.)
    nu = np.empty(K, dtype=object)
    for i in xrange(K):
        nu[i] = pm.Beta('nu_%d' % i, alpha=1., beta=alpha0, value = 0.9)
    
    
    @pm.deterministic(trace=False)
    def Pi(nu=nu):
        pi = np.zeros((K,))
        tmp = 1.
        for i in xrange(K):
            if i!=0:
                tmp = tmp * (1. - nu[i-1])
                pi[i] = nu[i] * tmp
            else:
                pi[i] = nu[i]
        
        pi = 1. / pi.sum() * pi
        return pi
        
    # The hidden cluster value z_(1:D)
    z = np.empty(D, dtype=object)
    for d in xrange(D):
        z[d] = pm.Categorical('z_%d' % d,
                              Pi,
                              value=d % K) 

    # A helper function to compute the thermal load
    def features_func(x=None, z=None, xi_shared=None, xi_cluster=None):
        """
        Return the value of the thermal load for x.
        """
        ns = x.shape[0]
        pmv = pmv_function(xi_shared=xi_shared, xi_cluster=xi_cluster[z], X=x)
        return np.hstack([pmv[:,None]/10., np.ones((ns, 1))])

    # The thermal load associated with each person, E in the paper
    features = np.empty(D, dtype=object)
    for d in xrange(D):
        features[d] = pm.Deterministic(eval = features_func,
                                  name = 'features_%d' % d,
                                  parents = {'z': z[d],
                                             'xi_shared': xi_shared,
                                             'xi_cluster': xi_cluster,
                                             'x': x[d]},
                                  doc = 'The features for person %d' % d,
                                  trace = False,
                                  verbose = 0,
                                  plot = False)
                                  
    # A helper function to compute the likelihood of each person
    def log_like(value=None, mu=None, sigma=None, features=None, z=None, gamma=None):
        nc = mu.shape[0]
        for i in xrange(nc):
            mud = mu[i]
            if any(mud[:-1] < mud[1:]):
                return -np.inf     
                
        mud = mu[z]
        sigmad = sigma[z]
        ns = features.shape[0]
        logp = loglike_of_mlr(features, mud, sigmad)
        return gamma * logp[np.arange(ns), value.astype(np.int)].sum()
        
    # The log likelihood associated with each person
    y_obs = np.empty(D, dtype=object)
    for d in xrange(D):
        y_obs[d] = pm.Stochastic(logp = log_like,
                                 parents = {'mu': mu,
                                            'sigma': sigma,
                                            'features': features[d],
                                            'z': z[d],
                                            'gamma': gamma},
                                 doc = 'The log likelihood associated with person %d' % d,
                                 name = 'y_obs_%d' % d,
                                 value = y[d],
                                 dtype=np.int,
                                 observed=True,
                                 plot=False)

    return locals()
    
############################################################################
# assign_step_functions for cluster_pmv #
############################################################################
def assign_pysmc_step_functions_cluster_pmv(mcmc):
    """
    Assign step functions to an mcmc sampler.
    """
    mcmc.use_step_method(ps.RandomWalk, mcmc.alpha_xi)
    mcmc.use_step_method(ps.RandomWalk, mcmc.xi_shared)
    for i in xrange(mcmc.K):
        mcmc.use_step_method(ps.RandomWalk, mcmc.mu[i])
        mcmc.use_step_method(ps.RandomWalk, mcmc.sigma[i])
        mcmc.use_step_method(ps.RandomWalk, mcmc.xi_cluster[i])
    for d in xrange(mcmc.D):
        mcmc.use_step_method(ps.DiscreteRandomWalk, mcmc.z[d])
        
############################################################################
# assign_step_functions for nocluster_pmv #
############################################################################
def assign_pysmc_step_functions_nocluster_pmv(mcmc):
    """
    Assign step functions to an mcmc sampler.
    """
    mcmc.use_step_method(ps.RandomWalk, mcmc.alpha_xi)
    mcmc.use_step_method(ps.RandomWalk, mcmc.mu)
    mcmc.use_step_method(ps.RandomWalk, mcmc.sigma)
    mcmc.use_step_method(ps.RandomWalk, mcmc.xi_cluster)
    mcmc.use_step_method(ps.RandomWalk, mcmc.xi_shared)
    
############################################################################
# assign_step_functions for cluster_pmv_DP #
############################################################################
def assign_pysmc_step_functions_cluster_pmv_DP(mcmc):
    """
    Assign step functions to an mcmc sampler.
    """
    mcmc.use_step_method(ps.RandomWalk, mcmc.alpha_xi)
    mcmc.use_step_method(ps.RandomWalk, mcmc.xi_shared)
    mcmc.use_step_method(ps.RandomWalk, mcmc.alpha0)
    for i in xrange(mcmc.K):
        mcmc.use_step_method(ps.RandomWalk, mcmc.mu[i])
        mcmc.use_step_method(ps.RandomWalk, mcmc.sigma[i])
        mcmc.use_step_method(ps.RandomWalk, mcmc.xi_cluster[i])
        mcmc.use_step_method(ps.RandomWalk, mcmc.nu[i])
    for d in xrange(mcmc.D):
        mcmc.use_step_method(ps.DiscreteRandomWalk, mcmc.z[d])