# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:54:50 2018

distributions for fitting to data and plotting fit results

@author: david.ross@nist.gov
"""


import numpy as np
from scipy import special
from scipy import misc
from scipy import stats
from scipy import integrate


def back_dist(x, mu, sig, lamb):
    exp_term = np.exp((2*mu + lamb*sig**2 - 2*x)*lamb/2)
    erfc_term = special.erfc((mu + lamb*sig**2 - x)/(sig*np.sqrt(2)))

    return lamb/2*exp_term*erfc_term


def log_back_dist(x, mu, sig, lamb):
    exp_term = (2*mu + lamb*sig**2 - 2*x)*lamb/2
    erfc_term = special.erfc((mu + lamb*sig**2 - x)/(sig*np.sqrt(2)))

    return np.log(lamb/2) + exp_term + np.log(erfc_term)


def gamma_dist(x, alpha, beta):
    return (beta**alpha) / special.gamma(alpha) * (x**(alpha-1)) * np.exp(-beta*x)


def log_gamma_dist(x, alpha, beta):
    return np.where(x>0, alpha*np.log(beta) - np.log(special.gamma(alpha)) + (alpha-1)*np.log(x) - beta*x, -np.inf)


def gamma_conv_normal_dist(x, alpha, scale, mu, sig):
    z = ((scale*(mu-x) + sig**2)**2)/2/(scale**2)/(sig**2)

    prefactor = (2**(-alpha/2)) / (scale**alpha) * (sig**(alpha-2))
    exp_term = np.exp(-((x-mu)**2)/2/(sig**2))
    hyper_term_1 = sig * special.hyp1f1(alpha/2, 1/2, z) / np.sqrt(2) / special.gamma((alpha+1)/2)
    hyper_term_2 = (x*scale - scale*mu - sig**2) * special.hyp1f1((1+alpha)/2, 3/2, z) / scale / special.gamma(alpha/2)

    return prefactor * exp_term * (hyper_term_1 + hyper_term_2)


def log_gamma_conv_normal_dist(x, alpha, scale, mu, sig):
    z = ((scale*(mu-x) + sig**2)**2)/2/(scale**2)/(sig**2)

    log_prefactor = -alpha/2*np.log(2) - alpha*np.log(scale) + (alpha-2)*np.log(sig)
    log_exp_term = -((x-mu)**2)/2/(sig**2)
    #log_hyper_term_1 = np.log(sig) + np.log(special.hyp1f1(alpha/2, 1/2, z)) - 0.5*np.log(2) - np.log(special.gamma((alpha+1)/2))
    #log_hyper_term_2 = np.log(x*scale - scale*mu - sig**2) + np.log(special.hyp1f1((1+alpha)/2, 3/2, z)) - np.log(scale) - np.log(special.gamma(alpha/2))
    hyper_term_1 = sig * special.hyp1f1(alpha/2, 1/2, z) / np.sqrt(2) / special.gamma((alpha+1)/2)
    hyper_term_2 = (x*scale - scale*mu - sig**2) * special.hyp1f1((1+alpha)/2, 3/2, z) / scale / special.gamma(alpha/2)

    value = log_prefactor + log_exp_term + np.log(hyper_term_1 + hyper_term_2)

    finite_test = np.isfinite(value)

    if finite_test.all():
        return value
    else:
        log_gamma = log_gamma_dist(x-mu, alpha, 1/scale)
        return np.where(finite_test, value, log_gamma)


def tail_dist(x, alpha, beta, B, mu, N):
    Z = 0
    gamma_terms = 0
    for i in range(2, N+1):
        Z += np.exp(-i*B)
        gamma_terms += np.exp(-i*B) * gamma_dist(x-mu, i*alpha, beta)

    return gamma_terms/Z


def log_tail_dist(x, alpha, beta, B, mu, N):
    Z = []
    gamma_terms = []
    for i in range(2, N+1):
        Z.append(-i*B)
        gamma_terms.append(-i*B + log_gamma_dist(x-mu, i*alpha, beta))

    return misc.logsumexp(gamma_terms, axis=0) - misc.logsumexp(Z)


def log_gamma_tail_back_dist(x, mu, sig, lamb, alpha, beta, log_B, num_tail_terms, theta1, theta_tail):
    theta_back = 1 - theta1 - theta_tail
    
    y_back = np.log(theta_back) + log_back_dist(x, mu=mu, sig=sig, lamb=lamb)
    y_gamma = np.log(theta1) + log_gamma_conv_normal_dist(x, alpha=alpha, scale=1/beta, mu=mu, sig=sig)
    y_tail = np.log(theta_tail) + log_tail_dist(x, alpha=alpha, beta=beta, mu=mu, B=np.exp(log_B), N=num_tail_terms)
    
    return misc.logsumexp([y_back, y_gamma, y_tail], axis=0) 


def lognormal_conv_normal_dist(z, lognorm_mu, lognorm_sig, mu, sig):
    def temp_funct(z, lognorm_mu, lognorm_sig, mu, sig):
        lower_integration_limit = (z - mu) - 5*sig
        if lower_integration_limit<0:
            lower_integration_limit = 0
        upper_integration_limit = (z - mu) + 5*sig
        
        def integrand(x):
            f_x = stats.lognorm.pdf(x, s=lognorm_sig, scale=np.exp(lognorm_mu))
            g_x = stats.norm.pdf(z-x, loc=mu, scale=sig)
            return f_x*g_x
        
        return integrate.quad(integrand, lower_integration_limit, upper_integration_limit)[0]
    return np.vectorize(temp_funct)(z, lognorm_mu, lognorm_sig, mu, sig)
