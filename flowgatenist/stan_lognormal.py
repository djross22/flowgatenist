# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:16:19 2018

@author: djross
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility
# import gc
from copy import deepcopy

import flowgatenist as flow
from flowgatenist import gaussian_mixture as nist_gmm
import flowgatenist.distributions as fit_dist
import flowgatenist.batch_process as batch_p

import matplotlib.cm as color_maps
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse

from scipy import stats
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd

import pystan
import pickle

import seaborn as sns


def batch_stan_lognormal_binned_optimize(data_directory,
                                         fl_channel='BL1-A-MEF',
                                         max_points=100000,
                                         update_progress=True,
                                         samples_to_stan=None,
                                         iterations=5,
                                         max_errors=5,
                                         fit_bins=1000,
                                         refitting=True,
                                         exclude_files=None,
                                         show_plots=False,
                                         fit_max=30000,
                                         fit_min=500,
                                         fit_to_singlets=True):
    
    # If refitting=True then Stan optimizing is initialized with s_opt,
    #     which tries to use the same gamma_signal parameters as the previous fit
    #
    # fit_max is the maximum data value kept for fitting
    # fit_min is the minimum value for data thrown out

    if update_progress:
        print('Start batch_stan_lognormal_binned_optimize: ' + str(pd.Timestamp.now().round('s')))
        
    os.chdir(data_directory)
    
    coli_files, blank_file_list, bead_file_list = batch_p.auto_find_files(exclude_string=exclude_files)
    
    samples, start_string = batch_p.find_sample_names(coli_files)
    
    #print(samples)
    
    sm_back, stan_back_fit = batch_p.get_stan_back_fit(fl_channel)
    
    back_mu_mean = np.mean(stan_back_fit.extract(permuted=True)['mu'])
    back_log_sigma_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma']))
    back_log_lamb_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb']))
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
        
    if fit_to_singlets:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.Stan lognormal Optimize.pdf'
    else:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.cell events.Stan lognormal Optimize.pdf'
    pdf = PdfPages(pdf_file)
    sns.set()
    
    if update_progress:
        print('    Optimizing E. coli data: ' + str(pd.Timestamp.now().round('s')))
        
    sm = batch_p.get_stan_model('fit mixture lognormal and back_binned.fixed noise.stan')
        
    if samples_to_stan is None:
        samples_to_stan = samples
    else:
        samples_to_stan = [glob.glob("*" + sam + "*.fcs_pkl") for sam in samples_to_stan]
        samples_to_stan = [item for sublist in samples_to_stan for item in sublist]
        samples_to_stan = [s[len(start_string):s.rfind('.')] for s in samples_to_stan]
    if update_progress:
        for sam in samples_to_stan:
            print('        ' + sam)
            
    fig2, axs2 = plt.subplots(len(samples), 1)
    fig2.set_size_inches([12, 4*len(samples)])
    
    mu = back_mu_mean
    sig = np.exp(back_log_sigma_mean)
    lamb = np.exp(back_log_lamb_mean)
    
    x = np.linspace(-fit_max/10, fit_max, 200)
    x2 = x[x>mu]
    bins2 = np.linspace(-fit_max/10, fit_max, 200)
    
    out_params = []
        
    for file, sam, a2 in zip(coli_files, samples, axs2):
        stan_model = sm
        
        data = pickle.load(open(file, 'rb'))
        data.flow_frame = data.flow_frame[:max_points]
        
        if fit_to_singlets:
            data_to_fit = data.flow_frame.loc[data.flow_frame['is_singlet']]
        else:
            data_to_fit = data.flow_frame.loc[data.flow_frame['is_cell']]
        
        stan_signal = data_to_fit[fl_channel].copy()
        stan_signal = stan_signal[stan_signal<max(stan_signal)]
        stan_signal = stan_signal[stan_signal<fit_max]
        
        # Trim off data points with signal > 12.5 the geometric mean 
        #     to avoid baising fits from outliers
        # But only do this if geo_mean>fit_min to avoid throwing out good data for low
        #     intensity and low density samples
        geo_mean = np.exp(np.mean(np.log(stan_signal[stan_signal > 0])))
        if fit_to_singlets:
            threshold = 12.5*geo_mean
        else:
            threshold = 25*geo_mean
        if geo_mean>fit_min:
            stan_signal = stan_signal[stan_signal<threshold]
        
        stan_signal = stan_signal.values
        
        binned_signal, bin_edges = np.histogram(stan_signal, bins=fit_bins)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        binned_df = pd.DataFrame(bin_mids, columns=['bin_mids'])
        binned_df['counts'] = pd.Series(binned_signal)
        binned_df = binned_df[binned_df['counts']>0]
        
        stan_data = dict(signal=binned_df['bin_mids'], counts=binned_df['counts'], N=len(binned_df['counts']),
                         mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                         log_lamb_mean=back_log_lamb_mean)
                
        if fit_to_singlets:
            stan_opt_file = sam + '.stan lognormal optimize.stan_fit_pkl'
        else:
            stan_opt_file = sam + '.cell events.stan lognormal optimize.stan_fit_pkl'
        
        #if (sam in samples_to_stan):
        if (sam in samples_to_stan):
            #print(sam + str(tail_B))
            if update_progress:
                print('Optimizing ' + sam)
            
            log_post = -np.inf
            best_opt = None
            got_init = False

            try:
                prev_opt = pickle.load(open(stan_opt_file, 'rb'))
                
                if refitting:
                    stan_init = init_stan_fit(stan_data, s_opt=prev_opt, var=0.95)
                else:
                    #stan_init = init_stan_fit(stan_data, s_params=prev_opt['par'], var=0.95)
                    stan_init = init_stan_fit(stan_data)
                    
                best_opt = stan_model.optimizing(data=stan_data, init=stan_init, verbose=True, as_vector=False)
                got_init = True
            except (FileNotFoundError, RuntimeError):
                got_init = False
            
            if not got_init:
                for iter in range(iterations):
                    #print('iteration ' + str(iter))
                    optimizing = True
                    n = 0
                    opt_failed = False
                    while optimizing:
                        try:
                            n += 1
                            #print('    try number ' + str(n))
                            stan_init = init_stan_fit(stan_data)
                                
                            opt = stan_model.optimizing(data=stan_data, init=stan_init, verbose=True, as_vector=False)
                            optimizing = False
                        except RuntimeError:
                            print('    try number ' + str(n) + ', caught an error...')
                            if (n >= max_errors):
                                optimizing = False
                                print('Stan optimization failed ' + str(max_errors) + ' times')
                                opt_failed = True
                                #raise RuntimeError
                            else:
                                optimizing = True
                                print('            ... trying again...')
                    
                    #print('    log posterior: ' + str(opt['value']))
                    if not opt_failed:
                        if (opt['value'] > log_post):
                            best_opt = opt
                            log_post = opt['value']
                        
            if best_opt is not None:
                stan_opt = best_opt
                with open(stan_opt_file, 'wb') as f:
                    pickle.dump(stan_opt, f)
        else:
            try:
                stan_opt = pickle.load(open(stan_opt_file, 'rb'))
                print('    Loading ' + stan_opt_file)
            except FileNotFoundError:
                stan_opt = None
                print('    Failed to load ' + stan_opt_file)
                
        if stan_opt is not None:
            opt = stan_opt['par']
            out_params.append(opt)
            
            label = sam + ', geo_mean = ' + str(geo_mean)
            
            plot_fit_results(data_to_fit[fl_channel], params=opt, back_params=stan_back_fit.extract(permuted=True),
                             axs=a2, sample=label, x_min=-fit_max/10, x_max=fit_max)            
        else:
            out_params.append(None)
                    
    pdf.savefig(fig2)
    if not show_plots:
        plt.close(fig2)
    pdf.close()
    
    if update_progress:
        print('                   ... done: ' + str(pd.Timestamp.now().round('s')))
        
    for out_p in out_params:
        try:
            del out_p['lognormal_signal']
            del out_p['noise_signal']
            del out_p['ps']
            del out_p['target_add']
        except TypeError:
            pass
        
    return out_params
    

def init_stan_fit(data, s_opt=None, s_params=None, var=0.9, with_tail=True):
    # data (dict) is the data to be used for the Stan fit
    # s_opt is a previous result from the Stan.optimizing method applied to the same datasset
    #     to be used for initialization
    # s_params is a dictionary to be used for initialization of all the fit params except 'gamma_signal'
    # var is the relative variability to be applied to the fit parameters for initialization
    
    if s_opt is None:
        if s_params is None:
            #theta1 = np.random.uniform(0.75, 0.95)
            #theta_back = 1 - theta1
            
            #signal = np.repeat(data['signal'], data['counts'])
            #signal = signal - data['mu_mean']
            signal = data['signal']
            mean_log = np.mean( np.log(signal[signal>0]) )
            std_log = np.std( np.log(signal[signal>0]) )
    
            lognorm_mu = mean_log * np.random.uniform(var, 1/var)
            lognorm_sig = std_log * np.random.uniform(var, 1/var)
        else:
            #theta1 = s_params['theta1'] * np.random.uniform(var, 1/var)
            #theta_back = 1 - theta1
    
            lognorm_mu = s_params['lognorm_mu'] * np.random.uniform(var, 1/var)
            lognorm_sig = s_params['lognorm_sig'] * np.random.uniform(var, 1/var)

        # Lognormal model with noise has not been successfully implements
        #lognormal_signal = abs(data['signal'] - np.random.normal(data['mu_mean'], np.exp(data['log_sigma_mean']), size=len(data['signal'])))
   
    else:
        opt = s_opt['par']
        
        #if dimension of opt['theta'] is 2...
        
        #theta1 = opt['theta1'] * np.random.uniform(var, 1/var)
        #theta_back = 1 - theta1
    
        lognorm_mu = opt['lognorm_mu'] * np.random.uniform(var, 1/var)
        lognorm_sig = opt['lognorm_sig'] * np.random.uniform(var, 1/var)
        
        #if (len(opt['lognorm_sig']) == len(data['signal'])):
        #    lognormal_signal = opt['lognormal_signal'] * np.random.uniform(var, 1/var, size=len(opt['lognormal_signal']))
        #else:
        #    lognormal_signal = abs(data['signal'] - np.random.normal(data['mu_mean'], np.exp(data['log_sigma_mean']), size=len(data['signal'])))
        
    #return dict(theta=np.array([theta_back, theta1]), lognorm_mu=lognorm_mu,
    #            lognorm_sig=lognorm_sig, lognormal_signal=lognormal_signal, theta1=theta1,
    #            theta_back=theta_back)
    return dict(lognorm_mu=lognorm_mu, lognorm_sig=lognorm_sig)
            

def plot_fit_results(data, params, back_params, axs, sample='', x_min=-1000, x_max=16000):

    plt.rcParams['font.size'] = 14

    mu = np.mean(back_params['mu'])
    sig = np.mean(back_params['sigma'])
    lamb = np.mean(back_params['lamb'])

    x = np.linspace(x_min, x_max, 400)
    x2 = x[x>mu]
    bins = np.linspace(x_min, x_max, 200)
    
            
    lognorm_mu = np.mean(params['lognorm_mu'])
    lognorm_sig = np.mean(params['lognorm_sig'])

    theta_back = np.mean(params['theta_back'])
    theta1 = np.mean(params['theta1'])

    y_back = theta_back*fit_dist.back_dist(x, mu=mu, sig=sig, lamb=lamb)
    y_back2 = theta_back*fit_dist.back_dist(x2, mu=mu, sig=sig, lamb=lamb)
    
    y_lognormal = theta1*fit_dist.lognormal_conv_normal_dist(x, lognorm_mu=lognorm_mu, lognorm_sig=lognorm_sig, mu=mu, sig=sig)
    
    bin_values = axs.hist(data, density=True, bins=bins, edgecolor='none')[0];
    bin_values = bin_values[bin_values>0]
  
    axs.plot(x, y_back, linewidth=3);
    axs.plot(x, y_lognormal, linewidth=3);
    axs.plot(x, y_back + y_lognormal, c='k');#linewidth=3);
    #axs.plot([threshold, threshold], [1e-8, 1e-2], color='k')
    
    label = sample
    axs.text(0.5, 0.9, label, horizontalalignment='center',
             verticalalignment='center', transform=axs.transAxes)

    axs.set_yscale('log', nonposy='clip')
    axs.set_ylim(0.5*bin_values.min(), 2*bin_values.max())
    axs.set_xlim(x_min, x_max);
    axs.set_xlabel('Signal (MEF)')
    axs.set_ylabel('Probability Density');
        
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label]):
        item.set_fontsize(14)
    for item in (axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(12)

