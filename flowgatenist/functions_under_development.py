# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:27:27 2018

scripts for batch processing flow cytometry data

@author: david.ross@nist.gov
"""

import glob  # filenames and pathnames utility
import os    # operating sytem utility
import sys
# import gc
from copy import deepcopy

import flowgatenist as flow
from flowgatenist import gaussian_mixture as nist_gmm

import flowgatenist.distributions as fit_dist
import flowgatenist.stan_utility as stan_utility

from flowgatenist.batch_process import auto_find_files
from flowgatenist.batch_process import find_sample_names
from flowgatenist.batch_process import get_stan_back_fit
from flowgatenist.batch_process import get_stan_model
from flowgatenist.batch_process import pickle_stan_sampling
from flowgatenist.batch_process import unpickle_stan_sampling

import matplotlib.cm as color_maps
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import matplotlib.image as mpimg

from scipy import stats
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd

import pystan
import pickle

import seaborn as sns

from sklearn.mixture import GaussianMixture as SkMixture

import warnings


plt.rcParams['axes.labelsize'] = 14




def time_series(main_directory,
                sample,
                new_file=False,
                max_points=100000,
                fl_channel='BL1-A-MEF',
                mean_cutoff=400,
                out_frame_file='time course means.pkl',
                show_plots=False):
    
    os.chdir(main_directory)
    days = pd.Series(glob.glob('*Day*'))
    
    ordered_days = pd.DataFrame(days, columns=['day'])
    
    ordered_days['number'] = pd.Series([int(n[4:]) for n in ordered_days['day']])
    ordered_days = ordered_days.sort_values(by=['number'])
    ordered_days = ordered_days.reset_index(drop=True)
    ordered_days['directory'] = main_directory + '/' + ordered_days['day']
    
    data_files = []
    generations = []
    for d in ordered_days['directory']:
        os.chdir(d)
        file = glob.glob('*' + sample + '.fcs_pkl')[0]
        generations.append(int(file[file.find('(G')+2:file.find(')_')]))
        data_files.append(file)
        
    ordered_days['data_files'] = pd.Series(data_files)
    ordered_days['generations'] = pd.Series(generations)
    ordered_days['plot_label'] = pd.Series([sample + ', Generation ' + str(g) for g in generations])
    
    # For New RPU experiment, drop Day 3 becasue there were no beads on Day 3:
    #ordered_days = ordered_days.drop(1)
    
    coli_data_temp = []
    number_events = []
    
    for index, row in ordered_days.iterrows():
        os.chdir(row['directory'])
        file = row['data_files']
        data = pickle.load(open(file, 'rb'))
        #number_events.append(data.flow_frame.loc[data.flow_frame['is_cell']].count())
        number_events.append(len(data.flow_frame.loc[data.flow_frame['is_cell']]))
        data.flow_frame = data.flow_frame[:max_points]
        coli_data_temp.append(data)
        #gc.collect()
        
    coli_data = deepcopy(coli_data_temp)
    coli_data_temp = []
    
    gated_data = [data.flow_frame.loc[data.flow_frame['is_cell']] for data in coli_data]
    singlet_data = [data.flow_frame.loc[data.flow_frame['is_singlet']] for data in coli_data]
    #anti_gated_data = [data.flow_frame.loc[~data.flow_frame['is_cell']] for data in coli_data]
    all_data = [data.flow_frame for data in coli_data]
    
    os.chdir(main_directory)
    pdf_file = 'Time Course.Fluorescence Histograms.' + sample + '.pdf'
    pdf = PdfPages(pdf_file)
    
    means = []
    geo_means = []
    medians = []
    
    bins = np.linspace(-1000, 20000, 200)
    bins2 = np.linspace(mean_cutoff, 20000, 200)
    sns.set()
    plt.rcParams["figure.figsize"] = [16, 4*len(all_data)]
    fig, axs = plt.subplots(len(all_data), 2)
    if axs.ndim == 1:
        axs = np.array([ axs ])
        
    for i, k, data, gated, singlet in zip(range(len(all_data)), list(ordered_days.index.values), all_data, gated_data, singlet_data):
        
        means.append(np.mean(singlet[singlet[fl_channel]>mean_cutoff][fl_channel]))
        geo_means.append(np.exp(np.mean(np.log(singlet[singlet[fl_channel]>mean_cutoff][fl_channel]))))
        medians.append(np.median(singlet[singlet[fl_channel]>mean_cutoff][fl_channel]))
        
        axs[i,1].set_yscale('log', nonpositive='clip')
        for j in range(2):
            axs[i,j].set_xlim(left=-1000, right=20000)
            axs[i,j].text(0.5, 0.9, ordered_days['plot_label'][k], horizontalalignment='center', verticalalignment='center',
                          transform=axs[i,j].transAxes)
            axs[i,j].hist(data[fl_channel], density=False, bins=bins, alpha=0.3)
            axs[i,j].hist(gated[fl_channel], density=False, bins=bins, alpha=0.3, color='red')
            axs[i,j].hist(singlet[fl_channel], density=False, bins=bins2, alpha=0.3, color='yellow')
            
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    x = ordered_days['generations']
    y1 = means
    y2 = geo_means
    y3 = medians
    plt.rcParams["figure.figsize"] = [8, 8]
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y1, 'o', color='red', markersize=12, label='Mean');
    axs.plot(x, y2, 'o', color='blue', markersize=12, label='Geo_Mean');
    axs.plot(x, y3, '^', color='green', markersize=12, label='Median');
    #axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    x = ordered_days['generations']
    y1 = number_events
    plt.rcParams["figure.figsize"] = [8, 8]
    fig, axs = plt.subplots(1, 1)
    axs.plot(x, y1, 'o', color='red', markersize=12, label='Number of cell events');
    #axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    pdf.close()
    
    out_means = pd.Series(means, index=ordered_days['generations'])
    out_geo_means = pd.Series(geo_means, index=ordered_days['generations'])
    out_medians = pd.Series(medians, index=ordered_days['generations'])
    
    if new_file:
        out_frame = pd.DataFrame(out_means, columns=[sample + ' means'])
    else:
        out_frame = pickle.load(open(out_frame_file, 'rb'))
        out_frame[sample + ' means'] = out_means
        
    out_frame[sample + ' geo_means'] = out_geo_means
    out_frame[sample + ' medians'] = out_medians
    
    with open(out_frame_file, 'wb') as f:
        pickle.dump(out_frame, f)
        

def batch_stan_gamma_binned_optimize(data_directory,
                               fl_channel='BL1-A-MEF',
                               max_points=100000,
                               update_progress=True,
                               samples_to_stan=None,
                               num_tail_terms=7,
                               iterations=5,
                               max_errors=5,
                               fit_bins=1000,
                               refitting=True,
                               exclude_files=None,
                               show_plots=False,
                               fit_max=30000,
                               fit_min=500,
                               fixed_tail_B=False,
                               tail_B_list=None,
                               scale_list=None,
                               fit_to_singlets=True):
    
    # If refitting=True then Stan optimizing is initialized with s_opt,
    #     which tries to use the same gamma_signal parameters as the previous fit
    #
    # tail_B_list gives a list of the fixed tail_B values
    # If one of the values in tail_B_list is None, the method uses the regular
    # Stan model without fixed tail_B for the corresponding sample.
    # fit_max is the maximum data value kept for fitting
    # fit_min is the minimum value for data thrown out

    if update_progress:
        print('Start batch_stan_gamma_binned_optimize: ' + str(pd.Timestamp.now().round('s')))
        
    os.chdir(data_directory)
    
    coli_files, blank_file_list, bead_file_list = auto_find_files(exclude_string=exclude_files)
    
    samples, start_string = find_sample_names(coli_files)
    
    #print(samples)
    
    if fixed_tail_B:
        if tail_B_list is not None:
            if len(tail_B_list) != len(samples):
                err_str1 = 'If fixed_tail_B is True, the number of values in the list tail_B_list must be equal to the number of samples in the dataset.'
                err_str2 = '\n The number of samples is ' + str(len(samples))
                raise ValueError(err_str1 + err_str2)
        else:
            raise ValueError('If fixed_tail_B is True, method call must include a tail_B_list of tail_B values.')
            
        if scale_list is not None:
            if len(scale_list) != len(samples):
                err_str1 = 'If fixed_tail_B is True, the number of values in the list scale_list must be equal to the number of samples in the dataset.'
                err_str2 = '\n The number of samples is ' + str(len(samples))
                raise ValueError(err_str1 + err_str2)
        else:
            raise ValueError('If fixed_tail_B is True, method call must include a scale_list of beta values.')
    else:
        tail_B_list = [None for sam in samples]
        scale_list = [None for sam in samples]
    
    sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
    
    back_mu_mean = np.mean(stan_back_fit.extract(permuted=True)['mu'])
    back_log_sigma_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma']))
    back_log_lamb_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb']))
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
        
    if fit_to_singlets:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.Stan gamma Optimize.pdf'
    else:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.cell events.Stan gamma Optimize.pdf'
    pdf = PdfPages(pdf_file)
    sns.set()
    
    if update_progress:
        print('    Optimizing E. coli data: ' + str(pd.Timestamp.now().round('s')))
        
    sm = get_stan_model('fit mixture gamma back and tail_binned.fixed noise.stan')
        
    if not all(v is None for v in tail_B_list):
        sm_fixed_tail = get_stan_model('fit mixture gamma_binned.fixed beta noise and tail.stan')
    
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
        
    for file, sam, a2, tail_B, set_scale in zip(coli_files, samples, axs2, tail_B_list, scale_list):
        if tail_B is None:
            stan_model = sm
        else:
            stan_model = sm_fixed_tail
            if (update_progress and (sam in samples_to_stan)):
                print('            Using fixed tail model for sample ' + sam)
        
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
        
        if tail_B is None:
            stan_data = dict(signal=binned_df['bin_mids'], counts=binned_df['counts'], N=len(binned_df['counts']),
                             N_tail=num_tail_terms, mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                             log_lamb_mean=back_log_lamb_mean)
        else:
            stan_data = dict(signal=binned_df['bin_mids'], counts=binned_df['counts'], N=len(binned_df['counts']),
                             N_tail=num_tail_terms, mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                             log_lamb_mean=back_log_lamb_mean, tail_B=tail_B, scale=set_scale)
                
        if fit_to_singlets:
            stan_opt_file = sam + '.stan optimize.stan_fit_pkl'
        else:
            stan_opt_file = sam + '.cell events.stan optimize.stan_fit_pkl'
        
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
        
            alpha = opt['alpha']
            if tail_B is None:
                log_B = opt['log_B']
                beta = opt['beta']
                scale = 1/opt['beta']
            else:
                log_B = np.log(tail_B)
                beta = 1/set_scale
                scale = set_scale
        
            theta_back = opt['theta_back']
            theta1 = opt['theta1']
            theta_tail = opt['theta_tail']
        
            y_back = theta_back*fit_dist.back_dist(x, mu=mu, sig=sig, lamb=lamb)
            y_back2 = theta_back*fit_dist.back_dist(x2, mu=mu, sig=sig, lamb=lamb)
            
            y_gamma = theta1*np.exp(fit_dist.log_gamma_conv_normal_dist(x2, alpha=alpha, scale=scale, mu=mu, sig=sig))
            y_tail = theta_tail*np.exp(fit_dist.log_tail_dist(x2, alpha=alpha, beta=beta, mu=mu, B=np.exp(log_B), N=num_tail_terms))
            
            hist_data = data_to_fit[fl_channel]
            a2.hist(hist_data, density=True, bins=bins2, edgecolor='none');
      
            a2.plot(x, y_back, linewidth=3);
            a2.plot(x2, y_gamma, linewidth=3);
            a2.plot(x2, y_tail, linewidth=3);
            a2.plot(x2, y_back2 + y_gamma + y_tail, linewidth=3);
            a2.plot([threshold, threshold], [1e-8, 3e-4], color='k')
            # geo_mean = np.exp(np.mean(np.log(hist_data[hist_data > 0])))
            label = sam + ', geo_mean = ' + str(geo_mean)
            a2.text(0.5, 0.9, label, horizontalalignment='center', verticalalignment='center', transform=a2.transAxes)
        
            a2.set_yscale('log', nonpositive='clip')
            a2.set_ylim(1e-8, 1e-2);
            a2.set_xlim(-fit_max/10, fit_max);
            
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
            del out_p['gamma_signal']
            del out_p['noise_signal']
            del out_p['log_norm_Z']
            del out_p['ps']
            del out_p['norm_x']
            del out_p['target_add']
        except TypeError:
            pass
        
    return out_params


def stan_gamma_binned_optimize_init_with_last(data_directory,
                                              fl_channel='BL1-A-MEF',
                                              max_points=100000,
                                              update_progress=True,
                                              samples=None,
                                              num_tail_terms=8,
                                              iterations=5,
                                              max_errors=5,
                                              fit_bins=1000,
                                              show_plots=False,
                                              fit_max=30000,
                                              fit_min=500,
                                              fit_to_singlets=True):
    
    # If refitting=True then Stan optimizing is initialized with s_opt,
    #     which tries to use the same gamma_signal parameters as the previous fit
    #
    # tail_B_list gives a list of the fixed tail_B values
    # If one of the values in tail_B_list is None, the method uses the regular
    # Stan model without fixed tail_B for the corresponding sample.
    # fit_max is the maximum data value kept for fitting
    # fit_min is the minimum value for data thrown out
    
    # If fit_to_singlets=True, the method uses events with 'is_singlet'=True
    #     for fitting; if fit_to_singlet=False, it uses events with 'is_cell'=True

    if update_progress:
        print('Start batch_stan_gamma_binned_optimize: ' + str(pd.Timestamp.now().round('s')))
        
    os.chdir(data_directory)
    
    coli_files = [glob.glob("*" + sam + "*.fcs_pkl") for sam in samples]
    coli_files = [item for sublist in coli_files for item in sublist]
    if update_progress:
        for file in coli_files:
            print('        ' + file)
    
    sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
    
    back_mu_mean = np.mean(stan_back_fit.extract(permuted=True)['mu'])
    back_log_sigma_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma']))
    back_log_lamb_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb']))
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    
    if fit_to_singlets:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + samples[0] + '.Stan gamma Optimize.pdf'
    else:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + samples[0] + '.cell events.Stan gamma Optimize.pdf'
    pdf = PdfPages(pdf_file)
    sns.set()
    
    if update_progress:
        print('    Optimizing E. coli data: ' + str(pd.Timestamp.now().round('s')))
        
    sm = get_stan_model('fit mixture gamma back and tail_binned.fixed noise.stan')
            
    fig2, axs2 = plt.subplots(len(samples), 1)
    fig2.set_size_inches([12, 4*len(samples)])
    
    mu = back_mu_mean
    sig = np.exp(back_log_sigma_mean)
    lamb = np.exp(back_log_lamb_mean)
    
    x = np.linspace(-fit_max/10, fit_max, 200)
    x2 = x[x>mu]
    bins2 = np.linspace(-fit_max/10, fit_max, 200)
    
    out_params = []
        
    last_opt = None
    for file, sam, a2, in zip(coli_files, samples, axs2):
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
                         N_tail=num_tail_terms, mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                         log_lamb_mean=back_log_lamb_mean)
                
        if fit_to_singlets:
            stan_opt_file = sam + '.stan optimize.stan_fit_pkl'
        else:
            stan_opt_file = sam + '.cell events.stan optimize.stan_fit_pkl'
        
        if update_progress:
            print('Optimizing ' + sam)
        
        log_post = -np.inf
        best_opt = None
        
        if last_opt is None:
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
                        
        else:
            stan_init = init_stan_fit(stan_data, s_params=last_opt['par'], var=0.95)
            best_opt = stan_model.optimizing(data=stan_data, init=stan_init, verbose=True, as_vector=False)
                    
        stan_opt = best_opt
        with open(stan_opt_file, 'wb') as f:
            pickle.dump(stan_opt, f)
                
        last_opt = stan_opt
        opt = stan_opt['par']
        out_params.append(opt)
    
        alpha = opt['alpha']
        log_B = opt['log_B']
        beta = opt['beta']
        scale = 1/opt['beta']

        theta_back = opt['theta_back']
        theta1 = opt['theta1']
        theta_tail = opt['theta_tail']
    
        y_back = theta_back*fit_dist.back_dist(x, mu=mu, sig=sig, lamb=lamb)
        y_back2 = theta_back*fit_dist.back_dist(x2, mu=mu, sig=sig, lamb=lamb)
        
        y_gamma = theta1*np.exp(fit_dist.log_gamma_conv_normal_dist(x2, alpha=alpha, scale=scale, mu=mu, sig=sig))
        y_tail = theta_tail*np.exp(fit_dist.log_tail_dist(x2, alpha=alpha, beta=beta, mu=mu, B=np.exp(log_B), N=num_tail_terms))
        
        hist_data = data_to_fit[fl_channel]
        a2.hist(hist_data, density=True, bins=bins2, edgecolor='none');
  
        a2.plot(x, y_back, linewidth=3);
        a2.plot(x2, y_gamma, linewidth=3);
        a2.plot(x2, y_tail, linewidth=3);
        a2.plot(x2, y_back2 + y_gamma + y_tail, linewidth=3);
        a2.plot([threshold, threshold], [1e-8, 1e-2], color='k')
        # geo_mean = np.exp(np.mean(np.log(hist_data[hist_data > 0])))
        label = sam + ', geo_mean = ' + str(geo_mean)
        a2.text(0.5, 0.9, label, horizontalalignment='center', verticalalignment='center', transform=a2.transAxes)
    
        a2.set_yscale('log', nonpositive='clip')
        a2.set_ylim(1e-8, 1e-2);
        a2.set_xlim(-fit_max/10, fit_max);
            
                    
    pdf.savefig(fig2)
    if not show_plots:
        plt.close(fig2)
    pdf.close()
    
    if update_progress:
        print('                   ... done: ' + str(pd.Timestamp.now().round('s')))
        
    for out_p in out_params:
        try:
            del out_p['gamma_signal']
            del out_p['noise_signal']
            del out_p['log_norm_Z']
            del out_p['ps']
            del out_p['norm_x']
            del out_p['target_add']
        except TypeError:
            pass
        
    return out_params
    

def init_stan_fit(data, s_opt=None, s_params=None, var=0.9, with_tail=True, with_noise=True):
    # data (dict) is the data to be used for the Stan fit
    # s_opt is a previous result from the Stan.optimizing method applied to the same datasset
    #     to be used for initialization
    # s_params is a dictionary to be used for initialization of all the fit params except 'gamma_signal'
    # var is the relative variability to be applied to the fit parameters for initialization
    
    if s_opt is None:
        if s_params is None:
            theta1 = np.random.uniform(0.75, 0.95)
            if with_noise:
                theta_back = np.random.uniform(0.01, 0.03)
            else:
                theta_back = 0
            theta_tail = 1 - theta1 - theta_back
    
            log_difference = np.random.uniform(3.8, 5.7)
            log_average = np.random.uniform(-4.3, -2.3)
            log_B = np.random.uniform(-1, 1)
        else:
            theta1 = s_params['theta1'] * np.random.uniform(var, 1/var)
            if with_noise:
                theta_back = s_params['theta_back'] * np.random.uniform(var, 1/var)
            else:
                theta_back = 0
            theta_tail = s_params['theta_tail'] * np.random.uniform(var, 1/var)
            if (theta_tail == 0):
                theta_tail = 0.0000001
            theta_sum = theta1 + theta_back + theta_tail
            theta1 = theta1/theta_sum
            theta_back = theta_back/theta_sum
            theta_tail = theta_tail/theta_sum
    
            log_difference = s_params['log_difference'] + np.random.uniform(-np.log(var), np.log(var))
            log_average = s_params['log_average'] + np.random.uniform(-np.log(var), np.log(var))
            try:
                log_B = s_params['log_B'] + np.random.uniform(-np.log(var), np.log(var))
            except KeyError:
                log_B = 0

        #gamma_signal = np.random.uniform(0, 100000, data['N'])

        if with_noise:
            gamma_signal = smooth_gamma_signal_init(data=data, var=var)
            
            #gamma_signal = abs(data['signal'] - np.random.normal(data['mu_mean'], np.exp(data['log_sigma_mean']), size=len(data['signal'])))

        #mu = np.random.normal(data['mu_mean'], data['mu_stdv'])
        #log_sigma = np.random.normal(data['log_sigma_mean'], data['log_sigma_stdv'])
        #log_lamb = np.random.normal(data['log_lamb_mean'], data['log_lamb_stdv'])

        #return dict(theta1=theta1, theta_tail=theta_tail, log_difference=log_difference, log_average=log_average,
        #           log_B=log_B, gamma_signal=gamma_signal, mu=mu, log_sigma=log_sigma, log_lamb=log_lamb)

   
    else:
        opt = s_opt['par']
        
        #if dimension of opt['theta'] is 2...
        
        theta1 = opt['theta1'] * np.random.uniform(var, 1/var)
        if with_noise:
            theta_back = opt['theta_back'] * np.random.uniform(var, 1/var)
        else:
            theta_back = 0
        theta_tail = opt['theta_tail'] * np.random.uniform(var, 1/var)
        
        theta_sum = theta1 + theta_back + theta_tail
        theta1 = theta1/theta_sum
        theta_back = theta_back/theta_sum
        theta_tail = theta_tail/theta_sum

        log_difference = opt['log_difference'] + np.random.uniform(-np.log(var), np.log(var))
        log_average = opt['log_average'] + np.random.uniform(-np.log(var), np.log(var))
        log_B = opt['log_B'] + np.random.uniform(-np.log(var), np.log(var))
        
        if with_noise:
            if (len(opt['gamma_signal']) == len(data['signal'])):
                gamma_signal = opt['gamma_signal'] * np.random.uniform(var, 1/var, size=len(opt['gamma_signal']))
            else:
                gamma_signal = smooth_gamma_signal_init(data=data, var=var)
                #gamma_signal = abs(data['signal'] - np.random.normal(data['mu_mean'], np.exp(data['log_sigma_mean']), size=len(data['signal'])))
            
    if ( ('log_ave_mean' in data) and ('log_ave_sd' in data) and ('log_dif_mean' in data) and ('log_dif_sd' in data) ):
        log_difference = np.random.normal(loc=data['log_dif_mean'], scale=data['log_dif_sd'])
        
        log_average = np.random.normal(loc=data['log_ave_mean'], scale=data['log_ave_sd'])
        
    log_alpha = log_average + log_difference
    log_beta = log_average - log_difference
        
    if with_tail:
        if with_noise:
            return dict(theta=np.array([theta_back, theta1, theta_tail]), log_difference=log_difference, log_average=log_average,
                        log_B=log_B, gamma_signal=gamma_signal, log_alpha=log_alpha, log_beta=log_beta)
        else:
            return dict(theta_tail=theta_tail/(theta_tail+theta1), log_difference=log_difference, log_average=log_average,
                        log_B=log_B, log_alpha=log_alpha, log_beta=log_beta)
            
    else:
        theta_sum = theta1 + theta_back
        theta1 = theta1/theta_sum
        theta_back = theta_back/theta_sum
        return dict(theta=np.array([theta_back, theta1]), log_difference=log_difference, log_average=log_average,
                    log_B=log_B, gamma_signal=gamma_signal, log_alpha=log_alpha, log_beta=log_beta)


def smooth_gamma_signal_init(data, var=0.95):
    signal = data['signal']
    back_sigma = np.exp(data['log_sigma_mean'])
    back_mu = data['mu_mean']
    gamma_signal = ( signal - back_mu + 3*back_sigma)*( np.exp(-4*back_sigma/( signal - back_mu + 3*back_sigma) ) )
    gamma_signal[signal - back_mu <= -3*back_sigma] = 0
    gamma_signal = gamma_signal * np.random.uniform( var, 1/var, size=len(gamma_signal) )
    gamma_signal = gamma_signal + back_sigma/50
    
    return gamma_signal


def time_series_from_stan_optimizing(main_directory,
                sample,
                max_points=100000,
                fl_channel='BL1-A-MEF',
                out_frame_file=None,
                show_plots=False):
    
    if out_frame_file is None:
        out_frame_file = sample + '.Stan optimize vs. time.pkl'
    
    os.chdir(main_directory)
    days = pd.Series(glob.glob('*Day*'))
    
    ordered_days = pd.DataFrame(days, columns=['day'])
    
    ordered_days['day_number'] = pd.Series([int(n[4:]) for n in ordered_days['day']])
    ordered_days = ordered_days.sort_values(by=['day_number'])
    ordered_days = ordered_days.reset_index(drop=True)
    ordered_days['directory'] = main_directory + '/' + ordered_days['day']
    
    data_files = []
    generations = []
    for d in ordered_days['directory']:
        os.chdir(d)
        file = glob.glob('*' + sample + '.fcs_pkl')[0]
        generations.append(int(file[file.find('(G')+2:file.find(')_')]))
        data_files.append(file)
        
    ordered_days['data_files'] = pd.Series(data_files)
    ordered_days['generations'] = pd.Series(generations)
    
    # For New RPU experiment, drop Day 3 becasue there were no beads on Day 3:
    #ordered_days = ordered_days.drop(1)
    
    stan_opt_file = sample + '.stan optimize.stan_fit_pkl'
    
    os.chdir(main_directory)
    pdf_file = 'Time Course.Fluorescence Fits.' + sample + '.pdf'
    pdf = PdfPages(pdf_file)

    #coli_data_temp = []
    number_singlet_events = []
    #opt_list = []
    gamma_means = []
    gamma_alpha = []
    gamma_scale = []
    fraction_back = []
    fraction_singlet = []
    effective_tail_fraction = []
    number_dark_events = []
    number_bright_cells = []
    number_back_event = []
    
    generations = list(ordered_days['generations'])
    directories = list(ordered_days['directory'])
    data_files = list(ordered_days['data_files'])
    
    fig2, axs2 = plt.subplots(len(generations), 1)
    fig2.set_size_inches([12, 4*len(generations)])
    
    x = np.linspace(-1000, 20000, 200)
    bins2 = np.linspace(-1000, 20000, 200)
    
    for gen, direct, file, a2 in zip(generations, directories, data_files, axs2):
        os.chdir(direct)
        data = pickle.load(open(file, 'rb'))
        n_sing = len(data.flow_frame.loc[data.flow_frame['is_singlet']])
        data.flow_frame = data.flow_frame[:max_points]
        #coli_data_temp.append(data)
        #opt_list.append(opt)
        singlet = data.flow_frame.loc[data.flow_frame['is_singlet']]
        
        stan_opt = pickle.load(open(stan_opt_file, 'rb'))
        
        back_file = data.metadata.backfile
        pickle_file = back_file[:back_file.rfind('.')] + '.fcs_pkl'
        back_data = pickle.load(open(pickle_file, 'rb'))
        back_events = len(back_data.flow_frame.loc[back_data.flow_frame['is_singlet']])
        number_back_event.append(back_events)
        
        sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
        
        back_mu_mean = np.mean(stan_back_fit.extract(permuted=True)['mu'])
        back_log_sigma_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma']))
        back_log_lamb_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb']))
        
        mu = back_mu_mean
        sig = np.exp(back_log_sigma_mean)
        lamb = np.exp(back_log_lamb_mean)
        x2 = x[x>mu]
        
        opt = stan_opt['par']
    
        alpha = opt['alpha']
        beta = opt['beta']
        scale = 1/opt['beta']
        log_B = opt['log_B']
        tail_B = np.exp(log_B)
    
        theta_back = opt['theta_back']
        theta1 = opt['theta1']
        theta_tail = opt['theta_tail']
        
        gamma_means.append(opt['gamma_mean'])
        gamma_alpha.append(alpha)
        gamma_scale.append(scale)
        
        fraction_back.append(theta_back)
        fraction_singlet.append(theta1)
        try:
            tail_fract = opt['effective_tail_fraction']
        except KeyError:
            tail_fract = theta_tail*(2*np.exp(tail_B) - 1)/(np.exp(tail_B) - 1)
        effective_tail_fraction.append(tail_fract)
        
        number_singlet_events.append(n_sing)
        number_dark_events.append(n_sing*theta_back)
        number_bright_cells.append(n_sing*(theta1 + tail_fract))
    
        y_back = theta_back*fit_dist.back_dist(x, mu=mu, sig=sig, lamb=lamb)
        y_back2 = theta_back*fit_dist.back_dist(x2, mu=mu, sig=sig, lamb=lamb)
        
        y_gamma = theta1*np.exp(fit_dist.log_gamma_conv_normal_dist(x2, alpha=alpha, scale=scale, mu=mu, sig=sig))
        y_tail = theta_tail*np.exp(fit_dist.log_tail_dist(x2, alpha=alpha, beta=beta, mu=mu, B=np.exp(log_B), N=7))
    
        a2.hist(singlet[fl_channel], density=True, bins=bins2, edgecolor='none');
  
        a2.plot(x, y_back, linewidth=3);
        a2.plot(x2, y_gamma, linewidth=3);
        a2.plot(x2, y_tail, linewidth=3);
        a2.plot(x2, y_back2 + y_gamma + y_tail, linewidth=3);
        label = sample + ', generation ' + str(gen)
        a2.text(0.5, 0.9, label, horizontalalignment='center', verticalalignment='center', transform=a2.transAxes)
    
        a2.set_yscale('log', nonpositive='clip')
        a2.set_ylim(1e-8, 1e-2);
        a2.set_xlim(-1000, 20000); 
        
    pdf.savefig(fig2)
    if not show_plots:
        plt.close(fig2)
    
    x = generations
    y_mean = gamma_means
    y_alpha = gamma_alpha
    y_scale = gamma_scale
    y_dark = np.array(number_dark_events)
    y_bright = np.array(number_bright_cells)
    y_back = np.array(number_back_event)
    y_dark_ratio = y_dark/(y_dark + y_bright)
    y_bright_ratio = y_bright/(y_dark + y_bright)
    y_back_ratio = y_back/(y_dark + y_bright)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig3, axs3 = plt.subplots(1, 1)
    axs3.plot(x, y_mean, 'o', color='red', markersize=12, label='Mean of gamma dist.');
    axs3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig3)
    if not show_plots:
        plt.close(fig3)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig4, axs4 = plt.subplots(1, 1)
    axs4.plot(x, y_alpha, 'o', color='blue', markersize=12, label='Alpha of gamma dist.');
    axs4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig4)
    if not show_plots:
        plt.close(fig4)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig5, axs5 = plt.subplots(1, 1)
    axs5.plot(x, y_scale, 'o', color='blue', markersize=12, label='Scale of gamma dist.');
    axs5.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig5)
    if not show_plots:
        plt.close(fig5)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig6, axs6 = plt.subplots(1, 1)
    axs6.plot(x, y_dark, 'o', color='blue', markersize=12, label='Dark');
    axs6.plot(x, y_bright, 'o', color='red', markersize=12, label='Bright');
    axs6.plot(x, y_back, 'o', color='green', markersize=12, label='Back');
    axs6.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig6)
    if not show_plots:
        plt.close(fig6)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig7, axs7 = plt.subplots(1, 1)
    axs7.plot(x, y_dark_ratio, 'o', color='blue', markersize=12, label='Dark fraction');
    axs7.plot(x, y_bright_ratio, 'o', color='red', markersize=12, label='Bright fraction');
    axs7.plot(x, y_back_ratio, 'o', color='green', markersize=12, label='Background');
    axs7.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig7)
    if not show_plots:
        plt.close(fig7)
    
    if show_plots:
        plt.show(fig7)
    pdf.close()
    
    out_means = pd.Series(gamma_means, index=generations)
    out_alpha = pd.Series(gamma_alpha, index=generations)
    out_scale = pd.Series(gamma_scale, index=generations)
    out_dark = pd.Series(number_dark_events, index=generations)
    out_bright = pd.Series(number_bright_cells, index=generations)
    out_back = pd.Series(number_back_event, index=generations)
    
    out_frame = pd.DataFrame(out_means, columns=['gamma_means'])
        
    out_frame['gamma_alpha'] = out_alpha
    out_frame['gamma_scale'] = out_scale
    out_frame['number_dark_events'] = out_dark
    out_frame['number_bright_cells'] = out_bright
    out_frame['number_back_event'] = out_back
    
    os.chdir(main_directory)
    
    with open(out_frame_file, 'wb') as f:
        pickle.dump(out_frame, f)
        
    out_frame.to_csv(out_frame_file[:out_frame_file.rfind('.')] + '.csv')
    

def batch_stan_gamma_binned_sampling(data_directory,
                                     samples_to_stan,
                                     fl_channel='BL1-A-MEF',
                                     max_points=100000,
                                     update_progress=True,
                                     num_tail_terms=8,
                                     fit_bins=1000,
                                     fit_max=None,
                                     iterations=1000,
                                     chains=4,
                                     use_prior_gamma=False,
                                     prior_gamma=None,
                                     show_plots=False,
                                     fit_to_singlets=True):
    
    # If use_prior_gamma=True, the aStan sampling will be run with an informative prior on
    #     the fit parameters log_difference and log_average.
    #     In that case, the prior_gamma input variable will be a 2 x 2 array:
    #         prior_gamma[0][0] = log_ave_mean, prior_gamma[0][1] = log_ave_sd,
    #         prior_gamma[1][0] = log_dif_mean, prior_gamma[1][1] = log_dif_sd.
    
    if update_progress:
        print('Start batch_stan_gamma_binned_sampling: ' + str(pd.Timestamp.now().round('s')))
        
    os.chdir(data_directory)
    
    sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
    
    back_mu_mean = np.mean(stan_back_fit.extract(permuted=True)['mu'])
    back_log_sigma_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma']))
    back_log_lamb_mean = np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb']))
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
    #pdf_file = data_directory[data_directory.rfind('/')+1:] + '.Stan gamma Optimize.pdf'
    #pdf = PdfPages(pdf_file)
    #sns.set()
    
    if update_progress:
        print('    Sampling fit to E. coli data: ' + str(pd.Timestamp.now().round('s')))

    sm_binned = get_stan_model('fit mixture gamma back and tail_binned.fixed noise.stan')
    
    sm_binned_prior = get_stan_model('fit mixture gamma back and tail_binned.prior gamma.stan')
    
    os.chdir(data_directory)
    
    for sample in samples_to_stan:
        if update_progress:
            print('        ' + sample + ', ' + str(pd.Timestamp.now().round('s')))
        if fit_to_singlets:
            stan_opt_file = sample + '.stan optimize.stan_fit_pkl'
        else:
            stan_opt_file = sample + '.cell events.stan optimize.stan_fit_pkl'
        previous_fit = pickle.load(open(stan_opt_file, 'rb'))
        data_file = glob.glob('*' + sample + '*.fcs_pkl')[0]
        data = pickle.load(open(data_file, 'rb'))
        data.flow_frame = data.flow_frame[:max_points]
        
        if fit_to_singlets:
            data_to_fit = data.flow_frame.loc[data.flow_frame['is_singlet']]
        else:
            data_to_fit = data.flow_frame.loc[data.flow_frame['is_cell']]
    
        stan_signal = data_to_fit[fl_channel].copy()
        stan_signal = stan_signal[stan_signal<max(stan_signal)]
        if fit_max is not None:
            stan_signal = stan_signal[stan_signal<fit_max]
        stan_signal = stan_signal.values
        
        binned_signal, bin_edges = np.histogram(stan_signal, bins=fit_bins)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        binned_df = pd.DataFrame(bin_mids, columns=['bin_mids'])
        binned_df['counts'] = pd.Series(binned_signal)
        binned_df = binned_df[binned_df['counts']>0]
        
        if use_prior_gamma:
            stan_data = dict(signal=binned_df['bin_mids'], counts=binned_df['counts'], N=len(binned_df['counts']),
                             N_tail=num_tail_terms, mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                             log_lamb_mean=back_log_lamb_mean,
                             log_ave_mean=prior_gamma[0][0], log_ave_sd=prior_gamma[0][1],
                             log_dif_mean=prior_gamma[1][0], log_dif_sd=prior_gamma[1][1])
        
            stan_init = [init_stan_fit(stan_data, s_opt=previous_fit, var=0.95) for j in range(4)]
            
            stan_fit = sm_binned_prior.sampling(data=stan_data, iter=iterations, chains=chains, init=stan_init, verbose=True)
            sm = sm_binned_prior
            
        else:
            stan_data = dict(signal=binned_df['bin_mids'], counts=binned_df['counts'], N=len(binned_df['counts']),
                             N_tail=num_tail_terms, mu_mean=back_mu_mean, log_sigma_mean=back_log_sigma_mean,
                             log_lamb_mean=back_log_lamb_mean)
        
            stan_init = [init_stan_fit(stan_data, s_opt=previous_fit, var=0.95) for j in range(4)]
            
            stan_fit = sm_binned.sampling(data=stan_data, iter=iterations, chains=chains, init=stan_init, verbose=True)
            sm = sm_binned
            
        if fit_to_singlets:
            stan_fit_file = sample + '.stan sampling output.stan_samp_pkl'
        else:
            stan_fit_file = sample + '.cell events.stan sampling output.stan_samp_pkl'
        pickle_stan_sampling(fit=stan_fit, model=sm, file=stan_fit_file)
            

def plot_fit_results(data, params, back_params, axs=None, num_tail_terms=12, sample='', x_min=-1000, x_max=16000):

    plt.rcParams['font.size'] = 14
    
    if axs is None:
        plt.rcParams["figure.figsize"] = [12, 4]
        fig, axs = plt.subplots(1,1)

    x = np.linspace(x_min, x_max, 200)
    bins = np.linspace(x_min, x_max, 200)

    bin_values = axs.hist(data, density=True, bins=bins, edgecolor='none')[0];
    bin_values = bin_values[bin_values>0]
    
    axs.text(0.5, 0.9, 'Sample: ' + sample, horizontalalignment='center', verticalalignment='center', transform=axs.transAxes)

    axs.set_yscale('log', nonpositive='clip')
    axs.set_ylim(0.5*bin_values.min(), 2*bin_values.max())
    axs.set_xlim(x_min, x_max);
    axs.set_xlabel('Signal (MEF)')
    axs.set_ylabel('Probability Density')
    
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label]):
        item.set_fontsize(14)
    for item in (axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(12)
        
    if params is not None:
        mu = np.mean(back_params['mu'])
        sig = np.mean(back_params['sigma'])
        lamb = np.mean(back_params['lamb'])
        
        x2 = x[x>mu]
    
        #opt = previous_fit['par']
    
        alpha = np.mean(params['alpha'])
        beta = np.mean(params['beta'])
        scale = 1/beta
    
        log_B = np.mean(params['log_B'])
    
        theta_back = np.mean(params['theta_back'])
        theta1 = np.mean(params['theta1'])
        theta_tail = np.mean(params['theta_tail'])
    
        y_back = theta_back*fit_dist.back_dist(x, mu=mu, sig=sig, lamb=lamb)
        y_back2 = theta_back*fit_dist.back_dist(x2, mu=mu, sig=sig, lamb=lamb)
    
        y_gamma = theta1*np.exp(fit_dist.log_gamma_conv_normal_dist(x2, alpha=alpha, scale=scale, mu=mu, sig=sig))
        y_tail = theta_tail*np.exp(fit_dist.log_tail_dist(x2, alpha=alpha, beta=beta, mu=mu, B=np.exp(log_B), N=num_tail_terms))
    
        axs.plot(x, y_back, linewidth=4);
        axs.plot(x2, y_gamma, linewidth=4);
        axs.plot(x2, y_tail, linewidth=4);
        axs.plot(x2, y_back2 + y_gamma + y_tail, linewidth=3, color='k', linestyle='--');
        
        
def batch_plot_stan_sampling(data_directory,
                             exclude_files=None,
                             fl_channel='BL1-A-MEF',
                             max_points=100000,
                             num_tail_terms=8,
                             fit_bins=1000,
                             show_plots=False,
                             fit_to_singlets=True,
                             x_min=-1000,
                             x_max=16000):
            
    os.chdir(data_directory)
    
    coli_files, blank_file_list, bead_file_list = auto_find_files(exclude_string=exclude_files)
    
    samples, start_string = find_sample_names(coli_files)
    
    sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
    back_fit_samples = stan_back_fit.extract(permuted=True)
    back_params = pd.DataFrame({ key: back_fit_samples[key] for key in ['mu', 'sigma', 'lamb'] })
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
        
    if fit_to_singlets:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.Stan gamma Sampling.pdf'
    else:
        pdf_file = data_directory[data_directory.rfind('/')+1:] + '.cell events.Stan gamma Sampling.pdf'
    pdf = PdfPages(pdf_file)
    sns.set()
            
    fig2, axs2 = plt.subplots(len(samples), 1)
    fig2.set_size_inches([12, 4*len(samples)])
        
    for file, sam, a2 in zip(coli_files, samples, axs2):
        data = pickle.load(open(file, 'rb'))
        data.flow_frame = data.flow_frame[:max_points]
        if fit_to_singlets:
            singlet_data = data.flow_frame.loc[data.flow_frame['is_singlet']]
        else:
            singlet_data = data.flow_frame.loc[data.flow_frame['is_cell']]
        stan_signal = singlet_data[fl_channel].copy()
        stan_signal = stan_signal[stan_signal<max(stan_signal)]
        
        if fit_to_singlets:
            stan_fit_file = sam + '.stan sampling output.stan_samp_pkl'
        else:
            stan_fit_file = sam + '.cell events.stan sampling output.stan_samp_pkl'
        if os.path.exists(stan_fit_file):
            sm, stan_fit = unpickle_stan_sampling(file=stan_fit_file)
        else:
            ### This is needed to be compatible with the initial method used for Stan sampling pickles
            stan_fit_file = sam + '.stan sampling output.stan_fit_pkl'
            if os.path.exists(stan_fit_file):
                stan_model_file = 'stan sampling output.stan_model_pkl'
                sm = pickle.load(open(stan_model_file, 'rb'))
                stan_fit = pickle.load(open(stan_fit_file, 'rb'))
            else:
                stan_fit = None
            ###
        
        if stan_fit is None:
            params = None
        else:
            stan_fit_samples = stan_fit.extract(permuted=True)
            params = pd.DataFrame({ key: stan_fit_samples[key] for key in ['alpha', 'beta', 'log_B', 'theta_back', 'theta1', 'theta_tail'] })
            
        plot_fit_results(stan_signal, params, back_params, a2, sample=sam, x_min=x_min, x_max=x_max, num_tail_terms=num_tail_terms)
        
    pdf.savefig(fig2)
    if not show_plots:
        plt.close(fig2)
    pdf.close()


def time_series_from_stan_sampling(main_directory,
                                   sample,
                                   max_points=100000,
                                   fl_channel='BL1-A-MEF',
                                   out_frame_file=None,
                                   show_plots=False,
                                   x_min=-1000,
                                   x_max=20000,
                                   num_tail_terms=8):
    
    if out_frame_file is None:
        out_frame_file = sample + '.Stan sampling vs. time.pkl'
    
    os.chdir(main_directory)
    days = pd.Series(glob.glob('*Day*'))
    
    ordered_days = pd.DataFrame(days, columns=['day'])
    
    ordered_days['day_number'] = pd.Series([int(n[4:]) for n in ordered_days['day']])
    ordered_days = ordered_days.sort_values(by=['day_number'])
    ordered_days = ordered_days.reset_index(drop=True)
    ordered_days['directory'] = main_directory + '/' + ordered_days['day']
    
    data_files = []
    generations = []
    for d, n in zip(ordered_days['directory'], ordered_days['day_number']):
    #for d in ordered_days['directory']:
        os.chdir(d)

        file_list = glob.glob('*' + sample + '.fcs_pkl')
        if len(file_list)>0:
            file = glob.glob('*' + sample + '.fcs_pkl')[0]
        else:
            file = None
        
        # Use next two lines to detirmine generation number from day number
        gen = (n-1)*10 +5
        generations.append(gen)
        # Use next line to detirmine generation number from file name
        #generations.append(int(file[file.find('(G')+2:file.find(')_')]))
        
        data_files.append(file)
        
    ordered_days['data_files'] = pd.Series(data_files)
    ordered_days['generations'] = pd.Series(generations)
    
    # Check to make sure each day has a 'stan sampling output.stan_fit_pkl' file
    #     and that the fit converged: neff >=10, Rhat < 1.1
    directories = list(ordered_days['directory'])
    days = list(ordered_days['day'])
    keep = [True for direct in directories]
    check_params = ['alpha', 'beta_Mathematica', 'theta1', 'gamma_mean']
    for i, (direct, day, file) in enumerate(zip(directories, days, data_files)):
        os.chdir(direct)
        
        stan_fit_file = sample + '.stan sampling output.stan_samp_pkl'
        
        if os.path.exists(stan_fit_file):
            sm_binned, stan_fit = unpickle_stan_sampling(stan_fit_file)
        else:
            ### This is needed to be compatible with the initial method used for Stan sampling pickles
            stan_fit_file = sample + '.stan sampling output.stan_fit_pkl'
            if os.path.exists(stan_fit_file):
                try:
                    sm_binned = get_stan_model(stan_file='fit mixture gamma back and tail_binned.fixed noise.stan')
                    sm_prior_gamma = get_stan_model(stan_file='fit mixture gamma back and tail_binned.prior gamma.stan')
                    stan_fit = pickle.load(open(stan_fit_file, 'rb'))
                except ModuleNotFoundError:
                    keep[i] = False
            ###
            else:
                keep[i] = False
                print('        ' + day + ', missing stan_fit_file')
        
        if file is None:
            keep[i] = False
            print('        ' + day + ', missing fcs.pkl file')
        
        if keep[i]:
            s = stan_fit.summary()
            fit_summary = pd.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])
            for c_p in check_params:
                if fit_summary['Rhat'][c_p] > 1.1:
                    keep[i] = False
                    print('        ' + day + ', not convereged, Rhat = ' + str(fit_summary['Rhat'][c_p]) + ', for paramater ' + c_p)
                if fit_summary['n_eff'][c_p] < 10:
                    keep[i] = False
                    print('        ' + day + ', not convereged, n_eff = ' + str(fit_summary['n_eff'][c_p]) + ', for paramater ' + c_p)
                
    ordered_days['keep'] = keep
    ordered_days = ordered_days[ordered_days['keep']]
    
    
    os.chdir(main_directory)
    pdf_file = 'Time Course.Fluorescence Sampling Fits.' + sample + '.pdf'
    pdf = PdfPages(pdf_file)

    #coli_data_temp = []
    number_singlet_events = []
    #opt_list = []
    gamma_means = []
    gamma_alpha = []
    gamma_scale = []
    fraction_back = []
    fraction_singlet = []
    effective_tail_fraction = []
    number_dark_events = []
    number_bright_cells = []
    number_back_event = []
    
    gamma_means_sd = []
    gamma_alpha_sd = []
    gamma_scale_sd = []
    fraction_back_sd = []
    fraction_singlet_sd = []
    effective_tail_fraction_sd = []
    number_dark_events_sd = []
    number_bright_cells_sd = []
    number_back_event_sd = []
    
    y_dark_ratio = []
    y_bright_ratio = []
    y_back_ratio = []
    y_dark_ratio_sd = []
    y_bright_ratio_sd = []
    y_back_ratio_sd = []
    
    generations = list(ordered_days['generations'])
    directories = list(ordered_days['directory'])
    data_files = list(ordered_days['data_files'])
    
    fig2, axs2 = plt.subplots(len(generations), 1)
    if type(axs2) is not np.ndarray:
        axs2 = [ axs2 ]
    
    fig2.set_size_inches([12, 4*len(generations)])
    
    x = np.linspace(-1000, 20000, 200)
    bins2 = np.linspace(-1000, 20000, 200)
    
    for gen, direct, file, a2 in zip(generations, directories, data_files, axs2):
        os.chdir(direct)
        
        data = pickle.load(open(file, 'rb'))
        n_sing = len(data.flow_frame.loc[data.flow_frame['is_singlet']])
        data.flow_frame = data.flow_frame[:max_points]
        singlet = data.flow_frame.loc[data.flow_frame['is_singlet']]
        
        sm_binned, stan_fit = unpickle_stan_sampling(stan_fit_file)
        
        stan_samples = stan_fit.extract(permuted=True)
        
        fit_alpha_mean = np.mean(stan_samples['alpha'])
        fit_beta_mean = np.mean(stan_samples['beta'])
        fit_scale_mean = np.mean(1/stan_samples['beta'])
        fit_log_B_mean = np.mean(stan_samples['log_B'])
        fit_theta1_mean = np.mean(stan_samples['theta1'])
        fit_theta_tail_mean = np.mean(stan_samples['theta_tail'])
        fit_theta_back_mean = np.mean(stan_samples['theta_back'])
        
        fit_alpha_stdv = np.std(stan_samples['alpha'])
        fit_beta_stdv = np.std(stan_samples['beta'])
        fit_scale_stdv = np.std(1/stan_samples['beta'])
        fit_log_B_stdv = np.std(stan_samples['log_B'])
        fit_theta1_stdv = np.std(stan_samples['theta1'])
        fit_theta_tail_stdv = np.std(stan_samples['theta_tail'])
        fit_theta_back_stdv = np.std(stan_samples['theta_back'])
        
        fit_gamma_mean_mean = np.mean(stan_samples['gamma_mean'])
        fit_gamma_mean_stdv = np.std(stan_samples['gamma_mean'])
        
        fit_dict = {'alpha': fit_alpha_mean, 'beta': fit_beta_mean, 'log_B': fit_log_B_mean,
                    'theta1': fit_theta1_mean, 'theta_tail': fit_theta_tail_mean,
                    'theta_back': fit_theta_back_mean}
        
        back_file = data.metadata.backfile
        pickle_file = back_file[:back_file.rfind('.')] + '.fcs_pkl'
        back_data = pickle.load(open(pickle_file, 'rb'))
        back_events = len(back_data.flow_frame.loc[back_data.flow_frame['is_singlet']])
        number_back_event.append(back_events)
    
        sm_back, stan_back_fit = get_stan_back_fit(fl_channel)
        back_fit_samples = stan_back_fit.extract(permuted=True)
        
        back_mu_mean = np.mean(back_fit_samples['mu'])
        back_log_sigma_mean = np.mean(np.log(back_fit_samples['sigma']))
        back_log_lamb_mean = np.mean(np.log(back_fit_samples['lamb']))
        
        mu = back_mu_mean
        sig = np.exp(back_log_sigma_mean)
        lamb = np.exp(back_log_lamb_mean)
        x2 = x[x>mu]
        
        alpha = fit_alpha_mean
        beta = fit_beta_mean
        scale = fit_scale_mean
        log_B = fit_log_B_mean
        tail_B = np.exp(log_B)
    
        theta_back = fit_theta_back_mean
        theta1 = fit_theta1_mean
        theta_tail = fit_theta_tail_mean
        
        gamma_means.append(fit_gamma_mean_mean)
        gamma_alpha.append(alpha)
        gamma_scale.append(scale)
        gamma_means_sd.append(fit_gamma_mean_stdv)
        gamma_alpha_sd.append(fit_alpha_stdv)
        gamma_scale_sd.append(fit_scale_stdv)
        
        fraction_back.append(theta_back)
        fraction_singlet.append(theta1)
        fraction_back_sd.append(fit_theta_back_stdv)
        fraction_singlet_sd.append(fit_theta1_stdv)
        
        try:
            tail_fract_array = stan_samples['effective_tail_fraction']
        except KeyError:
            tail_fract_array = stan_samples['theta_tail']*(2*np.exp(np.exp(stan_samples['log_B'])) - 1)/(np.exp(np.exp(stan_samples['log_B'])) - 1)
        tail_fract = np.mean(tail_fract_array)
        tail_fract_sd = np.std(tail_fract_array)
        effective_tail_fraction.append(tail_fract)
        effective_tail_fraction_sd.append(tail_fract_sd)
        
        number_singlet_events.append(n_sing)
        number_dark_events.append(n_sing*theta_back)
        
        if np.isfinite(tail_fract_array).all():
            bright_cells_array = stan_samples['theta1'] + tail_fract_array
        else:
            bright_cells_array = stan_samples['theta1']
        number_bright_cells.append(n_sing*np.mean(bright_cells_array))
        
        number_dark_events_sd.append(n_sing*fit_theta_back_stdv)
        number_bright_cells_sd.append(n_sing*np.std(bright_cells_array))
                
        y_dark_ratio.append(np.mean(stan_samples['theta_back']/(stan_samples['theta_back'] + bright_cells_array)))
        y_bright_ratio.append(np.mean(bright_cells_array/(stan_samples['theta_back'] + bright_cells_array)))
        y_back_ratio.append(np.mean(back_events/n_sing/(stan_samples['theta_back'] + bright_cells_array)))
        y_dark_ratio_sd.append(np.std(stan_samples['theta_back']/(stan_samples['theta_back'] + bright_cells_array)))
        y_bright_ratio_sd.append(np.std(bright_cells_array/(stan_samples['theta_back'] + bright_cells_array)))
        y_back_ratio_sd.append(np.std(back_events/n_sing/(stan_samples['theta_back'] + bright_cells_array)))
        
        params = pd.DataFrame({ key: stan_samples[key] for key in ['alpha', 'beta', 'log_B', 'theta_back', 'theta1', 'theta_tail'] })
        back_params = pd.DataFrame({ key: back_fit_samples[key] for key in ['mu', 'sigma', 'lamb'] })
        label = sample + ', generation ' + str(gen)
        plot_fit_results(singlet[fl_channel], params, back_params, a2, sample=label, x_min=x_min, x_max=x_max, num_tail_terms=num_tail_terms)
        
    pdf.savefig(fig2)
    if not show_plots:
        plt.close(fig2)
    
    x = generations
    y_mean = gamma_means
    y_alpha = gamma_alpha
    y_scale = gamma_scale
    y_dark = np.array(number_dark_events)
    y_bright = np.array(number_bright_cells)
    y_back = np.array(number_back_event)
    y_mean_sd = gamma_means_sd
    y_alpha_sd = gamma_alpha_sd
    y_scale_sd = gamma_scale_sd
    y_dark_sd = np.array(number_dark_events_sd)
    y_bright_sd = np.array(number_bright_cells_sd)
    y_back_sd = np.array(number_back_event_sd)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig3, axs3 = plt.subplots(1, 1)
    axs3.errorbar(x, y_mean, yerr=y_mean_sd, marker='o', color='red', markersize=10, label='Mean of gamma dist.');
    axs3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig3)
    if not show_plots:
        plt.close(fig3)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig4, axs4 = plt.subplots(1, 1)
    axs4.errorbar(x, y_alpha, yerr=y_alpha_sd, marker='o', color='blue', markersize=10, label='Alpha of gamma dist.');
    axs4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig4)
    if not show_plots:
        plt.close(fig4)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig5, axs5 = plt.subplots(1, 1)
    axs5.errorbar(x, y_scale, yerr=y_scale_sd, marker='o', color='blue', markersize=10, label='Scale of gamma dist.');
    axs5.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig5)
    if not show_plots:
        plt.close(fig5)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig6, axs6 = plt.subplots(1, 1)
    axs6.errorbar(x, y_dark, yerr=y_dark_sd, marker='o', color='blue', markersize=10, label='Dark');
    axs6.errorbar(x, y_bright, yerr=y_bright_sd, marker='o', color='red', markersize=10, label='Bright');
    #axs6.errorbar(x, y_back, yerr=y_back_sd, marker='o', color='green', markersize=10, label='Back');
    axs6.scatter(x, y_back, marker='o', c='green', s=10, label='Back');
    axs6.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig6)
    if not show_plots:
        plt.close(fig6)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig7, axs7 = plt.subplots(1, 1)
    axs7.errorbar(x, y_dark_ratio, yerr=y_dark_ratio_sd, marker='o', color='blue', markersize=10, label='Dark fraction');
    axs7.errorbar(x, y_bright_ratio, yerr=y_bright_ratio_sd, marker='o', color='red', markersize=10, label='Bright fraction');
    axs7.errorbar(x, y_back_ratio, yerr=y_back_ratio_sd, marker='o', color='green', markersize=10, label='Background');
    axs7.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig7)
    if not show_plots:
        plt.close(fig7)
    
    if show_plots:
        plt.show(fig7)
    pdf.close()
    
    out_means = pd.Series(gamma_means, index=generations)
    out_alpha = pd.Series(gamma_alpha, index=generations)
    out_scale = pd.Series(gamma_scale, index=generations)
    out_dark = pd.Series(y_dark_ratio, index=generations)
    out_bright = pd.Series(y_bright_ratio, index=generations)
    out_back = pd.Series(y_back_ratio, index=generations)
    
    out_means_sd = pd.Series(gamma_means_sd, index=generations)
    out_alpha_sd = pd.Series(gamma_alpha_sd, index=generations)
    out_scale_sd = pd.Series(gamma_scale_sd, index=generations)
    out_dark_sd = pd.Series(y_dark_ratio_sd, index=generations)
    out_bright_sd = pd.Series(y_bright_ratio_sd, index=generations)
    out_back_sd = pd.Series(y_back_ratio_sd, index=generations)
    
    out_frame = pd.DataFrame(out_means, columns=['gamma_means'])
    out_frame['gamma_means_sd'] = out_means_sd
        
    out_frame['gamma_alpha'] = out_alpha
    out_frame['gamma_alpha_sd'] = out_alpha_sd
    out_frame['gamma_scale'] = out_scale
    out_frame['gamma_scale_sd'] = out_scale_sd
    out_frame['fraction_dark_events'] = out_dark
    out_frame['fraction_dark_events_sd'] = out_dark_sd
    out_frame['fraction_bright_cells'] = out_bright
    out_frame['fraction_bright_cells_sd'] = out_bright_sd
    out_frame['fraction_back_event'] = out_back
    out_frame['fraction_back_event_sd'] = out_back_sd
    
    os.chdir(main_directory)
    
    with open(out_frame_file, 'wb') as f:
        pickle.dump(out_frame, f)
        
    out_frame.to_csv(out_frame_file[:out_frame_file.rfind('.')] + '.csv')
    
    return out_frame



def tail_time_series_from_stan_sampling(main_directory,
                                        sample,
                                        max_points=100000,
                                        fl_channel='BL1-A-MEF',
                                        out_frame_file=None,
                                        show_plots=False):
    
    if out_frame_file is None:
        out_frame_file = sample + '.tail vs. time.pkl'
    
    os.chdir(main_directory)
    days = pd.Series(glob.glob('*Day*'))
    
    ordered_days = pd.DataFrame(days, columns=['day'])
    
    ordered_days['day_number'] = pd.Series([int(n[4:]) for n in ordered_days['day']])
    ordered_days = ordered_days.sort_values(by=['day_number'])
    ordered_days = ordered_days.reset_index(drop=True)
    ordered_days['directory'] = main_directory + '/' + ordered_days['day']
    
    data_files = []
    generations = []
    for d, n in zip(ordered_days['directory'], ordered_days['day_number']):
    #for d in ordered_days['directory']:
        os.chdir(d)
        
        file_list = glob.glob('*' + sample + '.fcs_pkl')
        if len(file_list)>0:
            file = glob.glob('*' + sample + '.fcs_pkl')[0]
        else:
            file = None
        
        # Use next two lines to detirmine generation number from day number
        gen = (n-1)*10 +5
        generations.append(gen)
        # Use next line to detirmine generation number from file name
        #generations.append(int(file[file.find('(G')+2:file.find(')_')]))
        
        data_files.append(file)
        
    ordered_days['data_files'] = pd.Series(data_files)
    ordered_days['generations'] = pd.Series(generations)
    
    # Check to make sure each day has a 'stan sampling output.stan_fit_pkl' file
    #     and that the fit converged: neff >=10, Rhat < 1.1
    directories = list(ordered_days['directory'])
    days = list(ordered_days['day'])
    keep = [True for direct in directories]
    check_params = ['alpha', 'beta_Mathematica', 'theta1', 'gamma_mean',
                    'theta_tail', 'tail_B']
    for i, (direct, day) in enumerate(zip(directories, days)):
        os.chdir(direct)
        
        stan_fit_file = sample + '.stan sampling output.stan_samp_pkl'
        
        if os.path.exists(stan_fit_file):
            sm_binned, stan_fit = unpickle_stan_sampling(stan_fit_file)
        else:
            ### This is needed to be compatible with the initial method used for Stan sampling pickles
            stan_fit_file = sample + '.stan sampling output.stan_fit_pkl'
            if os.path.exists(stan_fit_file):
                try:
                    sm_binned = get_stan_model(stan_file='fit mixture gamma back and tail_binned.fixed noise.stan')
                    sm_prior_gamma = get_stan_model(stan_file='fit mixture gamma back and tail_binned.prior gamma.stan')
                    stan_fit = pickle.load(open(stan_fit_file, 'rb'))
                except ModuleNotFoundError:
                    keep[i] = False
            ###
            else:
                keep[i] = False
                print('        ' + day + ', missing stan_fit_file')
        
        if file is None:
            keep[i] = False
            print('        ' + day + ', missing fcs.pkl file')
            
        if keep[i]:
            s = stan_fit.summary()
            fit_summary = pd.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])
            for c_p in check_params:
                if fit_summary['Rhat'][c_p] > 1.1:
                    keep[i] = False
                    print('        ' + day + ', not convereged, Rhat = ' + str(fit_summary['Rhat'][c_p]) + ', for paramater ' + c_p)
                if fit_summary['n_eff'][c_p] < 10:
                    keep[i] = False
                    print('        ' + day + ', not convereged, n_eff = ' + str(fit_summary['n_eff'][c_p]) + ', for paramater ' + c_p)
                
    ordered_days['keep'] = keep
    ordered_days = ordered_days[ordered_days['keep']]
    
    
    os.chdir(main_directory)
    pdf_file = 'Time Course.Fluorescence Tail parameters.' + sample + '.pdf'
    pdf = PdfPages(pdf_file)

    number_singlet_events = []
    
    tail_B_list = []
    theta_tail_list = []
    theta1_list = []
    effective_tail_fraction = []
    
    tail_B_sd = []
    theta_tail_sd = []
    theta1_sd = []
    effective_tail_fraction_sd = []
    
    generations = list(ordered_days['generations'])
    directories = list(ordered_days['directory'])
    data_files = list(ordered_days['data_files'])
    
    for gen, direct, file in zip(generations, directories, data_files):
        os.chdir(direct)
        
        data = pickle.load(open(file, 'rb'))
        n_sing = len(data.flow_frame.loc[data.flow_frame['is_singlet']])
        data.flow_frame = data.flow_frame[:max_points]
        singlet = data.flow_frame.loc[data.flow_frame['is_singlet']]
        
        number_singlet_events.append(n_sing)
        
        sm_binned, stan_fit = unpickle_stan_sampling(stan_fit_file)
        
        stan_samples = stan_fit.extract(permuted=True)
        
        fit_tail_B_mean = np.mean(stan_samples['tail_B'])
        fit_theta1_mean = np.mean(stan_samples['theta1'])
        fit_theta_tail_mean = np.mean(stan_samples['theta_tail'])
        
        fit_tail_B_stdv = np.std(stan_samples['tail_B'])
        fit_theta1_stdv = np.std(stan_samples['theta1'])
        fit_theta_tail_stdv = np.std(stan_samples['theta_tail'])
        
        tail_B_list.append(fit_tail_B_mean)
        theta_tail_list.append(fit_theta_tail_mean)
        theta1_list.append(fit_theta1_mean)
        
        tail_B_sd.append(fit_tail_B_stdv)
        theta_tail_sd.append(fit_theta_tail_stdv)
        theta1_sd.append(fit_theta1_stdv)
        
        try:
            tail_fract_array = stan_samples['effective_tail_fraction']
        except KeyError:
            tail_fract_array = stan_samples['theta_tail']*(2*np.exp(np.exp(stan_samples['log_B'])) - 1)/(np.exp(np.exp(stan_samples['log_B'])) - 1)
        tail_fract = np.mean(tail_fract_array)
        tail_fract_sd = np.std(tail_fract_array)
        effective_tail_fraction.append(tail_fract)
        effective_tail_fraction_sd.append(tail_fract_sd)
    
    number_singlet_events = np.array(number_singlet_events)
    
    tail_B_list = np.array(tail_B_list)
    theta_tail_list = np.array(theta_tail_list)
    theta1_list = np.array(theta1_list)
    effective_tail_fraction = np.array(effective_tail_fraction)
    
    tail_B_sd = np.array(tail_B_sd)
    theta_tail_sd = np.array(theta_tail_sd)
    theta1_sd = np.array(theta1_sd)
    effective_tail_fraction_sd = np.array(effective_tail_fraction_sd)
    
    x = generations
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig3, axs3 = plt.subplots(1, 1)
    axs3.errorbar(x, theta_tail_list, yerr=theta_tail_sd, marker='o', color='red', markersize=10, label='theta_tail');
    axs3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig3)
    if not show_plots:
        plt.close(fig3)
    
    plt.rcParams["figure.figsize"] = [8, 8]
    fig4, axs4 = plt.subplots(1, 1)
    axs4.errorbar(x, tail_B_list, yerr=tail_B_sd, marker='o', color='blue', markersize=10, label='tail_B');
    axs4.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    pdf.savefig(fig4)
    if not show_plots:
        plt.close(fig4)
        
    if show_plots:
        plt.show(fig4)
    pdf.close()
    
    out_tail_B = pd.Series(tail_B_list, index=generations)
    out_theta_tail = pd.Series(theta_tail_list, index=generations)
    out_theta1 = pd.Series(theta1_list, index=generations)
    out_tail_fraction = pd.Series(effective_tail_fraction, index=generations)
    
    out_tail_B_sd = pd.Series(tail_B_sd, index=generations)
    out_theta_tail_sd = pd.Series(theta_tail_sd, index=generations)
    out_theta1_sd = pd.Series(theta1_sd, index=generations)
    out_tail_fraction_sd = pd.Series(effective_tail_fraction_sd, index=generations)
        
    out_frame = pd.DataFrame(out_theta1, columns=['theta1'])
    out_frame['theta1_sd'] = out_theta1_sd
        
    out_frame['theta_tail'] = out_theta_tail
    out_frame['theta_tail_sd'] = out_theta_tail_sd
    out_frame['tail_B'] = out_tail_B
    out_frame['out_tail_B_sd'] = out_tail_B_sd
    
    out_frame['tail_fraction'] = out_tail_fraction
    out_frame['tail_fraction_sd'] = out_tail_fraction_sd
    
    os.chdir(main_directory)
    
    with open(out_frame_file, 'wb') as f:
        pickle.dump(out_frame, f)
        
    out_frame.to_csv(out_frame_file[:out_frame_file.rfind('.')] + '.csv')
    
    return out_frame


def stan_find_max(fit_data, x_min=None, x_max=None, axs=None, iterations=1000):
    from math import isnan
    from math import isinf
    
    if x_min is not None:
        fit_data = fit_data[fit_data['x']>x_min]
    else:
        x_min = fit_data['x'].min()
    if x_max is not None:
        fit_data = fit_data[fit_data['x']<x_max]
    else:
        x_max = fit_data['x'].max()
    
    poly_fit = np.polyfit(fit_data['x'], fit_data['y'], deg=3)
    fit_func = np.poly1d(poly_fit)
    
    sm = get_stan_model('third order polynomial fit.stan')
    
    stan_init = [ dict( coef=np.multiply( fit_func.coefficients, np.random.uniform(0.95, 1/0.95, size=4) ) ) for i in range(4) ]
    
    stan_data = dict( x=fit_data['x'], y=fit_data['y'], N=len(fit_data['x']), input_coef=fit_func.coefficients )
    
    stan_samp = sm.sampling(data=stan_data, init=stan_init, verbose=True, iter=iterations, chains=4, control=dict(max_treedepth=15, adapt_delta=0.9) )
    
    s = stan_samp.summary()
    fit_summary = pd.DataFrame(s['summary'], columns=s['summary_colnames'], index=s['summary_rownames'])
    check_params = [ 'norm_coef[' + str(i) + ']' for i in range(4)]
    n_iter = len(stan_samp.extract()['lp__'])
    pass_check = True
    for c_p in check_params:
        rhat = fit_summary['Rhat'][c_p]
        if (rhat > 1.1 or isnan(rhat) or isinf(rhat)):
            pass_check = False
            print('Warning: Rhat for parameter {} is {}!'.format(c_p, rhat))
        ratio = fit_summary['n_eff'][c_p]/n_iter
        if ratio < 0.001:
            pass_check = False
            print('Warning: n_eff / iter for parameter {} is {}!'.format(c_p, ratio))
    if not stan_utility.check_energy(stan_samp):
        pass_check = False
    if not stan_utility.check_treedepth(stan_samp, max_depth=15):
        pass_check = False
    if not stan_utility.check_div(stan_samp):
        pass_check = False
        
    samples = stan_samp.extract(permuted=True)
    stan_mode = np.mean(samples['fit_mode'])
    stan_mode_sd = np.std(samples['fit_mode'])
    
    if axs is None:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches([12, 6])
    
    x = np.linspace(x_min, x_max, 200)
    
    axs[0].scatter(fit_data['x'], fit_data['y'], color='darkblue');
    axs[0].plot(x, fit_func(x), c='limegreen');
    
    y_lim = list( axs[0].get_ylim() )
    axs[0].plot([ stan_mode, stan_mode ], y_lim, c='k', linewidth=3);
    axs[0].plot([ stan_mode+stan_mode_sd, stan_mode+stan_mode_sd ], y_lim, c='orange', linewidth=1);
    axs[0].plot([ stan_mode-stan_mode_sd, stan_mode-stan_mode_sd ], y_lim, c='orange', linewidth=1);
    
    axs[1].scatter(fit_data['x'], fit_data['y'] - fit_func(fit_data['x']), color='darkblue');
    axs[1].plot(x, np.zeros( len(x) ), c='limegreen');
    
    y_lim = list( axs[1].get_ylim() )
    axs[1].plot([ stan_mode, stan_mode ], y_lim, c='k', linewidth=3);
    axs[1].plot([ stan_mode+stan_mode_sd, stan_mode+stan_mode_sd ], y_lim, c='orange', linewidth=1);
    axs[1].plot([ stan_mode-stan_mode_sd, stan_mode-stan_mode_sd ], y_lim, c='orange', linewidth=1);
    
    if not pass_check:
        axs[0].set_facecolor('mistyrose')
        axs[1].set_facecolor('mistyrose')
    
    return (stan_mode, stan_mode_sd)


def find_max(fit_data, x_min=None, x_max=None, axs=None, plot_residuals=True):
    from math import isnan
    from math import isinf
    
    if x_min is not None:
        fit_data = fit_data[fit_data['x']>x_min]
    else:
        x_min = fit_data['x'].min()
    if x_max is not None:
        fit_data = fit_data[fit_data['x']<x_max]
    else:
        x_max = fit_data['x'].max()
    
    poly_fit = np.polyfit(fit_data['x'], fit_data['y'], deg=3)
    fit_func = np.poly1d(poly_fit)
    
    fit_extrema = fit_func.deriv().roots
    mode_position = fit_extrema[fit_extrema>x_min]
    mode_position = mode_position[mode_position<x_max]
    second_deriv = fit_func.deriv(m=2)
    seconds = second_deriv(mode_position)
    mode_position = mode_position[seconds<0]
    if len(mode_position)>0:
        mode_position = mode_position[0]
    else:
        mode_position = np.nan
        
    if not np.isreal(mode_position):
        mode_position = np.nan
        
    if axs is None:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches([12, 6])
    
    x = np.linspace(x_min, x_max, 200)
    
    if plot_residuals:
        axs[0].scatter(fit_data['x'], fit_data['y'], color='darkblue');
        axs[0].plot(x, fit_func(x), c='limegreen');
        
        y_lim = list( axs[0].get_ylim() )
        axs[0].plot([ mode_position, mode_position ], y_lim, c='orange');
        
        axs[1].scatter(fit_data['x'], fit_data['y'] - fit_func(fit_data['x']), color='darkblue');
        axs[1].plot(x, np.zeros( len(x) ), c='limegreen');
        
        y_lim = list( axs[1].get_ylim() )
        axs[1].plot([ mode_position, mode_position ], y_lim, c='orange');
    else:
        axs.scatter(fit_data['x'], -fit_data['y'], color='darkblue');
        axs.plot(x, -fit_func(x), c='firebrick');
        
        y_lim = list( axs.get_ylim() )
        axs.plot([ mode_position, mode_position ], y_lim, c='r');
        
    
    return mode_position


def dark_and_bright_modes(data_file, sample='', fl_channel='BL1-A-MEF', dark_cutoff=300,
                          fit_bins=50, dark_span=[65, 100], bright_span=[0.35, 0.4],
                          fl_max=10000, axs=None, verbose=False):
    
    if axs is None:
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches([12, 4*2])
        return_plot = True
    else:
        return_plot = False
    
    a1 = axs[0::2][0]
    a2 = axs[1::2][0]
    
    samp = sample
        
    data = pickle.load(open(data_file, 'rb'))
    data = data.flow_frame
    data = data[fl_channel]
    data = data[data>data.min()]
    data = data[data<data.max()]
    
    # Find peak of dark events
    dark_data = data[data<dark_cutoff]
    
    bin_values, bin_edges = np.histogram(dark_data, bins=fit_bins, density=True)
    bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
    
    fit_data = pd.DataFrame(bin_mids, columns=['x'])
    fit_data['y'] = bin_values
    fit_data = fit_data[fit_data['y']>0]
    fit_data['y'] = np.log10(fit_data['y'])
    
    max_idx = fit_data['y'].idxmax()
    max_pos = fit_data['x'][max_idx]
    
    dark_data = data[data<max_pos+dark_span[1]]
    dark_data = dark_data[dark_data>max_pos-dark_span[0]]
    
    bin_values, bin_edges = np.histogram(dark_data, bins=fit_bins, density=True)
    bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
    
    fit_data = pd.DataFrame(bin_mids, columns=['x'])
    fit_data['y'] = bin_values
    fit_data = fit_data[fit_data['y']>0]
    fit_data['y'] = np.log10(fit_data['y'])
    
    dark_mode = find_max(fit_data=fit_data, axs=a1)
    
    if verbose:
        print('x_min, dark_mode, m_max: {}, {}, {}'.format(fit_data['x'].min(), dark_mode, fit_data['x'].min()))
    
    # Find peak of bright events
    bright_data = data[data>dark_cutoff]
    bright_data = bright_data[bright_data<fl_max]
    if verbose:
        print( 'bright min, max: {}, {}'.format( bright_data.min(), bright_data.max() ) )
    
    if len(bright_data>10):
        bin_values, bin_edges = np.histogram(bright_data, bins=500, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        max_idx = fit_data['y'].idxmax()
        max_pos = fit_data['x'][max_idx]
        if verbose:
            print( 'max_idx, max_pos: {}, {}'.format( max_idx, max_pos ) )
        
        bright_data = data[data>dark_cutoff]
        bright_data = bright_data[ bright_data<max_pos*(1+bright_span[1])+dark_span[1] ]
        bright_data = bright_data[ bright_data>max_pos*(1-bright_span[0])-dark_span[0] ]
        
        bin_values, bin_edges = np.histogram(bright_data, bins=fit_bins, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        bright_mode = find_max(fit_data=fit_data, axs=a2)
    else:
        bright_mode = np.nan
    
    if verbose:
        print('x_min, bright_mode, m_max: {}, {}, {}'.format(fit_data['x'].min(), bright_mode, fit_data['x'].min()))
        
    # Find minima between dark and bright peaks
    if not ( np.isnan(bright_mode) or bright_mode<0 ):
        if np.isnan(dark_mode):
            lower_cut = -200
        else:
            lower_cut = dark_mode
        mid_data = data[data>lower_cut]
        mid_data = mid_data[mid_data<bright_mode]
        if verbose:
            print( 'middle, min, max: {}, {}'.format( mid_data.min(), mid_data.max() ) )
            
        bin_values, bin_edges = np.histogram(mid_data, bins=200, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = -np.log10(fit_data['y'])
        
        max_idx = fit_data['y'].idxmax()
        max_pos = fit_data['x'][max_idx]
        if verbose:
            print( 'middle, max_idx, max_pos: {}, {}'.format( max_idx, max_pos ) )
        
        #mid_data = mid_data[ mid_data<max_pos+dark_span[1]*5 ]
        #mid_data = mid_data[ mid_data>max_pos-dark_span[0]*5 ]
        mid_data = mid_data[ mid_data<0.6*max_pos+0.4*bright_mode ]
        mid_data = mid_data[ mid_data>0.5*max_pos+0.5*lower_cut ]
        
        bin_values, bin_edges = np.histogram(mid_data, bins=int(round(fit_bins/2)), density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = -np.log10(fit_data['y'])
        
        dist_min = find_max(fit_data=fit_data, axs=a2[2], plot_residuals=False)
    else:
        dist_min = np.nan
    
    # Plot results
    label = samp + ' dark peak'
    a1[0].text(1, 1.03, label, horizontalalignment='center', verticalalignment='center',
      transform=a1[0].transAxes, fontsize=14)
    
    label = samp + ' bright peak'
    a2[0].text(1, 1.03, label, horizontalalignment='center', verticalalignment='center',
      transform=a2[0].transAxes, fontsize=14)
    
    if ( np.isnan(bright_mode) or bright_mode<0 ):
        bins = 500
    else:
        bins = np.linspace(data.min(), 3*bright_mode, 500)
    plot_values = a1[2].hist(data, density=True, bins=bins, edgecolor='none')[0];
    plot_values = plot_values[plot_values>0]
    a1[2].set_yscale('log', nonpositive='clip')
    a1[2].set_ylim(0.5*plot_values.min(), 2*plot_values.max());
    a1[2].plot([ bright_mode, bright_mode ], [ 0.5*plot_values.min(), 2*plot_values.max() ], c='k');
    a1[2].plot([ dark_mode, dark_mode ], [ 0.5*plot_values.min(), 2*plot_values.max() ], c='k');
    a1[2].plot([ dist_min, dist_min ], [ 0.5*plot_values.min(), 2*plot_values.max() ], c='r', linewidth=1);
    
    if return_plot:
        return ( np.array([dark_mode, bright_mode, dist_min]), fig )
    else:
        return np.array([dark_mode, bright_mode, dist_min])
    

def batch_find_max(data_files, samples=None, fl_channel='BL1-A-MEF', dark_cutoff=-100,
                   fit_bins=50, dark_span=[65, 100], bright_span=[0.35, 0.4], fl_max=10000,
                   out_frame_file=None, experiment='', update_progress=True):
    
    if out_frame_file is None:
        out_frame_file = experiment + '.distribution modes.' + fl_channel + '.pkl'
        
    pdf_file = out_frame_file[:out_frame_file.rfind('.')] + '.pdf'
    pdf = PdfPages(pdf_file)
    
    fig, axs = plt.subplots(len(data_files)*2, 3)
    fig.set_size_inches([12, 4*len(data_files)*2])
    
    axs1 = axs[0::2]
    axs2 = axs[1::2]
    
    if samples is None:
        samples = [ '' for f in data_files ]
        
    dark_mode_list = []
    dark_mode_sd_list = []
    bright_mode_list = []
    bright_mode_sd_list = []
    
    for file, samp, a1, a2 in zip(data_files, samples, axs1, axs2):
        if update_progress:
            print(samp + ': ' + str(pd.Timestamp.now().round('s')))
        
        data = pickle.load(open(file, 'rb'))
        data = data.flow_frame
        data = data[fl_channel]
        data = data[data>data.min()]
        data = data[data<data.max()]
        
        dark_data = data[data<dark_cutoff]
        
        
        bin_values, bin_edges = np.histogram(dark_data, bins=fit_bins, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        max_idx = fit_data['y'].idxmax()
        max_pos = fit_data['x'][max_idx]
        
        dark_data = data[data<max_pos+dark_span[1]]
        dark_data = dark_data[dark_data>max_pos-dark_span[0]]
        
        bin_values, bin_edges = np.histogram(dark_data, bins=fit_bins, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        dark_mode, dark_mode_sd = stan_find_max(fit_data=fit_data, axs=a1)
        dark_mode_list.append(dark_mode)
        dark_mode_sd_list.append(dark_mode_sd)
        
        label = samp + ' dark peak'
        a1[0].text(1, 1.03, label, horizontalalignment='center', verticalalignment='center',
                transform=a1[0].transAxes, fontsize=14)
        
        bright_data = data[data>dark_cutoff]
        bright_data = bright_data[bright_data<fl_max]
        
        bin_values, bin_edges = np.histogram(bright_data, bins=500, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        max_idx = fit_data['y'].idxmax()
        max_pos = fit_data['x'][max_idx]
        
        bright_data = data[ data<max_pos*(1+bright_span[1])+dark_span[1] ]
        bright_data = bright_data[bright_data>max_pos*(1-bright_span[0])-dark_span[0]]
        
        bin_values, bin_edges = np.histogram(bright_data, bins=fit_bins, density=True)
        bin_mids = (bin_edges[1:] + bin_edges[:-1])/2
        
        fit_data = pd.DataFrame(bin_mids, columns=['x'])
        fit_data['y'] = bin_values
        fit_data = fit_data[fit_data['y']>0]
        fit_data['y'] = np.log10(fit_data['y'])
        
        bright_mode, bright_mode_sd = stan_find_max(fit_data=fit_data, axs=a2)
        bright_mode_list.append(bright_mode)
        bright_mode_sd_list.append(bright_mode_sd)
        
        label = samp + ' bright peak'
        a2[0].text(1, 1.03, label, horizontalalignment='center', verticalalignment='center',
          transform=a2[0].transAxes, fontsize=14)
        
        if np.isnan(bright_mode):
            bins = 500
        else:
            bins = np.linspace(data.min(), 3*bright_mode, 500)
        plot_values = a1[2].hist(data, density=True, bins=bins, edgecolor='none')[0];
        plot_values = plot_values[plot_values>0]
        a1[2].set_yscale('log', nonpositive='clip')
        a1[2].set_ylim(0.5*plot_values.min(), 2*plot_values.max());
        a1[2].plot([ bright_mode, bright_mode ], [ 0.5*plot_values.min(), 2*plot_values.max() ], c='k');
        a1[2].plot([ dark_mode, dark_mode ], [ 0.5*plot_values.min(), 2*plot_values.max() ], c='k');
    
    out_frame = pd.DataFrame(samples, columns=['sample'])
    out_frame['file'] = data_files
        
    out_frame['dark_mode'] = dark_mode_list
    out_frame['dark_mode_sd'] = dark_mode_sd_list
    out_frame['bright_mode'] = bright_mode_list
    out_frame['bright_mode_sd'] = bright_mode_sd_list
        
    with open(out_frame_file, 'wb') as f:
        pickle.dump(out_frame, f)
        
    out_frame.to_csv(out_frame_file[:out_frame_file.rfind('.')] + '.csv')
    
    pdf.savefig(fig)
    plt.show(fig)
    pdf.close()
    
def cytometry_plots_for_SI(data_root_dir="C:\\Users\\djross\\Documents\\Jcloud\\GSF-IMS\\E-Coli\\pLMSF-lacI",
                           plot_save_dir="C:\\Users\\djross\\Documents\\Jcloud\\GSF-IMS\\LacI sensor landscape paper\\Figures\\Cytometry example",
                           start_date="2020-02-07", end_date=None, max_points=30000, show_plots=False, box_size=8):
    
    os.chdir(data_root_dir)
    
    data_dir_list = glob.glob("*Cytom-12-plasmids")
    data_dir_list.sort()
    data_dir_list = np.array(data_dir_list)
    if start_date is not None:
        data_dates = np.array([x[:10] for x in data_dir_list])
        data_dir_list = data_dir_list[data_dates>=start_date]
    if end_date is not None:
        data_dates = np.array([x[:10] for x in data_dir_list])
        data_dir_list = data_dir_list[data_dates<=end_date]
    
    for d in data_dir_list:
        for p in ["plate_1", "plate_2"]:
            os.chdir(data_root_dir)
            os.chdir(d)
            try:
                os.chdir(p + "\\Jupyter notebooks")
                print(os.getcwd())
                plot_and_save_cytometry_histograms(plot_save_dir=plot_save_dir, max_points=max_points, show_plots=show_plots,
                                                   box_size=box_size)
            except:
                print("Error:", sys.exc_info()[0])
            
def plot_and_save_cytometry_histograms(plot_save_dir, max_points=30000, show_plots=False, box_size=8):
    notebook_dir = os.getcwd()
    cytometry_directory = notebook_dir[:notebook_dir.rfind("\\")]
    main_directory = cytometry_directory[:cytometry_directory.rfind("\\")]
    
    experiment = cytometry_directory[cytometry_directory[:cytometry_directory.rfind('\\')].rfind('\\')+1:cytometry_directory.rfind('\\')]
    plate_str = cytometry_directory[cytometry_directory.find("plate_"):]
    
    os.chdir(main_directory)
    layout_file = glob.glob('*cytom-' + plate_str + '*.csv')[0]
    
    plate_layout_0 = pd.read_csv(layout_file)
    plate_layout_0.dropna(inplace=True)
    plate_layout = plate_layout_0[plate_layout_0['strain']!="none"].copy()
    
    os.chdir(cytometry_directory)
    
    back_fit_file = glob.glob('*BL1-A-MEF*.stan_samp_pkl')[0]
    sm_back, stan_back_fit = unpickle_stan_sampling(file=back_fit_file)
    stan_back_fit_samples = stan_back_fit.extract(permuted=True)
    back_mu = stan_back_fit_samples['mu'].mean()
    
    plate_layout['coli_file'] = [ glob.glob('*' + w + '.fcs_pkl')[0] for w in plate_layout['well'] ]
    
    plate_layout['sample'] = [ p for p in plate_layout['plasmid'] ]
    plate_layout['sample'] += [ '-' + str(i) + '_' for i in plate_layout["inducerConcentration"]]
    plate_layout['sample'] += [ w for w in plate_layout['well']]
    
    plate_layout.sort_values(by=['plasmid', "inducerConcentration"], inplace=True)
    
    inducerConc = plate_layout["inducerConcentration"].tolist()
    inducerUnits = plate_layout['inducerUnits'].tolist()
    for i in range(len(inducerConc)):
        if inducerUnits[i] == "mmol/L":
            inducerUnits[i] = "umol/L"
            inducerConc[i] = inducerConc[i]*1000
    plate_layout["inducerConcentration"] = inducerConc
    plate_layout['inducerUnits'] = inducerUnits
    
    coli_data = []
    
    for file in plate_layout['coli_file']:
        data = pickle.load(open(file, 'rb'))
        data.flow_frame = data.flow_frame[:max_points]
        coli_data.append(data)
        
    plasmids = np.unique(plate_layout['plasmid'].values)
    
    singlet_data = [data.flow_frame.loc[data.flow_frame['is_singlet']] for data in coli_data]
    
    data_well_dict = {}
    for data, well in zip(singlet_data, plate_layout['well']):
        data_well_dict[well] = data
    
    fl_channel = 'BL1-A-MEF'
    bins = np.linspace(1.7, 5., 50)
    shift = 0.007
    
    for plas in plasmids:
        plot_file = f"{plas}_{experiment}_{plate_str}.png"
        
        plot_frame = plate_layout[plate_layout["plasmid"]==plas]

        plt.rcParams["figure.figsize"] = [box_size, 2*len(plot_frame)/8*box_size]
        fig, axs = plt.subplots(len(plot_frame), 1)
        fig.suptitle(f"{plas}, {experiment[:10]}", size=24, y=0.905)
        axs[0].get_shared_y_axes().join(*axs)
            
        for (index, row), ax in zip(plot_frame.iterrows(), axs):
            data = data_well_dict[row["well"]]
            x = data[fl_channel] - back_mu
            x = x[x>0]
            n, b = np.histogram(np.log10(x), density=True, bins=bins)
            b = 10**b
            ax.hist(b[:-1], bins=b, weights=n);
            ax.set_xscale('log')
            
            ax.set_ylabel("Density")#, size=20)
            label = row["inducerConcentration"]
            label = f"[IPTG] = {label} µmol/L"
            ax.text(0.5, 0.9, label, horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes, size=16)
            
            if ax is not axs[-1]:
                ax.set_xticklabels([])
        axs[-1].set_xlabel("YFP Fluorescence (MEF)")#, size=20);
        for ind, ax in enumerate(axs):
            box = ax.get_position()
            box.y0 = box.y0 + shift * (ind+1)
            box.y1 = box.y1 + shift * (ind+1)
            ax.set_position(box)
        
        ylim = axs[0].get_ylim();
        axs[0].set_ylim(ylim[0], ylim[1]*1.3);
        
        os.chdir(plot_save_dir)
        fig.savefig(plot_file, dpi=None, bbox_inches="tight")
        print(f"saved: {plot_file}")
        
        if show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
    
def pdf_cytometry_histograms(plot_save_dir="C:\\Users\\djross\\Documents\\Jcloud\\GSF-IMS\\LacI sensor landscape paper\\Figures\\Cytometry example",
                             image_file_list=None, pdf_file='pVER cytometry histograms.pdf'):
    os.chdir(plot_save_dir)
    if image_file_list is None:
        image_file_list = glob.glob("*.png")
        
    pdf = PdfPages(pdf_file)
    
    plt.rcParams["figure.figsize"] = [8, 16]
    
    for f in image_file_list:
        fig, axs = plt.subplots()
        png = mpimg.imread(f)
        axs.imshow(png);
        axs.set_yticks([])
        axs.set_xticks([])
        axs.axis('off');
        
        pdf.savefig(dpi=600)
        plt.close(fig)
        
    pdf.close()
        
    
    