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

######################################
#### Get info from Config Files
def get_python_directory():
    '''
    Utility function used to find directory containing Stan models
    '''
    return os.path.dirname(os.path.realpath(__file__))

return_directory = os.getcwd()

# Try local config directory first, use default config files if local
#     version does not exist
local_dir_str = 'Local Config Files'
default_dir_str = 'Config Files'
local_config_exists = os.path.isdir(os.path.join(get_python_directory(), local_dir_str))

if local_config_exists:
    os.chdir(os.path.join(get_python_directory(), local_dir_str))
    try:
        bead_calibration_frame = pd.read_csv('bead_calibration_data.csv')
    except:
        os.chdir(os.path.join(get_python_directory(), default_dir_str))
        bead_calibration_frame = pd.read_csv('bead_calibration_data.csv')
        os.chdir(os.path.join(get_python_directory(), local_dir_str))
    try:
        with open('top_directory.txt', 'r') as file:
            top_directory = file.read().replace('\n', '')
    except:
        os.chdir(os.path.join(get_python_directory(), default_dir_str))
        with open('top_directory.txt', 'r') as file:
            top_directory = file.read().replace('\n', '')
        os.chdir(os.path.join(get_python_directory(), local_dir_str))
        
else:
    os.chdir(os.path.join(get_python_directory(), default_dir_str))
    bead_calibration_frame = pd.read_csv('bead_calibration_data.csv')
    with open('top_directory.txt', 'r') as file:
        top_directory = file.read().replace('\n', '')
    
os.chdir(return_directory)

def central_2d_guassian(df_list,
                        alpha=0.3, 
                        x_channel='FSC-A', 
                        y_channel='SSC-A',
                        n_components=3, 
                        n_init=20, 
                        random_state=None, 
                        warm_start=True):

    """
    This batch process method performs uses a GMM model to find the central 
    portion of the largest GMM component.
    The input for this method is a list of DataFrames containing flow cytometry data.
    
    Parameters
    ----------
    df_list : 1D list or array of Pandas DataFrames
        The list of DataFrames to be analyized
        
        The method adds a column to each DataFrame, 'is_central', which contains a 
        Boolean indicating whether or not each event is in the central portion 
        of the largents GMM component
        
    alpha : float
        A value between 0 and 1 indicating the fraction of the Gaussian 
        probability distribution to include in the central portion of the 
        largents GMM component
    
    x_channel : string
        The name of the flow cytometry channel used for the x-dimension of the GMM fit
        The GMM fit uses the log-transformed values from this channel
        
    y_channel : string
        The name of the flow cytometry channel used for the y-dimension of the GMM fit
        The GMM fit uses the log-transformed values from this channel
        
    n_components : integer
        the number of GMM components used in the GMM fit
        
    n_init : integer
        the number of random initializations for the GMM fit

    random_state : int, RandomState instance or None, optional (default=None)
        The value of random_state is passed to the GMM model.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        The value of warm_start is passed to the GMM model.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several time on similar problems.

    Returns
    -------
    None
    """
    gm_model = SkMixture(n_components=n_components, random_state=random_state, n_init=n_init, warm_start=warm_start)
    for df in df_list:
        if df is not None:
            # Fit scatter plot (singlet events only) to a two component Gaussian Mixture model
            df_mm = df.copy()
            df_mm = df_mm[df_mm[x_channel]>0]
            df_mm = df_mm[df_mm[y_channel]>0]
            df_mm = df_mm[df_mm.is_singlet]

            x = df_mm[x_channel]
            y = df_mm[y_channel]
            X = np.array([x, y]).transpose()
            X = np.log10(X)
            gm = gm_model.fit(X)
            
            weights = gm.weights_
            central_index = np.where(weights==max(weights))[0][0]
            mu = gm.means_[central_index]
            cov = gm.covariances_[central_index]
            
            ########################################################
            # The code below here was partially copied from: https://github.com/RPGroup-PBoC/mwc_induction/tree/1.0
            # The license file for that software repository reads:
            #     Copyright 2017 by the authors

            #     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

            #     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

            #     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
            ########################################################
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered")
                warnings.filterwarnings("ignore", message="invalid value encountered")
                x = np.log10(df[x_channel])
                y = np.log10(df[y_channel])
            interval_array = gauss_interval(x, y, mu, cov)
            df['is_central'] = interval_array <= stats.chi2.ppf(alpha, 2)
    
    
def gauss_interval(x_val, y_val, mu, cov):
    ########################################################
    # Part of the code in this method was copied from: https://github.com/RPGroup-PBoC/mwc_induction/tree/1.0
    # The license file for that software repository reads:
    #     Copyright 2017 by the authors

    #     Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    #     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    #     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    ########################################################
    '''
    Computes the of the statistic
    (x - µx)'∑(x - µx) 
    for each of the elements in df columns x_val and y_val.
    
    Parameters
    ----------
    mu : 2D array-like.
        (x, y) location of bivariate normal
    cov : 2 x 2 array
        covariance matrix
    x_val, y_val : N-dimensional array.
        x and y values to be fit
    
    Returns
    -------
    interval_array : array-like, N-dimensional 
        array containing the result of the linear algebra operation:
        (x - µx)'∑(x - µx) 
    '''
    # Determine that the covariance matrix is not singular
    det = np.linalg.det(cov)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")
            
    # Compute the vector x defined as [[x - mu_x], [y - mu_y]]
    x_vect = np.array([x_val, y_val]).transpose()
    x_vect[:, 0] = x_vect[:, 0] - mu[0]
    x_vect[:, 1] = x_vect[:, 1] - mu[1]
    
    # compute the inverse of the covariance matrix
    inv_sigma = np.linalg.inv(cov)
    
    # compute the operation
    interval_array = np.zeros(len(x_val))
    for i, x in enumerate(x_vect):
        interval_array[i] = np.dot(np.dot(x, inv_sigma), x.T)
        
    return interval_array   


def get_stan_model(stan_file):
    '''
    Utility function used to get Stan model code
    '''
    return_directory = os.getcwd()
    
    os.chdir(os.path.join(get_python_directory(), 'Stan models'))
    stan_pickle_file = stan_file[:stan_file.rfind('.')] + '.stan_model_pkl'
    if os.path.exists(stan_pickle_file):
        if (os.path.getmtime(stan_file) > os.path.getmtime(stan_pickle_file)):
            sm_model = pystan.StanModel(file=stan_file)
            with open(stan_pickle_file, 'wb') as f:
                pickle.dump(sm_model, f)
        else:
            sm_model = pickle.load(open(stan_pickle_file, 'rb'))
    else:
        sm_model = pystan.StanModel(file=stan_file)
        with open(stan_pickle_file, 'wb') as f:
            pickle.dump(sm_model, f)
            
    os.chdir(return_directory)
    
    return sm_model


def pickle_stan_sampling(fit, model, file):
    '''
    Utility function used to save Stan model fit results
    '''
    with open(file, 'wb') as f:
        pickle.dump( (model, fit), f )
        

def unpickle_stan_sampling(file):
    '''
    Utility function used to retrieve saved Stan model fit results
    '''
    model, fit = pickle.load(open(file, 'rb'))
    return (model, fit)


def get_stan_back_fit(fl_channel):
    '''
    Utility function used to retrieve saved background fit results
    '''
    back_fit_base = '.' + fl_channel + '.back_fit.stan_samp_pkl'
    if len( glob.glob('*blank*' + back_fit_base) )==0:
        back_fit_file = glob.glob('*blank*stan_back_fit.stan_fit_pkl')[0]
        stan_back_file = back_fit_file[:back_fit_file.find('.stan_fit_pkl')] + ' model.stan_model_pkl'
        
        # Note: sm_back needs to be un-pickled before stan_back_fit
        sm_back = pickle.load(open(stan_back_file, 'rb'))
        stan_back_fit = pickle.load(open(back_fit_file, 'rb'))
    else:
        back_fit_file = glob.glob('*blank*' + back_fit_base)[0]
        sm_back, stan_back_fit = unpickle_stan_sampling(file=back_fit_file)
    ###
    return (sm_back, stan_back_fit)


def auto_find_files(ext='fcs_pkl', exclude_string=None, samples=None):
    '''
    Utility function used to find flow cytomtry data files
    '''
    all_files = glob.glob("*." + ext)
    blank_file_list = glob.glob("*blank*." + ext)
    bead_files = glob.glob("*bead*." + ext)
    coli_files = sorted(list(set(all_files) - set(blank_file_list) - set(bead_files)))
        
    if samples is not None:
        coli_files = [glob.glob("*" + sam + "*." + ext) for sam in samples]
        coli_files = [item for sublist in coli_files for item in sublist]
    
    if exclude_string is not None:
        exclude_files = glob.glob("*" + exclude_string + '*' + ext)
        coli_files = sorted(list(set(coli_files) - set(exclude_files)))
    
    return coli_files, blank_file_list, bead_files


def fcs_to_dataframe(data_directory, samples=None, blank_file=None,
                     beads_file=None, exclude_string=None):

    """
    This batch process method converts fcs files into Pandas DataFrames and pickles them.

    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the fcs files are located

    samples : array of str, shape (n,)
        a list or array of samples to be converted
        the method looks for .fcs file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the actual data files
        (i.e filenames that do not contain "blank" or "bead");
        
    blank_file : path, or path-like (e.g. str)
        blank data file to be converted.
        If blank_file=None then the method convertes the last blank run before the data files.
        
    beads_file : path, or path-like (e.g. str)
        beads data file to be converted.
        If beads_file=None then the method convertes all fcs files with "bead" in the filename.
        
    exclude_string : string
        string indicating files to be excluded form conversion
        If the filename includes exclude_string, it will not be converted

    Returns
    -------
    a list of the files converted
    """

    os.chdir(data_directory)  # = "change directory"
    
    if exclude_string is None:
        exclude_string = 'adjust'

    coli_files, blank_file_list, bead_file_list = auto_find_files(ext='fcs',
                                                                  exclude_string=exclude_string,
                                                                  samples=samples)
        
    earliest_sample_time = pd.Timestamp(2100, 1, 1, 12)
    for file in coli_files:
        data = flow.io.FCSDataFrame(file)
        #data.metadata._infile = file
        if data.metadata.acquisition_start_time < earliest_sample_time:
            earliest_sample_time = data.metadata.acquisition_start_time
        
        pickle_file = file[:file.rfind('.')] + '.fcs_pkl' 
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
            
    if blank_file is None:
        latest_blank_before_samples = None
        best_blank_file = None
        latest_blank_time = pd.Timestamp(1900, 1, 1, 12)
        for b_file in blank_file_list:
            b_data = flow.io.FCSDataFrame(b_file)
            b_time = b_data.metadata.acquisition_start_time
            if ((b_time > latest_blank_time) and (b_time < earliest_sample_time)):
                latest_blank_before_samples = b_data
                best_blank_file = b_file
                latest_blank_time = b_time
        if latest_blank_before_samples is not None:
            #latest_blank_before_samples.metadata._infile = best_blank_file
            b_pickle_file = best_blank_file[:best_blank_file.rfind('.')] + '.fcs_pkl' 
            with open(b_pickle_file, 'wb') as f:
                pickle.dump(latest_blank_before_samples, f)
    else:
        b_data = flow.io.FCSDataFrame(blank_file)
        b_pickle_file = blank_file[:blank_file.rfind('.')] + '.fcs_pkl' 
        with open(b_pickle_file, 'wb') as f:
            pickle.dump(b_data, f)
        best_blank_file = blank_file
    
    if beads_file is not None:
        bead_file_list = [beads_file]
    for bead_f in bead_file_list:
        bead_data = flow.io.FCSDataFrame(bead_f)
        bead_pickle_file = bead_f[:bead_f.rfind('.')] + '.fcs_pkl' 
        with open(bead_pickle_file, 'wb') as f:
            pickle.dump(bead_data, f)
        
    return coli_files + [best_blank_file] + bead_file_list


def find_analysis_memory(data_directory, top_directory, update_progress,
                         num_memory_inits, cell_type, save_analysis_memory):
    '''
    Utility function used to set up and/or find the directories for storing Analysis Memory
    '''
    data_type = 'Flow Cytometry'
    top_pos = data_directory.lower().find(top_directory.lower())
    if (top_pos < 0):
        if update_progress:
            print('Top level directory, ' + top_directory + ', not found')
        use_analysis_memory = False
        save_analysis_memory = False
        cell_mem_dir = ''
    else:
        use_analysis_memory = (num_memory_inits > 0)
        top_directory = data_directory[:top_pos] + top_directory
        #print(top_directory)
        mem_dir = os.path.join(top_directory, data_type + ' Analysis Memory')
        #print(mem_dir)
        cell_mem_dir = os.path.join(mem_dir, cell_type)
        os.makedirs(cell_mem_dir, exist_ok=True)
        
    return (use_analysis_memory, save_analysis_memory, cell_mem_dir)


def background_subtract_gating(data_directory,
                               top_directory=top_directory,
                               samples=None,
                               back_file=None,
                               num_back_clusters=None,
                               num_cell_clusters=4,
                               back_init=None,
                               random_cell_inits=50,
                               ssc_back_cutoff=2.9,
                               fsc_back_cutoff=2.5,
                               num_memory_inits=5,
                               save_analysis_memory=True,
                               cell_type='E. coli',
                               max_events=100000,
                               init_events=30000,
                               update_progress=True,
                               show_plots=True,
                               x_channel='FSC-H',
                               y_channel='SSC-H'):

    """
    This batch process method performs a background subtraction gating to select
    flow cytometry events that are most likely to be cells.
    The input files for this method need to be pickled FCSDataFrame objects
    The method uses Gaussian Mixture Model (GMM) fits to both the background data
    and the cell data. For the fit to the cell data (which can take some time),
    it runs random_cell_inits randomly initialized GMM fits,
    and num_memory_inits GMM fits initialized from memory (recently completed
    sucessful GMM fits to similar samples).

    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located
        
    top_directory : path, or path-like (e.g. str)
        indicates the top level directory where the automated analysis will
        expect to find the "Flow Cytometry Analysis Memory" directory

    samples : array-like of str, shape (n,)
        a list or array of cell samples to be gated as a batch
        The method applies the same gating to every sample in the batch.
        The method looks for .fcs_pkl file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the actual data files
        (i.e filenames that do not contain "blank" or "bead");
        
    back_file : path, or path-like (e.g. str)
        pickled blank data file to be used for gating.
        If back_file=None then the method uses the first file in an
        automatically detected list of blank files.
        
    num_back_clusters : int
        the number of GMM clusters used to represent the probability density
        of the background signal in the scatter plot
        If num_back_clusters=None, it is automatically chosen to minimize the BIC
        
    num_cell_clusters : int
        the number of GMM clusters used to represent the probability density
        of the cell signal in the scatter plots.
        
    back_init : int
        the number of initializations to run for the GMM fit to the background data
        If back_init=None, it is automatically chosen.
        
    random_cell_inits : int
        the number of randomly (k-means) initialized GMM fits to run for the cell data
        
    ssc_back_cutoff : float
        cutoff applied to log10(SSC-H) to reduce chances of spurious background
        clusters being identified as cells. Clusters with mean(log10(SSC-H))<ssc_back_cutoff
        are taken to be background clusters.
        
    fsc_back_cutoff : float
        cutoff applied to log10(FSC-H) to reduce chances of spurious background
        clusters being identified as cells. Clusters with mean(log10(FSC-H))<fsc_back_cutoff
        are taken to be background clusters.
        
    num_memory_inits : int
        the number of GMMs initialized from memory (previous GMM fits) for cell data
        
    save_analysis_memory : Boolean
        if True, then GMM fits to cell data are saved in the memory store to
        be used for initialization of future GMM fits
        
    cell_type : str
        used to identify the correct GMM memory initialization directory
        
    max_events : int
        the maximum number of cytometry events from each cell data file to use for the GMM fit
        This parameter is used to avoid running out of memory, since the method concatenates
        the data from all the cell data files for use in the GMM fit.
        
    init_events : int
        the number of randomly sampled cytometry events to use for the
        randomly initialized GMM fit that is then used to initialize
        the fit to all of the data.
        
    update_progress : Boolean
        if True, the method prints status updates as it goes.
        
    show_plots : Boolean
        If True, the method dynamically shows plots.
        If False, it just saves the plots to a pdf file without showing them.

    x_channel : str
        used to identify the FSC channel name 

    y_channel : str
        used to identify the SSC channel name

    Returns
    -------
    None
    """

    os.chdir(data_directory)  # = "change directory"

    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()

    pdf_file = 'cell vs. background gating plots.pdf'
    pdf = PdfPages(pdf_file)
        
    a1, a2, a3 = find_analysis_memory(data_directory=data_directory,
                                      top_directory=top_directory,
                                      update_progress=update_progress,
                                      num_memory_inits=num_memory_inits,
                                      cell_type=cell_type,
                                      save_analysis_memory=save_analysis_memory)
    use_analysis_memory = a1
    save_analysis_memory = a2
    cell_mem_dir = a3
        
    # make sure to change working directory back to data_directory to load and pickle data files:
    os.chdir(data_directory)
    
    if update_progress:
        print('Start background_subtract_gating: ' + str(pd.Timestamp.now().round('s')))

    coli_files, blank_file_list, bead_file_list = auto_find_files(samples=samples)
    
    if update_progress:
        print("\n".join(['    ' + f for f in coli_files]))
        
    sample_names, start_string = find_sample_names(coli_files)
    
    # flowgatenist is set up to use background data from a blank sample
    # that is ideally run right at the beginneing of an experiment,
    # before any cell samples are run. The background data is used by
    # the gating procedure to determine whether an event is more likely
    # to be a background event or a cell event.
    # It is best to use a buffer blank measured before any cell samples
    if back_file is None:
        back_file = blank_file_list[0]

    back_data = pickle.load(open(back_file, 'rb'))
    # Add file for relevant background data to meta data and pickle
    # (this file is its own background)
    back_data.metadata._backfile = back_file

    with open(back_file, 'wb') as f:
        pickle.dump(back_data, f)

    back_data.flow_frame = back_data.flow_frame.loc[back_data.flow_frame[x_channel] > 0]
    back_data.flow_frame = back_data.flow_frame.loc[back_data.flow_frame[y_channel] > 0]
    back_data.flow_frame[f'log_{x_channel}'] = np.log10(back_data.flow_frame[x_channel])
    back_data.flow_frame[f'log_{y_channel}'] = np.log10(back_data.flow_frame[y_channel])

    # Plot the scattering data for the background sample and save to pdf file
    if update_progress:
        print('Plotting background 2D histogram, ' + str(pd.Timestamp.now().round('s')))
    x = back_data.flow_frame[f'log_{x_channel}']
    y = back_data.flow_frame[f'log_{y_channel}']
    plt.style.use('classic')
    plt.rcParams["figure.figsize"] = [8, 4]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Forward and side scatter data from blank samples', size=16)
    for ax, norm in zip(axs, [None, colors.LogNorm()]):
        ax.hist2d(x, y, bins=200, norm=norm);
        ax.set_xlabel(f'log10({x_channel})', rasterized=True, size=14)
        ax.set_ylabel(f'log10({y_channel})', rasterized=True, size=14)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    # The scattering channels of the background data are fit to a Gaussian
    # mixture model, taken from the scikit-learn package and modified to have
    # a number of fixed mixture components, for use in background subtraction
    # for the background fit, there are no fixed components, so the sentax is
    # the same as the scikit-learn GMM
    gmm_data = back_data.flow_frame.loc[:, [f'log_{x_channel}', f'log_{y_channel}']].copy()

    if num_back_clusters is None:
        if update_progress:
            print('Automated calculation of num_back_clusters... ' + str(pd.Timestamp.now().round('s')))
        n_components = np.arange(1, 16)
        models = [nist_gmm.GaussianMixture(n, covariance_type='full',
                                           n_init=10).fit(gmm_data) for n in n_components]
        bic_data = np.array([m.bic(gmm_data) for m in models])
        
        # Plot the BIC vs. n_components result:
        plt.rcParams["figure.figsize"] = [6, 4]
        fig, axs = plt.subplots()
        fig.suptitle('BIC plot for automated calculation of num_back_clusters', size=16)
        axs.plot(n_components, bic_data, 'o')
        axs.set_xlabel('num_back_clusters')
        axs.set_ylabel('BIC')
        
        cutoff = 0.02*(bic_data.max() - bic_data.min()) + bic_data.min()

        num_back_clusters = n_components[np.where(bic_data == bic_data[bic_data < cutoff][0])[0][0]]
        if update_progress:
            print('                   ... num_back_clusters = ' + str(num_back_clusters) + ', ' + str(pd.Timestamp.now().round('s')))

    if back_init is None:
        if update_progress:
            print('Automated calculation of back_init... ' + str(pd.Timestamp.now().round('s')))
        n_i = np.arange(100)
        models = [nist_gmm.GaussianMixture(n_components=num_back_clusters,
                                           covariance_type='full').fit(gmm_data) for n in n_i]
        bic_data = np.array([m.bic(gmm_data) for m in models])
        prob_sub_optimal = len(bic_data[bic_data >
                                        bic_data.min() + 0.1*(bic_data.max() -
                                                     bic_data.min())])/len(bic_data)
        back_init = int( round( np.log(0.001) / np.log(prob_sub_optimal) ) )
        back_init = min(200, back_init)
        back_init = max(100, back_init)
        if update_progress:
            print('                 ... back_init = ' + str(back_init) + ', ' + str(pd.Timestamp.now().round('s')))

    # set up and run the GMM fit
    if update_progress:
        print('Running back_fit gmm... ' + str(pd.Timestamp.now().round('s')))
        
    # If back_data has more than init_events, then run random
    #     initializations on just init_events points to speed up fitting
    if len(gmm_data)>init_events:
        if update_progress:
            print('    Running sub-sampled back_fit gmms... ' + str(pd.Timestamp.now().round('s')))
        gmm_data_small = gmm_data.sample(n=init_events)
        gmm = nist_gmm.GaussianMixture(n_components=num_back_clusters,
                                       covariance_type='full', n_init=back_init)
        back_fit_small = gmm.fit(gmm_data_small)
        
        back_means_init = back_fit_small.means_
        back_weights_init = back_fit_small.weights_
        back_precisions_init = [np.linalg.inv(cov) for cov in back_fit_small.covariances_]
        
        gmm = nist_gmm.GaussianMixture(n_components=num_back_clusters,
                                       covariance_type='full', n_init=1,
                                       weights_init=back_weights_init,
                                       means_init=back_means_init,
                                       precisions_init=back_precisions_init)
    else:
        gmm = nist_gmm.GaussianMixture(n_components=num_back_clusters,
                                       covariance_type='full', n_init=back_init)
        
    
    if update_progress:
        print('    Running full back_fit gmm... ' + str(pd.Timestamp.now().round('s')))
    back_fit = gmm.fit(gmm_data)
    
    if update_progress:
        print('              ... done, ' + str(pd.Timestamp.now().round('s')))

    # add fit result to metadata and pickle
    back_data.metadata._scatter_back_fit = back_fit
    with open(back_file, 'wb') as f:
        pickle.dump(back_data, f)

    # Plot a scatter plot of the background data, color-coded to the
    # different GMM components, and with Elipses to show the Gaussians
    if update_progress:
        print('Plotting gmm fit to background data, ' + str(pd.Timestamp.now().round('s')))
    sns.set()
    labels = back_fit.predict(gmm_data)
    probs = back_fit.predict_proba(gmm_data)
    size = 20 * probs.max(1) ** 2  # square emphasizes differences
    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('GMM fit results for blank data', y=0.93,
                 verticalalignment='bottom', size=16)
    
    for ax in axs:
        ax.scatter(gmm_data[f'log_{x_channel}'], gmm_data[f'log_{y_channel}'], c=labels,
                   cmap='viridis', s=size, rasterized=True)
        ax.set_xlabel(f'log10({x_channel})', size=14)
        ax.set_ylabel(f'log10({y_channel})', size=14)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs[1])
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs[1])

    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)

    # Then load in the actual data files:
    if update_progress:
        print(f'Loading {cell_type} data files, ' + str(pd.Timestamp.now().round('s')))
    #coli_data = [flow.io.FCSDataFrame(file) for file in coli_files]
    coli_data = [pickle.load(open(file, 'rb')) for file in coli_files]

    for i, data in enumerate(coli_data):
        data.flow_frame = data.flow_frame.loc[data.flow_frame[x_channel] > 0]
        data.flow_frame = data.flow_frame.loc[data.flow_frame[y_channel] > 0]
        data.flow_frame[f'log_{x_channel}'] = np.log10(data.flow_frame[x_channel])
        data.flow_frame[f'log_{y_channel}'] = np.log10(data.flow_frame[y_channel])

        data.metadata._infile = coli_files[i]
        data.metadata._backfile = back_file
        data.metadata._scatter_back_fit = back_fit

    scatter_data = [data.flow_frame[[f'log_{x_channel}', f'log_{y_channel}']] for data in coli_data]

    # Plot the 2D histograms for all the E. coli fcs files, and save the
    # plots to the pdf file
    if update_progress:
        print('Plotting 2D histograms for scattering data from each file, ' + str(pd.Timestamp.now().round('s')))
    figure_rows = -(len(scatter_data)//-4)
    if cell_type == "yeast":
        x_max = 6
        y_max = 5.5
    else:
        x_max = 5.5
        y_max = 5.
        
    plt.style.use('classic')
    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    
    fig, axs = plt.subplots(figure_rows, 4)
    fig.suptitle('Forward and side scatter data from cell samples before auto-gating',
                 y=0.92, verticalalignment='bottom', size=16)
    if axs.ndim == 1:
        axs = np.array([ axs ])
    else:
        axs = axs.flatten()
        axs[0].get_shared_x_axes().join(*axs)
        axs[0].get_shared_y_axes().join(*axs)
    for ax, data, name in zip(axs, scatter_data, sample_names):
        ax.text(x=0.05, y=0.95, s=name, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        ax.hist2d(data[f'log_{x_channel}'],
           data[f'log_{y_channel}'], bins=200,
           norm=colors.LogNorm(), rasterized=True);
        ax.set_xlabel(f'log10({x_channel})', size=11)
        ax.set_ylabel(f'log10({y_channel})', size=11)
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    # Use the components of the fit to the background data as the fixed part
    # of the fit to the E. coli data for background subtraction
    f_weights = back_fit.weights_
    f_means = back_fit.means_
    f_covars = back_fit.covariances_

    # For the GMM that will be used to gate out the non-cell events,
    # combine all the data from the cell data files, so that the same
    # gating in then applied to every cell sample from this batch:
    gmm_data2 = [data.flow_frame.loc[:, [f'log_{x_channel}', f'log_{y_channel}']].copy() for data in coli_data]
    gmm_data2 = [data[:max_events] for data in gmm_data2]
    gmm_data4 = pd.concat(gmm_data2, ignore_index=True)
    gmm_data3 = gmm_data4.sample(n=init_events)
    
    # Then run the nist_gmm.GaussianMixture fit with the fixed components
    # plus num_cell_clusters additional components for the non-background (cell) data.
    if update_progress:
        print('Running sub-sampled scatter_cell_fit gmms... ' + str(pd.Timestamp.now().round('s')))
    # Start with random initializations using the randomly sampled data (gmm_data4),
    gmm3 = nist_gmm.GaussianMixture(n_components=num_back_clusters+num_cell_clusters,
                                    covariance_type='full',
                                    n_init=random_cell_inits,
                                    fixed_means=f_means,
                                    fixed_covars=f_covars,
                                    fixed_weights=f_weights)
    scatter_cell_fit3 = gmm3.fit(gmm_data3)
    
    means_init = scatter_cell_fit3.means_
    weights_init = scatter_cell_fit3.weights_
    precisions_init = [np.linalg.inv(cov) for cov in scatter_cell_fit3.covariances_]
    
    # Then use the best result to initialize a fit to the full data (gmm_data3)
    # The fits can take a while with the full data set:
    if update_progress:
        print('Running final scatter_cell_fit gmm... ' + str(pd.Timestamp.now().round('s')))
    gmm4 = nist_gmm.GaussianMixture(n_components=num_back_clusters+num_cell_clusters,
                                    covariance_type='full',
                                    n_init=1,
                                    fixed_means=f_means,
                                    fixed_covars=f_covars,
                                    fixed_weights=f_weights,
                                    weights_init=weights_init,
                                    means_init=means_init,
                                    precisions_init=precisions_init)
    scatter_cell_fit = gmm4.fit(gmm_data4)

    random_lower_bound = scatter_cell_fit.lower_bound_
    
    # Then run fits with initializations from memory:
    if use_analysis_memory:
        os.chdir(cell_mem_dir)
        
        scatter_fit_mem_files = glob.glob("*scatter_cell_fit*.pkl")
        scatter_fit_mem_files.sort(key=os.path.getmtime)
        scatter_fit_mem_files = scatter_fit_mem_files[::-1] # reverse order to get newest first
        #scatter_fit_mem_files = scatter_fit_mem_files[:num_memory_inits]
        scatter_fit_inits = [pickle.load(open(file, 'rb')) for file in scatter_fit_mem_files]
        
        # Only use memory inits with num_back_clusters and num_cell_clusters
        #     that match values for this call of the method
        num_back_clusters_mem = [init.n_fixed for init in scatter_fit_inits]
        num_cell_clusters_mem = [(init.n_components - init.n_fixed) for init in scatter_fit_inits]
        mem_inits_to_use = [i for i, (x, y) in enumerate(zip(num_back_clusters_mem, num_cell_clusters_mem)) if (x==num_back_clusters and y==num_cell_clusters)]
        
        scatter_fit_inits = [scatter_fit_inits[i] for i in mem_inits_to_use]
        scatter_fit_inits = scatter_fit_inits[:num_memory_inits]

        if (len(scatter_fit_inits) > 0):
            if update_progress:
                print('Running ' + str(len(scatter_fit_inits)) + ' gmm fits initialized from memory... ' + str(pd.Timestamp.now().round('s')))
    
            for i, old_fit in enumerate(scatter_fit_inits):
                means_init = old_fit.means_
                weights_init = old_fit.weights_
                precisions_init = [np.linalg.inv(cov) for cov in old_fit.covariances_]
                gmm3_mem = nist_gmm.GaussianMixture(n_components=num_back_clusters+num_cell_clusters,
                                                 covariance_type='full',
                                                 n_init=1,
                                                 fixed_means=f_means,
                                                 fixed_covars=f_covars,
                                                 fixed_weights=f_weights,
                                                 weights_init=weights_init,
                                                 means_init=means_init,
                                                 precisions_init=precisions_init)
                scatter_cell_fit_mem = gmm3_mem.fit(gmm_data4)
                
                if i==0:
                    best_mem_gmm = scatter_cell_fit_mem
                    best_mem_gmm_init = old_fit
                else:
                    if (scatter_cell_fit_mem.lower_bound_ > best_mem_gmm.lower_bound_):
                        best_mem_gmm = scatter_cell_fit_mem
                        best_mem_gmm_init = old_fit
        else:
            print('Warning: no memory fits with matching num_back_clusters and num_cell_clusters')
            
    # Then compare Memory initialized fits to random initialized fits
    # and take the result with the highest lower_bound_
    os.chdir(cell_mem_dir)
    if ((use_analysis_memory and len(scatter_fit_inits)>0) and (best_mem_gmm.lower_bound_ > random_lower_bound)):
        scatter_cell_fit = best_mem_gmm
        # increment gmm.used_as_init and re-pickle
        best_mem_gmm_init.used_as_init += 1
        print('    using best gmm from memory to initialize, ' + best_mem_gmm_init.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(best_mem_gmm_init.memory_file, 'wb') as f:
                pickle.dump(best_mem_gmm_init, f)
    else:
        scatter_cell_fit.used_as_init = 1
        data_date = coli_data[0].metadata.acquisition_start_time.date()
        scatter_cell_fit.memory_file = 'scatter_cell_fit.' + data_date.strftime('%Y-%m-%d') + '.pkl'
        j = 1
        while (os.path.isfile(scatter_cell_fit.memory_file)):
            scatter_cell_fit.memory_file = 'scatter_cell_fit.' + data_date.strftime('%Y-%m-%d') + '.' + str(j) + '.pkl'
            j += 1
        print('    using new randomly initialized gmm, ' + scatter_cell_fit.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(scatter_cell_fit.memory_file, 'wb') as f:
                pickle.dump(scatter_cell_fit, f)
        
    os.chdir(data_directory)
    
    if update_progress:
        print('                      ... done, ' + str(pd.Timestamp.now().round('s')))

    gmm_data4 = None
    gmm_data3 = None
    gmm_data2 = None
    
    # Mark events with 'is_cell' and 'back_prob' to gate out events that are
    # predicted to belong to the background distribution, and also any that
    # belong to a cluster with mean log(SSC-H)<ssc_back_cutoff:
    for i, (data, file) in enumerate(zip(coli_data, coli_files)):
        meta = data.metadata
        pickle_file = file

        frame = data.flow_frame
        gmm_data = frame[[f'log_{x_channel}', f'log_{y_channel}']]

        meta._scatter_cell_fit = scatter_cell_fit
        back_components = meta.scatter_back_fit.n_components

        back_sel = (scatter_cell_fit.means_[:, 1] < ssc_back_cutoff)|(scatter_cell_fit.means_[:, 0] < fsc_back_cutoff)
        ssc_back_idx = np.where(back_sel)[0].astype(int)
        ssc_back_idx = ssc_back_idx[ssc_back_idx >= back_components]

        cluster = meta.scatter_cell_fit.predict(gmm_data)

        frame['is_cell'] = (cluster >= back_components) & (~np.isin(cluster, ssc_back_idx))

        frame['back_prob'] = scatter_cell_fit.predict_proba(gmm_data)[:, :back_components].sum(axis=1)
        for n in ssc_back_idx:
            frame['back_prob'] += scatter_cell_fit.predict_proba(gmm_data)[:, n]

        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        #data = pickle.load(open(pickle_file, 'rb'))
        
    # then do all the same manipulations with the back_data
    data = back_data
    meta = data.metadata
    pickle_file = back_file

    frame = data.flow_frame
    gmm_data = frame[[f'log_{x_channel}', f'log_{y_channel}']]

    meta._scatter_cell_fit = scatter_cell_fit
    back_components = meta.scatter_back_fit.n_components

    back_sel = (scatter_cell_fit.means_[:, 1] < ssc_back_cutoff)|(scatter_cell_fit.means_[:, 0] < fsc_back_cutoff)
    ssc_back_idx = np.where(back_sel)[0].astype(int)
    ssc_back_idx = ssc_back_idx[ssc_back_idx >= back_components]

    cluster = meta.scatter_cell_fit.predict(gmm_data)

    frame['is_cell'] = (cluster >= back_components) & (~np.isin(cluster, ssc_back_idx))

    frame['back_prob'] = scatter_cell_fit.predict_proba(gmm_data)[:, :back_components].sum(axis=1)
    for n in ssc_back_idx:
        frame['back_prob'] += scatter_cell_fit.predict_proba(gmm_data)[:, n]

    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
        
    # define gated_data for plots and outputs
    gated_data = [data.flow_frame.loc[data.flow_frame['is_cell']] for data in coli_data]
    # 2D histogram plots of the scattering data with the background
    # events gated out, and save array of plots to pdf file:
    if update_progress:
        print('Plotting 2D histograms and scatter plots with background subtracted, ' + str(pd.Timestamp.now().round('s')))
    plt.style.use('classic')

    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    fig, axs = plt.subplots(figure_rows, 4)
    fig.suptitle('Forward and side scatter data from cell samples after auto-gating',
                 y=0.92, verticalalignment='bottom', size=16)
    if axs.ndim == 1:
        axs = np.array([ axs ])
    else:
        axs = axs.flatten()

    for ax, data, name in zip(axs, gated_data, sample_names):
        ax.text(x=0.05, y=0.95, s=name, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)
        ax.hist2d(data[f'log_{x_channel}'], data[f'log_{y_channel}'], bins=200,
                  norm=colors.LogNorm(), rasterized=True)
        ax.autoscale(enable=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f'log10({x_channel})', size=11)
        ax.set_ylabel(f'log10({y_channel})', size=11)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    # scattering plots of the same:
    sns.set()
    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    fig, axs = plt.subplots(figure_rows, 4)
    fig.suptitle('GMM fit results for cell sample data',
                 y=0.92, verticalalignment='bottom', size=16)
    if axs.ndim == 1:
        axs = np.array([ axs ])
    else:
        axs = axs.flatten()

    for ax, data, name in zip(axs, gated_data, sample_names):
        ax.autoscale(enable=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        labels = scatter_cell_fit.predict(data.loc[:, [f'log_{x_channel}', f'log_{y_channel}']]).copy()
        labels[labels < len(f_means)] = 0

        probs = scatter_cell_fit.predict_proba(data.loc[:, [f'log_{x_channel}', f'log_{y_channel}']])
        size = 15 * probs.max(1) ** 2   # square emphasizes differences
        ax.text(x=0.05, y=0.95, s=name, horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
        ax.scatter(data[f'log_{x_channel}'], data[f'log_{y_channel}'], c=labels, cmap='viridis',
                   s=size, rasterized=True)
        ax.set_xlabel(f'log10({x_channel})', size=11)
        ax.set_ylabel(f'log10({y_channel})', size=11)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    for data, gated, file in zip(coli_data, gated_data, coli_files):
        gmm_back_fraction = gated['back_prob'].sum() / gated['back_prob'].size
        #back_frac.append(gmm_back_fraction)
        #singlet_frac.append(singlet_back_fraction)
        
        meta = data.metadata
        meta._gmm_back_fraction = gmm_back_fraction

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    
    if update_progress:
        print('Done. ' + str(pd.Timestamp.now().round('s')))
    pdf.close()


def apply_background_subtract_gate(data_directory,
                                   gate_file_directory,
                                   samples=None,
                                   back_file=None,
                                   gate_file=None,
                                   ssc_back_cutoff=2.9,
                                   fsc_back_cutoff=2.5,
                                   show_plots=True,
                                   update_progress=True,
                                   x_channel='FSC-H',
                                   y_channel='SSC-H'):

    """
    This batch process method applies the background subtraction gating result from gate_file
    to the E. coli and background .fcs_pkl files in data_directory.
    The input files for this method need to be pickled FCSDataFrame objects
    The method applies the Gaussian Mixture Model (GMM) fits from gate_file to both 
    the background data and the cell data.

    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located

    samples : array of str, shape (n,)
        a list or array of cell samples to be gated as a batch
        The method applies the same gating to every sample in the batch.
        The method looks for .fcs_pkl file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the actual data files
        (i.e filenames that do not contain "blank" or "bead");
        
    gate_file : path, or path-like (e.g. str)
        pickled data file to be used for gating.
        If back_file=None then the method uses the first file in an
        automatically detected list of blank files from gate_file_directory.
        
    gate_file_directory : path, or path-like (e.g. str)
        The directory where the method looks for the gate_file.
        
    ssc_back_cutoff : float
        cutoff applied to log10(SSC-H) to reduce chances of spurious background
        clusters being identified as cells. Clusters with mean(log10(SSC-H))<ssc_back_cutoff
        are taken to be background clusters.
        
    fsc_back_cutoff : float
        cutoff applied to log10(FSC-H) to reduce chances of spurious background
        clusters being identified as cells. Clusters with mean(log10(FSC-H))<fsc_back_cutoff
        are taken to be background clusters.
        
    show_plots : Boolean
        If True, the method dynamically shows plots.
        If False, it just saves the plots to a pdf file without showing them. 
    
    x_channel : str
        used to identify the FSC channel name 

    y_channel : str
        used to identify the SSC channel name

    Returns
    -------
    None
    """

    os.chdir(gate_file_directory)  # = "change directory"
    if gate_file is None:
        gate_file = auto_find_files()[0][0]
        
    gate_data = pickle.load(open(gate_file, 'rb'))
    gate_back_fit = gate_data.metadata._scatter_back_fit
    gate_back_file = gate_data.metadata._backfile
    gate_scatter_cell_fit = gate_data.metadata._scatter_cell_fit
    
        
    # make sure to change working directory back to data_directory to load and pickle data files:
    os.chdir(data_directory)

    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()

    pdf_file = 'applied background gating plots.pdf'
    pdf = PdfPages(pdf_file)
    
    coli_files, blank_file_list, bead_file_list = auto_find_files(samples=samples)
    
    if update_progress:
        print("\n".join(['    ' + f for f in coli_files]))
        
    sample_names, start_string = find_sample_names(coli_files)
    
    # flowgatenist is set up to use background data from a blank sample
    # that is ideally run right at the beginneing of an experiment,
    # before any cell samples are run. The background data is used by
    # the gating procedure to detirmine whether an event is more likely
    # to be a background event or a cell event.
    # It is best to use a buffer blank measured before any cell samples
    if back_file is None:
        back_file = blank_file_list[0]

    back_data = pickle.load(open(back_file, 'rb'))
    # Add file for relevant background data to meta data and pickle
    back_data.metadata._backfile = os.path.join(gate_file_directory, gate_back_file)

    with open(back_file, 'wb') as f:
        pickle.dump(back_data, f)

    back_data.flow_frame = back_data.flow_frame.loc[back_data.flow_frame[x_channel] > 0]
    back_data.flow_frame = back_data.flow_frame.loc[back_data.flow_frame[y_channel] > 0]
    back_data.flow_frame[f'log_{x_channel}'] = np.log10(back_data.flow_frame[x_channel])
    back_data.flow_frame[f'log_{y_channel}'] = np.log10(back_data.flow_frame[y_channel])

    # Plot the scattering data for the background sample and save to pdf file
    if update_progress:
        print('Plotting background 2D histogram, ' + str(pd.Timestamp.now().round('s')))
    x_bins = np.linspace(2, 5.5, 200)
    y_bins = np.linspace(2.4, 5.5, 200)
    x = back_data.flow_frame[f'log_{x_channel}']
    y = back_data.flow_frame[f'log_{y_channel}']
    plt.style.use('classic')
    plt.rcParams["figure.figsize"] = [8, 4]
    fig, axs = plt.subplots(1, 2)
    axs[0].hist2d(x, y, bins=[x_bins, y_bins], rasterized=True)
    axs[1].hist2d(x, y, bins=[x_bins, y_bins], norm=colors.LogNorm(), rasterized=True)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    # add fit result to metadata and pickle
    back_data.metadata._scatter_back_fit = gate_back_fit
    with open(back_file, 'wb') as f:
        pickle.dump(back_data, f)

    # Plot a scatter plot of the background data, color-coded to the
    # different GMM components, and with Elipses to show the Gaussians
    back_plot_data = back_data.flow_frame.loc[:, [f'log_{x_channel}', f'log_{y_channel}']].copy()
    if update_progress:
        print('Plotting gmm fit to background data, ' + str(pd.Timestamp.now().round('s')))
    sns.set()
    labels = gate_back_fit.predict(back_plot_data)
    probs = gate_back_fit.predict_proba(back_plot_data)
    size = 10 * probs.max(1) ** 2  # square emphasizes differences
    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(back_plot_data[f'log_{x_channel}'], back_plot_data[f'log_{y_channel}'], c=labels, cmap='viridis', s=size*0.5, rasterized=True)
    axs[1].scatter(back_plot_data[f'log_{x_channel}'], back_plot_data[f'log_{y_channel}'], c=labels, cmap='viridis', s=size*0.5, rasterized=True)
    w_factor = 0.2 / gate_back_fit.weights_.max()
    for pos, covar, w in zip(gate_back_fit.means_, gate_back_fit.covariances_, gate_back_fit.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs[1])
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs[1])

    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)

    # Then load in the actual data files:
    if update_progress:
        print(f'Loading cytometry data files, ' + str(pd.Timestamp.now().round('s')))
    #coli_data = [flow.io.FCSDataFrame(file) for file in coli_files]
    coli_data = [pickle.load(open(file, 'rb')) for file in coli_files]

    for i, (data, file) in enumerate(zip(coli_data, coli_files)):
        data.flow_frame = data.flow_frame.loc[data.flow_frame[x_channel] > 0]
        data.flow_frame = data.flow_frame.loc[data.flow_frame[y_channel] > 0]
        data.flow_frame[f'log_{x_channel}'] = np.log10(data.flow_frame[x_channel])
        data.flow_frame[f'log_{y_channel}'] = np.log10(data.flow_frame[y_channel])

        meta = data.metadata
        
        meta._infile = file
        meta._backfile = back_file
        meta._scatter_back_fit = gate_back_fit
        
        # Mark events with 'is_cell' and 'back_prob' to gate out events that are
        # predicted to belong to the background distribution, and also any that
        # belong to a cluster with mean log(SSC-H)<ssc_back_cutoff:
        pickle_file = file

        frame = data.flow_frame
        gmm_data = frame[[f'log_{x_channel}', f'log_{y_channel}']]

        meta._scatter_cell_fit = gate_scatter_cell_fit
        back_components = meta.scatter_back_fit.n_components

        back_sel = (gate_scatter_cell_fit.means_[:, 1] < ssc_back_cutoff)|(gate_scatter_cell_fit.means_[:, 0] < fsc_back_cutoff)
        ssc_back_idx = np.where(back_sel)[0].astype(int)
        ssc_back_idx = ssc_back_idx[ssc_back_idx >= back_components]

        cluster = gate_scatter_cell_fit.predict(gmm_data)

        frame['is_cell'] = (cluster >= back_components) & (~np.isin(cluster, ssc_back_idx))

        frame['back_prob'] = gate_scatter_cell_fit.predict_proba(gmm_data)[:, :back_components].sum(axis=1)
        for n in ssc_back_idx:
            frame['back_prob'] += gate_scatter_cell_fit.predict_proba(gmm_data)[:, n]

        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        #data = pickle.load(open(pickle_file, 'rb'))
        
    # then do all the same manipulations with the back_data
    data = back_data
    meta = data.metadata
    pickle_file = back_file

    frame = data.flow_frame
    gmm_data = frame[[f'log_{x_channel}', f'log_{y_channel}']]

    meta._scatter_cell_fit = gate_scatter_cell_fit
    back_components = meta.scatter_back_fit.n_components

    back_sel = (gate_scatter_cell_fit.means_[:, 1] < ssc_back_cutoff)|(gate_scatter_cell_fit.means_[:, 0] < fsc_back_cutoff)
    ssc_back_idx = np.where(back_sel)[0].astype(int)
    ssc_back_idx = ssc_back_idx[ssc_back_idx >= back_components]

    cluster = meta.scatter_cell_fit.predict(gmm_data)

    frame['is_cell'] = (cluster >= back_components) & (~np.isin(cluster, ssc_back_idx))

    frame['back_prob'] = gate_scatter_cell_fit.predict_proba(gmm_data)[:, :back_components].sum(axis=1)
    for n in ssc_back_idx:
        frame['back_prob'] += gate_scatter_cell_fit.predict_proba(gmm_data)[:, n]

    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
        
    scatter_data = [data.flow_frame[[f'log_{x_channel}', f'log_{y_channel}']] for data in coli_data]

    # Plot the 2D histograms for all the E. coli fcs files, and save the
    # plots to the pdf file
    if update_progress:
        print('Plotting 2D histograms for scattering data from each file, ' + str(pd.Timestamp.now().round('s')))
    figure_rows = -(len(scatter_data)//-4)
    x_bins = np.linspace(2, 5.5, 200)
    y_bins = np.linspace(2.4, 5., 200)
    plt.style.use('classic')
    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    
    fig, axs = plt.subplots(figure_rows, 4)
    if axs.ndim == 1:
        axs = np.array([ axs ])
    for i, data in enumerate(scatter_data):
        axs[i//4, i%4].set_xlim(left=2, right=5.5)
        axs[i//4, i%4].set_ylim(bottom=2.4, top=5.)
        axs[i//4, i%4].text(2.1, 4.8, sample_names[i])
        axs[i//4, i%4].hist2d(data[f'log_{x_channel}'],
           data[f'log_{y_channel}'], bins=[x_bins, y_bins],
           norm=colors.LogNorm(), rasterized=True);
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
        
    # define gated_data for plots and outputs
    gated_data = [data.flow_frame.loc[data.flow_frame['is_cell']] for data in coli_data]
    # 2D histogram plots of the scattering data with the background
    # events gated out, and save array of plots to pdf file:
    if update_progress:
        print('Plotting 2D histograms and scatter plots with background subtracted, ' + str(pd.Timestamp.now().round('s')))
    plt.style.use('classic')

    x_bins = np.linspace(2, 5.5, 200)
    y_bins = np.linspace(2.4, 5., 200)
    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    fig, axs = plt.subplots(figure_rows, 4)
    if axs.ndim == 1:
        axs = np.array([ axs ])

    for i, data in enumerate(gated_data):
        axs[i//4, i%4].set_xlim(left=2, right=5.5)
        axs[i//4, i%4].set_ylim(bottom=2.4, top=5)
        axs[i//4, i%4].text(2.1, 4.8, sample_names[i])
        axs[i//4, i%4].hist2d(data.loc[:, f'log_{x_channel}'], data.loc[:, f'log_{y_channel}'], bins=[x_bins, y_bins],
                             norm=colors.LogNorm(), rasterized=True)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    # scattering plots of the same:
    sns.set()
    plt.rcParams["figure.figsize"] = [16, 4*figure_rows]
    fig, axs = plt.subplots(figure_rows, 4)
    if axs.ndim == 1:
        axs = np.array([ axs ])

    for i, data in enumerate(gated_data):
        labels = gate_scatter_cell_fit.predict(data.loc[:, [f'log_{x_channel}', f'log_{y_channel}']]).copy()
        labels[labels < len(gate_back_fit.means_)] = 0

        probs = gate_scatter_cell_fit.predict_proba(data.loc[:, [f'log_{x_channel}', f'log_{y_channel}']])
        size = 5 * probs.max(1) ** 2   # square emphasizes differences

        axs[i//4, i%4].set_xlim(left=2, right=6)
        axs[i//4, i%4].set_ylim(bottom=2.4, top=5)
        axs[i//4, i%4].text(2.1, 4.8, sample_names[i])
        axs[i//4, i%4].scatter(data[f'log_{x_channel}'], data[f'log_{y_channel}'], c=labels, cmap='viridis', s=size*3, rasterized=True)
    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    
    for data, gated, file in zip(coli_data, gated_data, coli_files):
        gmm_back_fraction = gated['back_prob'].sum() / gated['back_prob'].size
        #back_frac.append(gmm_back_fraction)
        #singlet_frac.append(singlet_back_fraction)
        
        meta = data.metadata
        meta._gmm_back_fraction = gmm_back_fraction

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    
    if update_progress:
        print('Done. ' + str(pd.Timestamp.now().round('s')))
    pdf.close()


def singlet_gating_width(data_directory,
                         top_directory=top_directory,
                         samples=None,
                         back_file=None,
                         singlet_clusters=8,
                         random_singlet_inits=50,
                         singlet_width_cutoff=None,
                         doublet_width_cutoff=None,
                         ssc_adjustment=900,
                         num_memory_inits=5,
                         save_analysis_memory=True,
                         cell_type='E. coli',
                         max_events=100000,
                         init_events=30000,
                         update_progress=True,
                         show_plots=True,
                         height_channel='SSC-H',
                         width_channel='SSC-W'):

    """
    This batch process method performs a gating to select
    flow cytometry events that are most likely to be single cell events.
    The input files for this method need to be pickled FCSDataFrame objects.
    The input data needs to first be gated to select cell events (e.g. background_subtract_gating).
    The method uses Gaussian Mixture Model (GMM) fits to the data from the scattering channels. 
    It runs random_singlet_inits randomly initialized GMM fits,
    and num_memory_inits GMM fits initialized from memory (recently completed
    sucessful GMM fits to similar samples).
    
    This version of the singlet gating method uses the SSC Height vs. the SSC Width for the 2D GMM fits

    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located
        
    top_directory : path, or path-like (e.g. str)
        indicates the top level directory where the automated analysis will
        expect to find the "Flow Cytometry Analysis Memory" directory

    samples : array of str, shape (n,)
        a list or array of samples to be converted
        The method looks for .fcs_pkl file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the cell data files
        (i.e filenames that do not contain "blank" or "bead");
        
    back_file : path, or path-like (e.g. str)
        pickled blank data file to also apply singlet gating to.
        If back_file=None then the method uses the first file in an
        automatically detected list of blank files.
        
    singlet_clusters : int
        the number of GMM clusters used for GMM fitting for singlet gating.
        
    back_init : int
        the number of initializations to run for the GMM fit to the background data
        If back_init=None, it is automatically chosen.
        
    random_singlet_inits : int
        the number of randomly (k-means) initialized GMM fits to run for the singlet fit
        
    singlet_width_cutoff : float
        cutoff applied to cluster means for identification of singlet clusters.
        Clusters with mean(SSC-W)<singlet_width_cutoff are taken to be singlets.
        
    doublet_width_cutoff : float
        cutoff applied to cluster means for identification of singlet clusters
        Clusters with mean(SSC-W)>singlet_width_cutoff and mean(SSC-W)<doublet_width_cutoff
        are ingored when assigning events to thier most likely clusters.
    
    ssc_adjustment : float
        the number subtracted from SSC-H before log transforming to make it plot more
        linearly vs. SSC-W, for better GMM clustering
        
    num_memory_inits : int
        the number of GMMs initialized from memory (previous GMM fits) for cell data
        
    save_analysis_memory : Boolean
        if True, then GMM fits to cell data are saved in the memory store to
        be used for initialization of future GMM fits
        
    cell_type : str
        used to identify the correct GMM memory initialization directory
        
    max_events : int
        the maximum number of cytometry events from each cell data file to use for the GMM fit
        This parameter is used to avoid running out of memory, since the method concatenates
        the data from all the cell data files for use in the GMM fit.
        
    update_progress : Boolean
        if True, the method prints status updates as it goes.
        
    show_plots : Boolean
        If True, the method dynamically shows plots.
        If False, it just saves the plots to a pdf file without showing them.

    height_channel : str
        used to identify the SSC height channel name 

    width_channel : str
        used to identify the SSC width channel name

     Returns
    -------
    None
    """

    os.chdir(data_directory)  # = "change directory"
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()

    pdf_file = 'singlet gating plots.pdf'
    pdf = PdfPages(pdf_file)

    a1, a2, a3 = find_analysis_memory(data_directory=data_directory,
                                      top_directory=top_directory,
                                      update_progress=update_progress,
                                      num_memory_inits=num_memory_inits,
                                      cell_type=cell_type,
                                      save_analysis_memory=save_analysis_memory)
    use_analysis_memory = a1
    save_analysis_memory = a2
    cell_mem_dir = a3
        
    # make sure to change working directory back to data_directory to load and pickle data files:
    os.chdir(data_directory)

    if update_progress:
        print('Starting singlet_fit gmm... ' + str(pd.Timestamp.now().round('s')))

    coli_files, blank_file_list, bead_file_list = auto_find_files(samples=samples)
    
    if update_progress:
        print("\n".join(['    ' + f for f in coli_files]))
    
    sample_names, start_string = find_sample_names(coli_files)

    # flowgatenist is set up to use background data from a blank sample
    # that is ideally run right at the beginneing of an experiment,
    # before any cell samples are run. The background data is used by
    # the gating procedure to detirmine whether an event is more likely
    # to be a background event or a cell event.
    # It is best to use a buffer blank measured before any cell samples
    if back_file is None:
        back_file = blank_file_list[0]
    back_data = pickle.load(open(back_file, 'rb'))
    if update_progress:
        print('    ' + back_file)

    # load in the actual data files:
    if update_progress:
        print(f'Loading {cell_type} data files, ' + str(pd.Timestamp.now().round('s')))
    #coli_data = [flow.io.FCSDataFrame(file) for file in coli_files]
    coli_data = [pickle.load(open(file, 'rb')) for file in coli_files]
    
    # Start singlet gating:
    # ssc_adjustment is an offset to use before log transforming the SSC-H
    # data to make it plot more linearly vs. SSC-W
    for data in coli_data:
        frame = data.flow_frame
        frame[f'adj_log_{height_channel}'] = np.log10(frame[height_channel] - ssc_adjustment)
        frame[f'log_{width_channel}'] = np.log10(frame[width_channel])
    # then do all the same manipulations with the back_data
    data = back_data
    frame = data.flow_frame
    frame[f'adj_log_{height_channel}'] = np.log10(frame[height_channel] - ssc_adjustment)

    singlet_gmm_data2 = [data.flow_frame.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}', 'is_cell']].copy() for data in coli_data]
    
    # Note: this for loop is done in the non-Pythonic way, using the itterator, i,
    # becasue that is what it takes to get 2nd line to work to pick out only events
    # for which is_cell=True
    for i, data in enumerate(singlet_gmm_data2):
        singlet_gmm_data2[i] = singlet_gmm_data2[i].loc[np.isfinite(singlet_gmm_data2[i][f'adj_log_{height_channel}'])]
        singlet_gmm_data2[i] = singlet_gmm_data2[i].loc[singlet_gmm_data2[i]['is_cell']]
        singlet_gmm_data2[i] = singlet_gmm_data2[i].loc[singlet_gmm_data2[i][f'log_{width_channel}'] > 0]

    singlet_gmm_data2 = [data.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']].copy() for data in singlet_gmm_data2]
    singlet_gmm_data2 = [data[:max_events] for data in singlet_gmm_data2]
    #singlet_gmm_data4 = pd.concat(singlet_gmm_data2)
    singlet_gmm_data4 = pd.concat(singlet_gmm_data2, ignore_index=True)
    singlet_gmm_data3 = singlet_gmm_data4.sample(n=init_events)
    
    singlet_gmm_data2 = None
        
    # First run fit with random initializations
    if update_progress:
        print('Running sub-sampled randomly initialized singlet_fit gmms... ' + str(pd.Timestamp.now().round('s')))
    # Start with random initializations using the randomly sampled data (singlet_gmm_data3),
    gmm_singlet3 = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                            covariance_type='full', n_init=random_singlet_inits)
    singlet_fit3 = gmm_singlet3.fit(singlet_gmm_data3)
    
    means_init = singlet_fit3.means_
    weights_init = singlet_fit3.weights_
    precisions_init = [np.linalg.inv(cov) for cov in singlet_fit3.covariances_]
    
    # Then use the best result to initialize a fit to the full data (singlet_gmm_data4)
    # The fits can take a while with the full data set:
    if update_progress:
        print('Running final randomly initialized singlet_fit gmm... ' + str(pd.Timestamp.now().round('s')))
    gmm4 = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                    covariance_type='full',
                                    n_init=1,
                                    weights_init=weights_init,
                                    means_init=means_init,
                                    precisions_init=precisions_init)
    
    singlet_fit = gmm4.fit(singlet_gmm_data4)
    
    #gmm_singlet = nist_gmm.GaussianMixture(n_components=singlet_clusters, covariance_type='full', n_init=random_singlet_inits)
    #singlet_fit = gmm_singlet.fit(singlet_gmm_data4.loc[:, ['log_ssc_w', 'adj_log_ssc']])
    
    random_lower_bound = singlet_fit.lower_bound_
    
    # Then run fits with initializations from memory:
    if use_analysis_memory:
        os.chdir(cell_mem_dir)

        singlet_fit_mem_files = glob.glob("*singlet_fit*.pkl")
        singlet_fit_mem_files.sort(key=os.path.getmtime)
        singlet_fit_mem_files = singlet_fit_mem_files[::-1] # reverse order to get newest first
        #singlet_fit_mem_files = singlet_fit_mem_files[:num_memory_inits]
        singlet_fit_inits = [pickle.load(open(file, 'rb')) for file in singlet_fit_mem_files]
        
        # Only use memory inits with singlet_fit_inits
        #     that matches the value for this call of the method
        num_singlet_clusters_mem = [init.n_components for init in singlet_fit_inits]
        mem_inits_to_use = [i for i, x in enumerate(num_singlet_clusters_mem) if x == singlet_clusters]
        
        singlet_fit_inits = [singlet_fit_inits[i] for i in mem_inits_to_use]
        singlet_fit_inits = singlet_fit_inits[:num_memory_inits]
        
        if (len(singlet_fit_inits) > 0):
            if update_progress:
                print('Running ' + str(len(singlet_fit_inits)) + ' gmm fits initialized from memory... ' + str(pd.Timestamp.now().round('s')))
                
            for i, old_fit in enumerate(singlet_fit_inits):
                means_init = old_fit.means_
                weights_init = old_fit.weights_
                precisions_init = [np.linalg.inv(cov) for cov in old_fit.covariances_]
                
                gmm_singlet_mem = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                                 covariance_type='full',
                                                 n_init=1,
                                                 weights_init=weights_init,
                                                 means_init=means_init,
                                                 precisions_init=precisions_init)
                singlet_fit_mem = gmm_singlet_mem.fit(singlet_gmm_data4.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']])
                
                if i==0:
                    best_mem_gmm = singlet_fit_mem
                    best_mem_gmm_init = old_fit
                else:
                    if (singlet_fit_mem.lower_bound_ > best_mem_gmm.lower_bound_):
                        best_mem_gmm = singlet_fit_mem
                        best_mem_gmm_init = old_fit
                
        else:
            print('Warning: no memory fits with matching num_back_clusters and num_cell_clusters')

    # Then compare Memory initialized fits to random initialized fits
    # and take the result with the highest lower_bound_
    #print('random: ' + str(random_lower_bound) + ', memory: ' + str(best_mem_gmm.lower_bound_))
    os.chdir(cell_mem_dir)
    if ((use_analysis_memory and len(singlet_fit_inits)>0) and (best_mem_gmm.lower_bound_ > random_lower_bound)):
        singlet_fit = best_mem_gmm
        # increment gmm.used_as_init and re-pickle
        best_mem_gmm_init.used_as_init += 1
        print('    using best gmm from memory to initialize, ' + best_mem_gmm_init.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(best_mem_gmm_init.memory_file, 'wb') as f:
                pickle.dump(best_mem_gmm_init, f)
    else:
        singlet_fit.used_as_init = 1
        data_date = coli_data[0].metadata.acquisition_start_time.date()
        singlet_fit.memory_file = 'singlet_fit.' + data_date.strftime('%Y-%m-%d') + '.pkl'
        j = 1
        while (os.path.isfile(singlet_fit.memory_file)):
            singlet_fit.memory_file = 'singlet_fit.' + data_date.strftime('%Y-%m-%d') + '.' + str(j) + '.pkl'
            j += 1
        print('    using new randomly initialized gmm, ' + singlet_fit.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(singlet_fit.memory_file, 'wb') as f:
                pickle.dump(singlet_fit, f)
                
    os.chdir(data_directory)
    
    if update_progress:
        print('                 ... done, ' + str(pd.Timestamp.now().round('s')))

    if update_progress:
        print('Plotting singlet_fit results, ' + str(pd.Timestamp.now().round('s')))
    plt.rcParams["figure.figsize"] = [8, 8]
    fig, axs = plt.subplots(1, 1)
    fig.suptitle('GMM fit results for singlet gating with width',
                 y=0.92, verticalalignment='bottom', size=16)
    num_points = 10000

    data = singlet_gmm_data4

    x = data[f'log_{width_channel}']  # [:num_points]
    y = data[f'adj_log_{height_channel}']  # [:num_points])

    labels = singlet_fit.predict(data.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']][:num_points]).copy()
    probs = singlet_fit.predict_proba(data.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']][:num_points])
    size = 40 * probs.max(1) ** 2   # square emphasizes differences

    #axs.set_xlim(-10, 200)
    axs.scatter(x[:num_points], y[:num_points], s=size, c=labels, cmap='viridis')
    axs.set_xlabel(f'log10({width_channel})')
    axs.set_ylabel(f'log10({height_channel} - {ssc_adjustment})')
    w_factor = 0.2 / singlet_fit.weights_.max()
    for pos, covar, w in zip(singlet_fit.means_, singlet_fit.covariances_, singlet_fit.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs)
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs)

    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    if update_progress:
        print('Applying singlet gate, ' + str(pd.Timestamp.now().round('s')))
    # Automatically find cutoffs for which clusters are considered singlet and doublet, etc.
    if singlet_width_cutoff is None:
        df = pd.DataFrame({'SSC-W_means' : singlet_fit.means_[:, 0], 'weights' : singlet_fit.weights_})
        singlet_width_cutoff = df.loc[df['weights'].idxmax()]['SSC-W_means'] * 1.4
        print(f'    Automatically detirmined singlet_width_cutoff = {singlet_width_cutoff}')
    if doublet_width_cutoff is None:
        df = pd.DataFrame({'SSC-W_means' : singlet_fit.means_[:, 0], 'weights' : singlet_fit.weights_})
        doublet_width_cutoff = df.loc[df['weights'].idxmax()]['SSC-W_means'] * 1.9
        print(f'    Automatically detirmined doublet_width_cutoff = {doublet_width_cutoff}')
    
    singlets = np.where(singlet_fit.means_[:, 0] <= singlet_width_cutoff)
    not_single_or_multiple = np.where((singlet_fit.means_[:, 0] > singlet_width_cutoff) &
                                      (singlet_fit.means_[:, 0] < doublet_width_cutoff))

    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('GMM fit results for singlet gating with width',
                 y=0.92, verticalalignment='bottom', size=16)
    num_points = 10000

    #data = singlet_gmm_data4

    #x = data['log_ssc_w']  # [:num_points]
    #y = data['adj_log_ssc']  # [:num_points])

    probs = singlet_fit.predict_proba(data.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']][:num_points])
    probs2 = probs.copy()
    probs2[:, not_single_or_multiple] = 0
    probs2 = probs2 / (probs2.sum(1)[:, np.newaxis])
    size = 40 * probs.max(1) ** 2   # square emphasizes differences
    labels = probs.argmax(1)
    size2 = 40 * probs2.max(1) ** 2   # square emphasizes differences
    labels2 = probs2.argmax(1)
    labels2[np.isin(labels2, singlets)] = -5
    
    axs[0].get_shared_x_axes().join(axs[0], axs[1])

    #axs[0].set_xlim(-10, 200)
    axs[0].scatter(x[:num_points], y[:num_points], s=size, c=labels, cmap='viridis')
    #axs[1].set_xlim(-10, 200)
    axs[1].scatter(x[:num_points], y[:num_points], s=size2, c=labels2, cmap='viridis')
    
    for ax in axs:
        ax.set_xlabel(f'log10({width_channel})')
        ax.set_ylabel(f'log10({height_channel} - {ssc_adjustment})')

    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    pd.options.mode.use_inf_as_na = True
    for i, (data, file) in enumerate(zip(coli_data, coli_files)):
        frame = data.flow_frame
        meta = data.metadata
        gmm_data = frame[[f'log_{width_channel}', f'adj_log_{height_channel}']].copy().fillna(0)

        meta._scatter_singlet_fit = singlet_fit

        probs = singlet_fit.predict_proba(gmm_data)
        probs2 = probs.copy()
        probs2[:, not_single_or_multiple] = 0
        probs2 = probs2 / (probs2.sum(1)[:, np.newaxis])
        labels2 = probs2.argmax(1)

        is_singlet_1 = np.isin(labels2, singlets)
        is_singlet_2 = frame['is_cell']
        is_singlet_3 = np.isfinite(frame[f'adj_log_{height_channel}'])
        is_singlet_4 = frame[width_channel] > 0

        frame['is_singlet'] = is_singlet_1 & is_singlet_2 & is_singlet_3 & is_singlet_4

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
            
    # then do all the same manipulations with the back_data
    data = back_data
    frame = data.flow_frame
    meta = data.metadata
    frame[f'log_{width_channel}'] = np.log10(frame[width_channel])
    gmm_data = frame[[f'log_{width_channel}', f'adj_log_{height_channel}']].copy().fillna(0)

    meta._scatter_singlet_fit = singlet_fit

    probs = singlet_fit.predict_proba(gmm_data)
    probs2 = probs.copy()
    probs2[:, not_single_or_multiple] = 0
    probs2 = probs2 / (probs2.sum(1)[:, np.newaxis])
    labels2 = probs2.argmax(1)

    is_singlet_1 = np.isin(labels2, singlets)
    is_singlet_2 = frame['is_cell']
    is_singlet_3 = np.isfinite(frame[f'adj_log_{height_channel}'])
    is_singlet_4 = frame[width_channel] > 0

    frame['is_singlet'] = is_singlet_1 & is_singlet_2 & is_singlet_3 & is_singlet_4

    pickle_file = back_file
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    # Done with back_data
    
    if update_progress:
        print('Plotting scatter plots showing singlets, ' + str(pd.Timestamp.now().round('s')))
    sns.set()
    plt.rcParams["figure.figsize"] = [16, 4*len(coli_data)]
    fig, axs = plt.subplots(len(coli_data), 2)
    fig.suptitle('Data for cell samples before (left) and after (right) automatic singlet gating with width',
                 y=0.903, verticalalignment='bottom', size=16)
    if axs.ndim == 1:
        axs = np.array([ axs ])

    for i, coli in enumerate(coli_data):
        axs[i, 0].get_shared_x_axes().join(axs[i, 0], axs[i, 1])
        axs[i, 0].get_shared_y_axes().join(axs[i, 0], axs[i, 1])
        data = coli.flow_frame
        data = data[data['is_cell']]
        data = data[data[width_channel] > 0]
        data2 = data.loc[data['is_singlet']].copy()

        x = data[f'log_{width_channel}']  # [:num_points]
        y = data[f'log_{height_channel}']  # [:num_points])
        x2 = data2[f'log_{width_channel}']  # [:num_points]
        y2 = data2[f'log_{height_channel}']  # [:num_points])

        probs = singlet_fit.predict_proba(data.loc[:, [f'log_{width_channel}', f'adj_log_{height_channel}']].fillna(0)[:num_points])
        probs2 = probs.copy()
        probs2[:, not_single_or_multiple] = 0
        probs2 = probs2 / (probs2.sum(1)[:, np.newaxis])
        size = 5 * probs.max(1) ** 2   # square emphasizes differences
        labels = probs.argmax(1)
        size2 = 5 * probs2.max(1) ** 2   # square emphasizes differences
        labels2 = probs2.argmax(1)
        labels2[np.isin(labels2, singlets)] = -5
        
        for ax in axs[i]:
            ax.set_xlim(-10, 150)
            ax.set_ylim(2.5, 5)
            ax.text(0.05, 0.95, sample_names[i],
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        #axs[i, 0].scatter(x2[:num_points], y2[:num_points], s=10, rasterized=True)
        axs[i, 1].hist2d(x2, y2, bins=100, norm=colors.LogNorm(), rasterized=True)
        #axs[i, 1].scatter(x[:num_points], y[:num_points], s=size2*2, c=labels2, cmap='viridis', rasterized=True)
        axs[i, 0].hist2d(x, y, bins=100, norm=colors.LogNorm(), rasterized=True)
    for ax in axs.flatten():
        ax.set_xlabel(f'log10({width_channel})')
        ax.set_ylabel(f'log10({height_channel})')

    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    pdf.close()

    singlet_gated_data = [data.flow_frame.loc[(data.flow_frame['is_cell']) & (data.flow_frame['is_singlet'])] for data in coli_data]

    #for back_frac, singlet_frac, data, gated, singlet in zip(old_back_fractions, old_singlet_back_fractions, coli_data, gated_data, singlet_gated_data):
    for data, singlet, file in zip(coli_data, singlet_gated_data, coli_files):
        singlet_back_fraction = singlet['back_prob'].sum() / singlet['back_prob'].size
        #back_frac.append(gmm_back_fraction)
        #singlet_frac.append(singlet_back_fraction)
        
        meta = data.metadata
        meta._gmm_singlet_back_fraction = singlet_back_fraction

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    
    if update_progress:
        print('Done. ' + str(pd.Timestamp.now().round('s')))


def singlet_gating(data_directory,
                   top_directory=top_directory,
                   samples=None,
                   back_file=None,
                   singlet_clusters=6,
                   random_singlet_inits=100,
                   singlet_mean_center=0,
                   singlet_mean_cutoff=0.15,
                   singlet_stdv_cutoff=0.1,
                   num_memory_inits=5,
                   save_analysis_memory=True,
                   cell_type='E. coli',
                   max_events=100000,
                   init_events=30000,
                   update_progress=True,
                   show_plots=True,
                   y_channel_h='SSC-H',
                   y_channel_a='SSC-A'):

    """
    This batch process method performs a gating to select
    flow cytometry events that are most likely to be single cell events.
    The input files for this method need to be pickled FCSDataFrame objects.
    The input data needs to first be gated to select cell events (e.g. background_subtract_gating).
    The method uses Gaussian Mixture Model (GMM) fits to the data from the scattering channels. 
    It runs random_singlet_inits randomly initialized GMM fits,
    and num_memory_inits GMM fits initialized from memory (recently completed
    sucessful GMM fits to similar samples).
    
    This version of the singlet gating method uses the SSC Height vs. the ratio of the SSC Area to Height for the 2D GMM fits

    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located
        
    top_directory : path, or path-like (e.g. str)
        indicates the top level directory where the automated analysis will
        expect to find the "Flow Cytometry Analysis Memory" directory

    samples : array of str, shape (n,)
        a list or array of samples to be converted
        The method looks for .fcs_pkl file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the cell data files
        (i.e filenames that do not contain "blank" or "bead");
        
    back_file : path, or path-like (e.g. str)
        pickled blank data file to also apply singlet gating to.
        If back_file=None then the method uses the first file in an
        automatically detected list of blank files.
        
    singlet_clusters : int
        the number of GMM clusters used for GMM fitting for singlet gating.
        
    back_init : int
        the number of initializations to run for the GMM fit to the background data
        If back_init=None, it is automatically chosen.
        
    random_singlet_inits : int
        the number of randomly (k-means) initialized GMM fits to run for the singlet fit
        
    singlet_mean_center : float
        center value for log-ratio of SSC area to height ('singlet_difference') for identification of singlet clusters
        Clusters with -singlet_mean_cutoff < mean('singlet_difference')-singlet_mean_center < singlet_mean_cutoff
        and stdv('singlet_difference') < singlet_stdv_cutoff are taken to be singlets.
        
    singlet_mean_cutoff : float
        cutoff applied to cluster means for identification of singlet clusters
        Clusters with -singlet_mean_cutoff < mean('singlet_difference')-singlet_mean_center < singlet_mean_cutoff
        and stdv('singlet_difference') < singlet_stdv_cutoff are taken to be singlets.
        
    singlet_stdv_cutoff : float
        cutoff applied to cluster standard deviationsa for identification of singlet clusters
        Clusters with -singlet_mean_cutoff < mean('singlet_difference')-singlet_mean_center < singlet_mean_cutoff
        and stdv('singlet_difference') < singlet_stdv_cutoff are taken to be singlets.
        
    num_memory_inits : int
        the number of GMMs initialized from memory (previous GMM fits) for cell data
        
    save_analysis_memory : Boolean
        if True, then GMM fits to cell data are saved in the memory store to
        be used for initialization of future GMM fits
        
    cell_type : str
        used to identify the correct GMM memory initialization directory
        
    max_events : int
        the maximum number of cytometry events from each cell data file to use for the GMM fit
        This parameter is used to avoid running out of memory, since the method concatenates
        the data from all the cell data files for use in the GMM fit.
        
    update_progress : Boolean
        if True, the method prints status updates as it goes.
        
    show_plots : Boolean
        If True, the method dynamically shows plots.
        If False, it just saves the plots to a pdf file without showing them.
        
    y_channel_h : str, default='SSC-H'
        Name of side-scatter height channel
        
    y_channel_a : str, default='SSC-A'
        Name of side-scatter area channel

    Returns
    -------
    None
    """

    os.chdir(data_directory)  # = "change directory"
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()

    pdf_file = 'singlet gating plots.pdf'
    pdf = PdfPages(pdf_file)

    a1, a2, a3 = find_analysis_memory(data_directory=data_directory,
                                      top_directory=top_directory,
                                      update_progress=update_progress,
                                      num_memory_inits=num_memory_inits,
                                      cell_type=cell_type,
                                      save_analysis_memory=save_analysis_memory)
    use_analysis_memory = a1
    save_analysis_memory = a2
    cell_mem_dir = a3
        
    # make sure to change working directory back to data_directory to load and pickle data files:
    os.chdir(data_directory)

    if update_progress:
        print('Starting singlet_fit gmm... ' + str(pd.Timestamp.now().round('s')))

    coli_files, blank_file_list, bead_file_list = auto_find_files(samples=samples)
    
    if update_progress:
        print("\n".join(['    ' + f for f in coli_files]))
            

    # flowgatenist is set up to use background data from a blank sample
    # that is ideally run right at the beginneing of an experiment,
    # before any cell samples are run. The background data is used by
    # the gating procedure to detirmine whether an event is more likely
    # to be a background event or a cell event.
    # It is best to use a buffer blank measured before any cell samples
    if back_file is None:
        back_file = blank_file_list[0]
    back_data = pickle.load(open(back_file, 'rb'))
    if update_progress:
        print('    ' + back_file)

    # load in the actual data files:
    if update_progress:
        print(f'Loading {cell_type} data files, ' + str(pd.Timestamp.now().round('s')))
    #coli_data = [flow.io.FCSDataFrame(file) for file in coli_files]
    coli_data = [pickle.load(open(file, 'rb')) for file in coli_files]
    
    sample_names, start_string = find_sample_names(coli_files)
    
    # Start singlet gating:
    # singlet_difference is the log of the ratio between the SSC-A and SSC-H.
    # On the Attune system, the scale of SSC-A and SSC-H are set so that they are approximately equal for singlet events
    # So the singlet cluster is centered on singlet_difference = 0
    # For other cytometers it might be centered on a different value
    for data in coli_data:
        data.flow_frame = data.flow_frame.loc[data.flow_frame[y_channel_a] > 0]
        data.flow_frame['singlet_difference'] = np.log10(data.flow_frame[y_channel_a]) - data.flow_frame[f'log_{y_channel_h}']
    
    # then do all the same manipulations with the back_data
    back_data.flow_frame = back_data.flow_frame.loc[back_data.flow_frame[y_channel_a] > 0]
    back_data.flow_frame['singlet_difference'] = np.log10(back_data.flow_frame[y_channel_a]) - back_data.flow_frame[f'log_{y_channel_h}']

    # Fit a GMM to the log_ssc vs. singlet_difference
    singlet_gmm_data2 = [data.flow_frame.loc[:, ['singlet_difference', f'log_{y_channel_h}', 'is_cell']].copy() for data in coli_data]
    
    # Note: this for loop is done in the non-Pythonic way, using the itterator, i,
    # becasue that is what it takes to get 2nd line to work to pick out only events
    # for which is_cell=True
    for i, data in enumerate(singlet_gmm_data2):
        singlet_gmm_data2[i] = singlet_gmm_data2[i].loc[np.isfinite(singlet_gmm_data2[i]['singlet_difference'])]
        singlet_gmm_data2[i] = singlet_gmm_data2[i].loc[singlet_gmm_data2[i]['is_cell']]

    singlet_gmm_data2 = [data.loc[:, ['singlet_difference', f'log_{y_channel_h}']].copy() for data in singlet_gmm_data2]
    singlet_gmm_data2 = [data[:max_events] for data in singlet_gmm_data2]
    #singlet_gmm_data4 = pd.concat(singlet_gmm_data2)
    singlet_gmm_data4 = pd.concat(singlet_gmm_data2, ignore_index=True)
    singlet_gmm_data3 = singlet_gmm_data4.sample(n=init_events)
    
    singlet_gmm_data2 = None
        
    # First run fit with random initializations
    if update_progress:
        print('Running sub-sampled randomly initialized singlet_fit gmms... ' + str(pd.Timestamp.now().round('s')))
    # Start with random initializations using the randomly sampled data (singlet_gmm_data3),
    gmm_singlet3 = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                            covariance_type='full', n_init=random_singlet_inits)
    singlet_fit3 = gmm_singlet3.fit(singlet_gmm_data3)
    
    means_init = singlet_fit3.means_
    weights_init = singlet_fit3.weights_
    precisions_init = [np.linalg.inv(cov) for cov in singlet_fit3.covariances_]
    
    # Then use the best result to initialize a fit to the full data (singlet_gmm_data4)
    # The fits can take a while with the full data set:
    if update_progress:
        print('Running final randomly initialized singlet_fit gmm... ' + str(pd.Timestamp.now().round('s')))
    gmm4 = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                    covariance_type='full',
                                    n_init=1,
                                    weights_init=weights_init,
                                    means_init=means_init,
                                    precisions_init=precisions_init)
    
    singlet_fit = gmm4.fit(singlet_gmm_data4)
    
    #gmm_singlet = nist_gmm.GaussianMixture(n_components=singlet_clusters, covariance_type='full', n_init=random_singlet_inits)
    #singlet_fit = gmm_singlet.fit(singlet_gmm_data4.loc[:, ['log_ssc_w', 'adj_log_ssc']])
    
    random_lower_bound = singlet_fit.lower_bound_
    
    # Then run fits with initializations from memory:
    if use_analysis_memory:
        os.chdir(cell_mem_dir)

        singlet_fit_mem_files = glob.glob("*singlet_fit*.pkl")
        singlet_fit_mem_files.sort(key=os.path.getmtime)
        singlet_fit_mem_files = singlet_fit_mem_files[::-1] # reverse order to get newest first
        #singlet_fit_mem_files = singlet_fit_mem_files[:num_memory_inits]
        singlet_fit_inits = [pickle.load(open(file, 'rb')) for file in singlet_fit_mem_files]
        
        # Only use memory inits with singlet_fit_inits
        #     that matches the value for this call of the method
        num_singlet_clusters_mem = [init.n_components for init in singlet_fit_inits]
        mem_inits_to_use = [i for i, x in enumerate(num_singlet_clusters_mem) if x == singlet_clusters]
        
        singlet_fit_inits = [singlet_fit_inits[i] for i in mem_inits_to_use]
        singlet_fit_inits = singlet_fit_inits[:num_memory_inits]
        
        if (len(singlet_fit_inits) > 0):
            if update_progress:
                print('Running ' + str(len(singlet_fit_inits)) + ' gmm fits initialized from memory... ' + str(pd.Timestamp.now().round('s')))
                
            for i, old_fit in enumerate(singlet_fit_inits):
                means_init = old_fit.means_
                weights_init = old_fit.weights_
                precisions_init = [np.linalg.inv(cov) for cov in old_fit.covariances_]
                
                gmm_singlet_mem = nist_gmm.GaussianMixture(n_components=singlet_clusters,
                                                           covariance_type='full',
                                                           n_init=1,
                                                           weights_init=weights_init,
                                                           means_init=means_init,
                                                           precisions_init=precisions_init)
                singlet_fit_mem = gmm_singlet_mem.fit(singlet_gmm_data4)
                #singlet_fit_mem = gmm_singlet_mem.fit(singlet_gmm_data4.loc[:, ['log_ssc_w', 'adj_log_ssc']])
                
                if i==0:
                    best_mem_gmm = singlet_fit_mem
                    best_mem_gmm_init = old_fit
                else:
                    if (singlet_fit_mem.lower_bound_ > best_mem_gmm.lower_bound_):
                        best_mem_gmm = singlet_fit_mem
                        best_mem_gmm_init = old_fit
                
        else:
            print('Warning: no memory fits with matching num_back_clusters and num_cell_clusters')

    # Then compare Memory initialized fits to random initialized fits
    # and take the result with the highest lower_bound_
    #print('random: ' + str(random_lower_bound) + ', memory: ' + str(best_mem_gmm.lower_bound_))
    os.chdir(cell_mem_dir)
    if ((use_analysis_memory and len(singlet_fit_inits)>0) and (best_mem_gmm.lower_bound_ > random_lower_bound)):
        singlet_fit = best_mem_gmm
        # increment gmm.used_as_init and re-pickle
        best_mem_gmm_init.used_as_init += 1
        print('    using best gmm from memory to initialize, ' + best_mem_gmm_init.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(best_mem_gmm_init.memory_file, 'wb') as f:
                pickle.dump(best_mem_gmm_init, f)
    else:
        singlet_fit.used_as_init = 1
        data_date = coli_data[0].metadata.acquisition_start_time.date()
        singlet_fit.memory_file = 'singlet_fit.' + data_date.strftime('%Y-%m-%d') + '.pkl'
        j = 1
        while (os.path.isfile(singlet_fit.memory_file)):
            singlet_fit.memory_file = 'singlet_fit.' + data_date.strftime('%Y-%m-%d') + '.' + str(j) + '.pkl'
            j += 1
        print('    using new randomly initialized gmm, ' + singlet_fit.memory_file)
        if save_analysis_memory:
            # os.chdir(cell_mem_dir)
            with open(singlet_fit.memory_file, 'wb') as f:
                pickle.dump(singlet_fit, f)
                
    os.chdir(data_directory)
    
    if update_progress:
        print('                 ... done, ' + str(pd.Timestamp.now().round('s')))

    if update_progress:
        print('Plotting singlet_fit results, ' + str(pd.Timestamp.now().round('s')))
    plt.rcParams["figure.figsize"] = [8, 8]
    fig, axs = plt.subplots(1, 1)
    fig.suptitle('GMM fit results for singlet gating',
                 y=0.92, verticalalignment='bottom', size=16)
    num_points = 10000

    data = singlet_gmm_data4

    x = data['singlet_difference']  # [:num_points]
    y = data[f'log_{y_channel_h}']  # [:num_points])

    labels = singlet_fit.predict(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points]).copy()
    probs = singlet_fit.predict_proba(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points])
    size = 40 * probs.max(1) ** 2   # square emphasizes differences

    axs.set_xlim(-0.6, 0.9)
    axs.scatter(x[:num_points], y[:num_points], s=size, c=labels, cmap='viridis')
    axs.set_xlabel('log10(SSC-A/SSC-H)')
    axs.set_ylabel('log10(SSC-H)')
    w_factor = 0.2 / singlet_fit.weights_.max()
    for pos, covar, w in zip(singlet_fit.means_, singlet_fit.covariances_, singlet_fit.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs)
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs)

    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    if update_progress:
        print('Applying singlet gate, ' + str(pd.Timestamp.now().round('s')))
        
    # Select clusters with -singlet_mean_cutoff<mean(singlet_difference)-singlet_mean_center<singlet_mean_cutoff
    # and stdv(singlet_difference)<singlet_stdv_cutoff
    sing_1 = singlet_fit.means_[:, 0] - singlet_mean_center < singlet_mean_cutoff
    sing_2 = singlet_fit.means_[:, 0] - singlet_mean_center > -1*singlet_mean_cutoff
    sing_3 = np.sqrt(singlet_fit.covariances_[:, 0, 0])<singlet_stdv_cutoff
    singlets = np.where(sing_1 & sing_2 & sing_3)

    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('GMM fit results for singlet gating',
                 y=0.92, verticalalignment='bottom', size=16)
    num_points = 10000

    probs = singlet_fit.predict_proba(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points])
    size = 40 * probs.max(1) ** 2   # square emphasizes differences
    labels = singlet_fit.predict(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points]).copy()
    labels2 = singlet_fit.predict(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points]).copy()
    labels2[np.isin(labels2, singlets)] = -5

    for ax, c in zip(axs, [labels, labels2]):
        ax.set_xlabel('log10(SSC-A/SSC-H)')
        ax.set_ylabel('log10(SSC-H)')
        ax.set_xlim(-0.6, 0.9)
        ax.scatter(x[:num_points], y[:num_points], s=size, c=c, cmap='viridis')

    pdf.savefig()
    if not show_plots:
        plt.close(fig)

    pd.options.mode.use_inf_as_na = True
    for i, (data, file) in enumerate(zip(coli_data, coli_files)):
        frame = data.flow_frame
        meta = data.metadata
        gmm_data = frame[['singlet_difference', f'log_{y_channel_h}']].copy().fillna(0)

        meta._scatter_singlet_fit = singlet_fit

        probs = singlet_fit.predict_proba(gmm_data)
        labels = singlet_fit.predict(gmm_data)

        is_singlet_1 = np.isin(labels, singlets)
        is_singlet_2 = frame['is_cell']
        is_singlet_3 = np.isfinite(frame['singlet_difference'])
        is_singlet_4 = frame[y_channel_a] > 0

        frame['is_singlet'] = is_singlet_1 & is_singlet_2 & is_singlet_3 & is_singlet_4

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
            
    # then do all the same manipulations with the back_data
    data = back_data
    frame = data.flow_frame
    meta = data.metadata
    gmm_data = frame[['singlet_difference', f'log_{y_channel_h}']].copy().fillna(0)

    meta._scatter_singlet_fit = singlet_fit

    probs = singlet_fit.predict_proba(gmm_data)
    labels = singlet_fit.predict(gmm_data)

    is_singlet_1 = np.isin(labels, singlets)
    is_singlet_2 = frame['is_cell']
    is_singlet_3 = np.isfinite(frame['singlet_difference'])
    is_singlet_4 = frame[y_channel_a] > 0

    frame['is_singlet'] = is_singlet_1 & is_singlet_2 & is_singlet_3 & is_singlet_4

    pickle_file = back_file
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    # Done with back_data
    
    if update_progress:
        print('Plotting scatter plots showing singlets, ' + str(pd.Timestamp.now().round('s')))
    sns.set()
    plt.rcParams["figure.figsize"] = [12, 4*len(coli_data)]
    fig, axs = plt.subplots(len(coli_data), 2)
    fig.suptitle('Data for cell samples before (left) and after (right) automatic singlet gating',
                 y=0.903, verticalalignment='bottom', size=16)
    if axs.ndim == 1:
        axs = np.array([ axs ])
    
    for ax in axs.flatten():
        ax.set_xlabel('log10(SSC-A/SSC-H)')
        ax.set_ylabel('log10(SSC-H)')

    for i, coli in enumerate(coli_data):
        axs[i, 0].get_shared_x_axes().join(axs[i, 0], axs[i, 1])
        axs[i, 0].get_shared_y_axes().join(axs[i, 0], axs[i, 1])
        data = coli.flow_frame
        data = data.loc[data['is_cell']].copy()
        data2 = data.loc[data['is_singlet']].copy()

        x = data['singlet_difference']  # [:num_points]
        y = data[f'log_{y_channel_h}']  # [:num_points])
        x2 = data2['singlet_difference']  # [:num_points]
        y2 = data2[f'log_{y_channel_h}']  # [:num_points])
        
        
        probs = singlet_fit.predict_proba(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points])
        size = 20 * probs.max(1) ** 2   # square emphasizes differences
        labels = singlet_fit.predict(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points]).copy()
        labels2 = singlet_fit.predict(data.loc[:, ['singlet_difference', f'log_{y_channel_h}']][:num_points]).copy()
        labels2[np.isin(labels2, singlets)] = -5
        
        for ax in axs[i]:
            ax.set_xlim(-0.6, 0.9)
            ax.set_ylim(2.5, 5)
            ax.text(0.05, 0.95, sample_names[i],
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        #axs[i, 0].scatter(x2[:num_points], y2[:num_points], s=10, rasterized=True)
        axs[i, 1].hist2d(x2, y2, bins=100, norm=colors.LogNorm(), rasterized=True)
        #axs[i, 1].scatter(x[:num_points], y[:num_points], s=size*2, c=labels2, cmap='viridis', rasterized=True)
        axs[i, 0].hist2d(x, y, bins=100, norm=colors.LogNorm(), rasterized=True)

    pdf.savefig()
    if not show_plots:
        plt.close(fig)
    pdf.close()

    singlet_gated_data = [data.flow_frame.loc[(data.flow_frame['is_cell']) & (data.flow_frame['is_singlet'])] for data in coli_data]

    #for back_frac, singlet_frac, data, gated, singlet in zip(old_back_fractions, old_singlet_back_fractions, coli_data, gated_data, singlet_gated_data):
    for data, singlet, file in zip(coli_data, singlet_gated_data, coli_files):
        singlet_back_fraction = singlet['back_prob'].sum() / singlet['back_prob'].size
        #back_frac.append(gmm_back_fraction)
        #singlet_frac.append(singlet_back_fraction)
        
        meta = data.metadata
        meta._gmm_singlet_back_fraction = singlet_back_fraction

        pickle_file = file
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
    
    if update_progress:
        print('Done. ' + str(pd.Timestamp.now().round('s')))


def draw_ellipse(position, covariance, ax=None, **kwargs):
    # Code for this function copied from:
    #     https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    #     A copy of the license terms from that code is here:
    '''
    The MIT License (MIT)

    Copyright (c) 2016 Jacob VanderPlas
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    '''
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def batch_apply_bead_cal(data_directory,
                         bead_file=None,
                         samples=None,
                         fl_channel='BL1-A'):
    """
    This batch process method applies bead-based calibrations to flow cytometry data files.
    The input files for this method need to be pickled FCSDataFrame objects.
    The bead data must first be processed using fit_bead_data() method. 
    The method adds a column to each input sample file. 
        The new column contains the calibrated fluorescence values.
        The new column name will be f'{fl_channel}-MEF'
    
    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located
        
    bead_file : path, or path-like (e.g. str)
        pickled bead data file to use for calibration.
        If bead_file=None then the method uses the first file in an
        automatically detected list of bead files.

    samples : array of str, shape (n,)
        a list or array of samples to be calibrated
        The method looks for .fcs_pkl file names that contain the sub-string
        correspnding to each string in samples.
        If samples is None (default), the method automatically selects
        sample files, attempting to get the cell data files
        (i.e filenames that do not contain "blank" or "bead");
        
    fl_channel : string
        name of the fluorescence channel to apply calibration results to

    Returns
    -------
    None
    """
    
    os.chdir(data_directory)
    
    coli_files, blank_file_list, bead_file_list = auto_find_files(samples=samples)
    
    if bead_file is None:
        bead_file = bead_file_list[0]
    bead_data = pickle.load(open(bead_file, 'rb'))
    
    calibrated_name = fl_channel + '-MEF'
    popt = bead_data.metadata.bead_calibration_params[fl_channel]
    
    def cal_funct(x):
        return bead_data.metadata.bead_function(x, *popt)
    
    for file in (coli_files + blank_file_list + bead_file_list):
        print(file)
        data = pickle.load(open(file, 'rb'))
        data.flow_frame[calibrated_name] = cal_funct(data.flow_frame[fl_channel])
        try:
            data.metadata._bead_calibration_params[fl_channel] = popt
        except AttributeError:
            data.metadata._bead_calibration_params = {fl_channel:popt}
        
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    
        
def batch_stan_background_fit(data_directory,
                              back_file=None,
                              fl_channel='BL1-A-MEF',
                              update_progress=True,
                              show_plots=True,
                              fit_max=None,
                              fit_min=None,
                              hist_bins=100,
                              control=dict(adapt_delta=0.99),
                              use_singlets=True,
                              use_cells=True,
                              iter=2000):

    
    if update_progress:
        print('Start batch_stan_background_fit: ' + str(pd.Timestamp.now().round('s')))
        
    os.chdir(data_directory)
    
    coli_files, blank_file_list, bead_file_list = auto_find_files()
    
    if back_file is None:
        back_file = blank_file_list[0]
    back_data = pickle.load(open(back_file, 'rb'))
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff()
        
    pdf_file = 'Stan background fit.' + fl_channel + '.pdf'
    pdf = PdfPages(pdf_file)
    
    sns.set()
    
    frame = back_data.flow_frame
    xmin = frame[frame['is_cell']][fl_channel].min()
    xmax = frame[frame['is_cell']][fl_channel].max()
    bins = np.linspace(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin), hist_bins)
    fig_b1, axs_b1 = plt.subplots(1, 1)
    fig_b1.suptitle('Background signal histogram')
    fig_b1.set_size_inches([12, 4])
    alpha = 0.75
    axs_b1.hist(frame[fl_channel], density=False,
                bins=bins, alpha=alpha)
    axs_b1.hist(frame[frame['is_cell']][fl_channel], density=False,
                bins=bins, alpha=alpha)
    axs_b1.hist(frame[frame['is_singlet']][fl_channel], density=False,
                bins=bins, alpha=alpha)
    axs_b1.set_yscale('log')
    axs_b1.set_xlabel(fl_channel)
    axs_b1.set_ylabel('Count')
    pdf.savefig(fig_b1)
    if not show_plots:
        plt.close(fig_b1)
    
    if update_progress:
        print('    Fitting background data: ' + str(pd.Timestamp.now().round('s')))
        
    frame = back_data.flow_frame
    stan_back_signal = frame
    if use_singlets:
        stan_back_signal = stan_back_signal[stan_back_signal['is_singlet']][fl_channel]
    elif use_cells:
        stan_back_signal = stan_back_signal[stan_back_signal['is_cell']][fl_channel]
    else:
        stan_back_signal = stan_back_signal[fl_channel]
    back_mean = stan_back_signal.mean()
    back_mstd = stan_back_signal.std()
    stan_back_signal = stan_back_signal[stan_back_signal > back_mean - 5*back_mstd]
    stan_back_signal = stan_back_signal[stan_back_signal < back_mean + 5*back_mstd]
    if fit_max is not None:
        stan_back_signal = stan_back_signal[stan_back_signal < fit_max]
    if fit_min is not None:
        stan_back_signal = stan_back_signal[stan_back_signal > fit_min]
    stan_back_signal = stan_back_signal.values
    stan_back_data = dict(signal=stan_back_signal, N=len(stan_back_signal))

    sm_back = get_stan_model('fit exp modified normal.stan')
    stan_back_fit = sm_back.sampling(data=stan_back_data, iter=iter, chains=4, control=control)
    
    start_string = findcommonstart(coli_files + blank_file_list)
    back_sample = back_file[len(start_string):back_file.rfind('.')]
    
    pickle_stan_sampling(fit=stan_back_fit, model=sm_back, file=back_sample + '.' + fl_channel + '.back_fit.stan_samp_pkl')
        
    #post_pred_back = stan_back_fit.extract(permuted=True)['post_pred_signal']
    frame = back_data.flow_frame
    xmin = stan_back_signal.min()
    xmax = stan_back_signal.max()
    bins = np.linspace(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin), hist_bins)
    fig_b2, axs_b2 = plt.subplots(1, 1)
    fig_b2.suptitle('Background distribution fit')
    fig_b2.set_size_inches([12, 4])
    
    hist_data = stan_back_signal
    bin_values = axs_b2.hist(hist_data, density=True, bins=bins,
                             color=sns.color_palette()[2])[0]
    bin_values = bin_values[bin_values>0]
    
    mu = np.mean(stan_back_fit.extract(permuted=True)['mu'])
    sig = np.exp(np.mean(np.log(stan_back_fit.extract(permuted=True)['sigma'])))
    lamb = np.exp(np.mean(np.log(stan_back_fit.extract(permuted=True)['lamb'])))
    y_back = fit_dist.back_dist(bins, mu=mu, sig=sig, lamb=lamb)
    axs_b2.plot(bins, y_back, linewidth=3, color='k');
    #axs_b2.hist(post_pred_back, density=True, bins=bins, alpha=0.3)
    
    axs_b2.set_yscale('log')
    axs_b2.set_xlabel(fl_channel)
    axs_b2.set_ylabel('Count')
    axs_b2.set_ylim(0.5*bin_values.min(), 2*bin_values.max())
    
    pdf.savefig(fig_b2)
    if not show_plots:
        plt.close(fig_b2)
    
    back_samples = stan_back_fit.extract(permuted=True)
    pairplot_data = pd.DataFrame({ key: back_samples[key] for key in ['mu', 'sigma', 'beta'] })
    pair_plot = sns.pairplot(pairplot_data)
    fig_b3 = pair_plot.fig
    fig_b3.suptitle('Exponentially modified normal fit: posterior samples', y=1.05)
    pdf.savefig(fig_b3)
    if not show_plots:
        plt.close(fig_b3)
        
    if show_plots:
        plt.show(fig_b3)
    pdf.close()
    
    # Save background fluorescence value as part of the metadata in the blank file
    try:
        back_data.metadata._background_signal[fl_channel] = mu
    except AttributeError:
        back_data.metadata._background_signal = {fl_channel:mu}
    
    with open(back_file, 'wb') as f:
        pickle.dump(back_data, f)
    print(f'Mean background signal level: {mu}')
    
    if update_progress:
        print('                   ... done: ' + str(pd.Timestamp.now().round('s')))
        
    return stan_back_fit
        

def estimate_upper_truncated_normal(data, truncation):
    shifted_data = truncation - data
    mu = shifted_data.mean()
    sig = shifted_data.std()
    
    if (mu > 3.5*sig):
        return (truncation-mu), sig
    else:
        w = (sig/mu)**2
        p3 = 1 + 5.74050101*w - 13.53427037*(w**2) + 6.88665552*(w**3)
        p4 = -0.00374615 + 0.17462558*w - 2.87168509*(w**2) + 17.48932655*(w**3) - 11.91716546*(w**4)
        q = p4/p3
        mu_t = truncation - (mu*(1-q))
        sig_t = np.sqrt(sig**2 + q*(mu**2))
        return mu_t, sig_t


def fit_bead_data(data_directory,
                  bead_file=None,
                  num_bead_clusters=3,
                  bead_init=100,
                  num_singlet_clusters=4,
                  singlet_init=50,
                  num_bead_populations=6,
                  bead_population_init=400,
                  pop_init_means=None,
                  pop_init_cov=None,
                  auto_1D_fit=False,
                  num_1D_clusters=None,
                  max_cytometer_signal=1048575,
                  outlier_quantile = 0.03,
                  show_plots=True,
                  update_progress=True,
                  debug=False,
                  ssc_min=50000,
                  fsc_min=20000,
                  ssc_max=None,
                  fsc_max=None,
                  skip_low_beads=0,
                  lower_threshold=None,
                  upper_threshold=None,
                  fit_VL1=False,
                  manuscript_style=False,
                  sub_fig_lower=False,
                  FSC_channel='FSC-H',
                  SSC_channel='SSC-H',
                  SSCwidth_channel='SSC-W',
                  covariance=1000,
                  singlet_low=200,
                  singlet_high=600,
                  fluoro_channel_1='BL1-A',
                  fluoro_channel_2='YL1-A',
                  fluoro_channel_3='VL1-A'):

    """
    This method performs uses fluorescence data for calibration neads to 
    detirmine the calibration curves to convert from from cytometry channel 
    signal to molecules of equivalent fluorophore.
    The input for this method is a pickeld FCSDataFrame object with the bead 
    data.
    
    Parameters
    ----------
    data_directory : path, or path-like (e.g. str)
        indicates the directory where the data files are located
        
    bead_file : path, or path-like (e.g. str)
        pickled bead data file to be used for calibration.
        If bead_file=None then the method uses the first file in an
        automatically detected list of bead files.
        
    num_bead_clusters : int
        the number of clusters used in the initial GMM fit to find the beads 
        using the scatter plot (side-scatter vs. forward-scatter).
        
    bead_init : int
        the number of random initializations to run for the initial GMM fit to 
        find the beads.
    
    num_singlet_clusters : int
        the number of clusters used in the second GMM fit to find the singlet
        bead events using the side-scatter height vs. side-scatter width.
        
    singlet_init : int
        the number of random initializations to run for the second GMM fit to 
        find the singlet bead events.
        
    num_bead_populations : int
        the number of clusters used in the final GMM fit to find the different 
        fluorescent beads using the fluorescence data (YL1-A vs. BL1-A).
        
    bead_population_init : int
        the number of random initializations to run for the final GMM fit to 
        find the different fluorescent beads.
        
    pop_init_means : array-like, shape (num_bead_populations, 2), optional
        user-provided initial means for the final GMM fit. Values should be 
        in the form [[bl1_0, yl1_0], [bl1_1, yl1_1], ..., [bl1_n, yl1_n]], 
        where bl1_n and yl1_n are the approximate fluorescence channel signals 
        for the nth bead in the calibration set. 
        Defaults to None. If None, GMM means are initialized using the default 
        method for flowgatenist.GaussianMixture().
        
    pop_init_cov : array-like, shape (num_bead_populations, 2), optional
        user-provided initial covariances for the final GMM fit. Values should be 
        in the form [[var_bl1_0, var_yl1_0], [var_bl1_1, var_yl1_1], ..., 
        [var_bl1_n, var_yl1_n]], where var_bl1_n and var_yl1_n are the 
        approximate variances of the fluorescence channel signals for the nth 
        bead in the calibration set. 
        Must be set (i.e. not None) if pop_init_means is not None.
        
    auto_1D_fit : Boolean
        if True, the method uses a 1D GMM fit of the BL1-A signal for the BL1 
        calibration. This is used if, for some of the beads, the YL1-A signal is 
        off-scale but BL1-A signal is not. 
        
    num_1D_clusters : int
        Only used if auto_1D_fit == True.
        The number of bead clusters used for the 1D GMM fit. 
        num_1D_clusters shold be greater than num_bead_populations.
        
    max_cytometer_signal : float
        The maximum possible signal from the cytometer
        
    outlier_quantile : float, between 0 and 1
        The fraction of data to be ignored as outliers in the final GMM fit.
        
    show_plots : Boolean
        If True, the method dynamically shows plots.
        If False, it just saves the plots to a pdf file without showing them.
        
    update_progress : Boolean
        if True, the method prints status updates as it goes.
        
    debug : Boolean
        if True, the method returns the final GMM result and the data used for
        the final GMM.
        
    ssc_min : float
        minimum value for side-scatter data; used to ignore debris or 
        left-over cells
        
    fsc_min : float
        minimum value for forward-scatter data; used to ignore debris or 
        left-over cells
        
    ssc_max : float
        maximum value for side-scatter data; used to ignore debris or 
        left-over cells
        
    fsc_max : float
        maximum value for forward-scatter data; used to ignore debris or 
        left-over cells
        
    skip_low_beads : int
        the number of lower-intensity beads to leave out of the calibration
        data. Used in conjuntion with Lower_threshold to use a fit to only a
        selected range of the calibration beads.
        
    lower_threshold : array-like, shape (2,) or (3,), optional
        if not None, used to ignore fluorescence singnals below a minimum,
        should be in the form [bl1_min, yl1_min, {vl1_min}]
        
    upper_threshold : array-like, shape (2,) or (3,), optional
        if not None, used to ignore fluorescence singnals above a maximum,
        should be in the form [bl1_max, yl1_max, {vl1_max}]
    
    fit_VL1 : Boolean
        if True, the method also uses the VL1-A signal for the final GMM fit
        (i.e. it uses a 3D GMM fit).
        
    manuscript_style : Boolean
        if True, figures are formatted in the style used for manuscript 
        preparation and the method returns the figures for additional 
        manipulation.
        
    sub_fig_lower : Boolean
        if True, figure sub-panels have lower-case letter designations.
        only used if manuscript_style == True.
    
    FSC_channel : str
        used to identify the FSC height channel name 

    SSC_channel : str
        used to identify the SSC channel name

    SSCwidth_channel : str
        used to identify the SSC width channel name

    covariance : float
        the covariance used to set the singlet bead gate

    singlet_low : float
        lower bound on the SSC-W parameter of the singlet population
    
    singlet_high : float
        upper bound on the SSC-W parameter of the singlet population
    
    fluoro_channel_1 : str
        channel name for the first fluorescence parameter to calibrate. Corresponds to calibration_data_b
    
    fluoro_channel_2 : str
        channel name for the second fluorescence parameter to calibrate. Corresponds to calibration_data_y

    fluoro_channel_3 : str
        channel name for the (optional) third fluorescence parameter to calibrate. Corresponds to calibration_data_v

    Returns
    -------
    Default: the MEF values and cytometry BL1-A and YL1-A signal values for 
        each bead, the variances of the BL1-A and YL1-A signals for each bead,
        and the covariance array resulting from the final GMM fit.
    if manuscript_style is True: the 'Manual initializatiion of GMM fit' figure
        and the 'Final GMM fit results...' figure.
    if debug is True: the final GMM fit and the data used for that fit
    -------
    """
                  
    if manuscript_style:
        sns.set_style("white")
        sns.set_style("ticks", {'xtick.direction':'in', 'xtick.top':True, 'ytick.direction':'in', 'ytick.right':True, })
        #sns.set_style({"axes.labelsize": 20, "xtick.labelsize" : 16, "ytick.labelsize" : 16})

        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16

        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['legend.edgecolor'] = 'k'

    os.chdir(data_directory)
    
    # Turn interactive plotting on or off depending on show_plots
    if show_plots:
        plt.ion()
    else:
        plt.ioff() 
    
    if bead_file is None:
        bead_file = glob.glob('*bead*.fcs_pkl')[0]
        
    pdf_file = bead_file[:bead_file.rfind('.')] + '.plots.pdf'
    pdf = PdfPages(pdf_file)
        
    bead_data = pickle.load(open(bead_file, 'rb'))
    
    # First, use GMM model to find main bead population from scatter plot
    bead_gmm_data = bead_data.flow_frame.loc[:, [FSC_channel, SSC_channel]].copy()
    bead_gmm_data = bead_gmm_data[bead_gmm_data[FSC_channel] < bead_gmm_data[FSC_channel].max()]
    bead_gmm_data = bead_gmm_data[bead_gmm_data[SSC_channel] < bead_gmm_data[SSC_channel].max()]
    bead_gmm_data = bead_gmm_data[bead_gmm_data[FSC_channel] > bead_gmm_data[FSC_channel].min()]
    bead_gmm_data = bead_gmm_data[bead_gmm_data[SSC_channel] > bead_gmm_data[SSC_channel].min()]
    # throw out bead data that is below a cut-off value to avoid including bacteria or debris
    #     in the fit
    bead_gmm_data = bead_gmm_data[bead_gmm_data[SSC_channel] > ssc_min]
    if ssc_max is not None:
        bead_gmm_data = bead_gmm_data[bead_gmm_data[SSC_channel] < ssc_max]
    bead_gmm_data = bead_gmm_data[bead_gmm_data[FSC_channel] > fsc_min]
    if fsc_max is not None:
        bead_gmm_data = bead_gmm_data[bead_gmm_data[FSC_channel] < fsc_max]
    
    bead_gmm = nist_gmm.GaussianMixture(n_components=num_bead_clusters,
                                   covariance_type='full',
                                   n_init=bead_init)
    
    bead_gmm.fit(bead_gmm_data)
    bead_fit = bead_gmm
    
    main_cluster = bead_fit.weights_.argmax()
    
    if update_progress: print(f"Main bead cluster mean: {bead_fit.means_[main_cluster]}")
    
    bead_data.flow_frame['is_main_cluster'] = bead_fit.predict(bead_data.flow_frame.loc[:, [FSC_channel, SSC_channel]]) == main_cluster
    
    main_beads_frame = bead_data.flow_frame[bead_data.flow_frame['is_main_cluster']]
    
    #sns.set()
    #plt.style.use('classic')
    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)
    if not manuscript_style:
        fig.suptitle('Initial GMM fit results to find beads',
                     y=0.92, verticalalignment='bottom', size=16)
    
    x_max = 500000
    y_max = 2000000
    x_bins = np.linspace(0, x_max, 200)
    y_bins = np.linspace(0, y_max, 200)
    
    x = bead_gmm_data[FSC_channel]
    y = bead_gmm_data[SSC_channel]

    labels = bead_fit.predict(bead_gmm_data)
    labels[np.isin(labels, [main_cluster])] = -2
    probs = bead_fit.predict_proba(bead_gmm_data)
    size = 10 * probs.max(1) ** 2  # square emphasizes differences

    axs[0].hist2d(x, y, bins=[x_bins, y_bins], norm=colors.LogNorm(), rasterized=True);
    axs[1].scatter(x, y, c=labels, cmap='viridis', s=size*0.5, rasterized=True);
    axs[1].set_xlim(0, x_max)
    axs[1].set_ylim(0, y_max)
    for ax in axs:
        ax.set_xlabel(FSC_channel)
        ax.set_ylabel(SSC_channel)
    w_factor = 0.2 / bead_fit.weights_.max()
    for pos, covar, w in zip(bead_fit.means_, bead_fit.covariances_, bead_fit.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs[1])
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs[1])
    draw_ellipse(bead_fit.means_[main_cluster], bead_fit.covariances_[main_cluster],
                 facecolor='none', edgecolor='k', ax=axs[0])
        
    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)
    
    # Next, use a second GMM fit to find singlet beads cluster
    singlet_gmm_data = main_beads_frame.loc[:, [SSCwidth_channel, SSC_channel]].copy()
    
    singlet_gmm_data = singlet_gmm_data[singlet_gmm_data[SSCwidth_channel] < singlet_gmm_data[SSCwidth_channel].max()]
    singlet_gmm_data = singlet_gmm_data[singlet_gmm_data[SSCwidth_channel] > singlet_gmm_data[SSCwidth_channel].min()]
    
    singlet_gmm = nist_gmm.GaussianMixture(n_components=num_singlet_clusters, covariance_type='full', n_init=singlet_init)
    
    singlet_gmm.fit(singlet_gmm_data)
    singlet_fit = singlet_gmm
    
    singlet_fit_frame = pd.DataFrame( [m[0] for m in singlet_fit.means_ ], columns=['mean'])
    singlet_fit_frame['covariance'] = [ c[0,0] for c in singlet_fit.covariances_ ]
    singlet_fit_frame['weight'] = singlet_fit.weights_
    
    singlet_select_frame = singlet_fit_frame[singlet_fit_frame['covariance']<covariance]
    singlet_select_frame = singlet_select_frame[singlet_select_frame['mean']>singlet_low]
    singlet_select_frame = singlet_select_frame[singlet_select_frame['mean']<singlet_high]
    singlet_cluster = singlet_select_frame.index[0]
    
    bead_data.flow_frame['is_singlet_cluster'] = (bead_data.flow_frame['is_main_cluster']) & (singlet_fit.predict(bead_data.flow_frame.loc[:, [SSCwidth_channel, SSC_channel]]) == singlet_cluster)
    
    singlet_beads_frame = bead_data.flow_frame[bead_data.flow_frame['is_singlet_cluster']]
    
    #sns.set()
    plt.rcParams["figure.figsize"] = [12, 6]
    fig, axs = plt.subplots(1, 2)
    if not manuscript_style:
        fig.suptitle('Second GMM fit results to find singlet bead events',
                     y=0.92, verticalalignment='bottom', size=16)
        
    x = singlet_gmm_data[SSCwidth_channel]
    y = singlet_gmm_data[SSC_channel]
    x_bins = np.linspace(0, x.max(), 200)
    y_bins = np.linspace(y.min(), y.max(), 200)

    labels = singlet_fit.predict(singlet_gmm_data)
    labels[np.isin(labels, [singlet_cluster])] = -3
    probs = singlet_fit.predict_proba(singlet_gmm_data)
    size = 10 * probs.max(1) ** 2  # square emphasizes differences

    axs[0].hist2d(x, y, bins=[x_bins, y_bins], norm=colors.LogNorm(), rasterized=True);
    axs[1].scatter(x, y, c=labels, cmap='viridis', s=size*0.5, rasterized=True);
    for ax in axs:
        ax.set_xlabel(SSCwidth_channel)
        ax.set_ylabel(SSC_channel)
        
    w_factor = 0.2 / singlet_fit.weights_.max()
    for pos, covar, w in zip(singlet_fit.means_, singlet_fit.covariances_, singlet_fit.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, edgecolor='k', ax=axs[1])
        draw_ellipse(pos, covar, alpha=0.5, facecolor='none', edgecolor='k', ax=axs[1])
    draw_ellipse(singlet_fit.means_[singlet_cluster], singlet_fit.covariances_[singlet_cluster],
                 facecolor='none', edgecolor='k', ax=axs[0])
            
    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)
    
    # Now run GMM model to identify bead clusters in fluorescent channels
    if fit_VL1:
        gmm_data = singlet_beads_frame.loc[:, [fluoro_channel_1, fluoro_channel_2, fluoro_channel_3]].copy()
        gmm_data = gmm_data[gmm_data[fluoro_channel_3] < gmm_data[fluoro_channel_3].max()]
        if lower_threshold is not None:
            gmm_data = gmm_data[gmm_data[fluoro_channel_3] > lower_threshold[2]]
            gmm_data = gmm_data[gmm_data[fluoro_channel_3] < upper_threshold[2]]
    else:
        gmm_data = singlet_beads_frame.loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
    
    channel_max_b = gmm_data[fluoro_channel_1].max()
    channel_max_y = gmm_data[fluoro_channel_2].max()
    gmm_data = gmm_data[gmm_data[fluoro_channel_1] < max_cytometer_signal]
    gmm_data = gmm_data[gmm_data[fluoro_channel_2] < max_cytometer_signal]
    if channel_max_b < max_cytometer_signal:
        gmm_data = gmm_data[gmm_data[fluoro_channel_1] < channel_max_b]
    if channel_max_y < max_cytometer_signal:
        gmm_data = gmm_data[gmm_data[fluoro_channel_2] < channel_max_y]
    if lower_threshold is not None:
        gmm_data = gmm_data[gmm_data[fluoro_channel_1] > lower_threshold[0]]
        gmm_data = gmm_data[gmm_data[fluoro_channel_1] < upper_threshold[0]]
        gmm_data = gmm_data[gmm_data[fluoro_channel_2] > lower_threshold[1]]
        gmm_data = gmm_data[gmm_data[fluoro_channel_2] < upper_threshold[1]]
    
    # Default: random initialization of GMM when pop_init_means=None
    if pop_init_means is None:
        gmm = nist_gmm.GaussianMixture(n_components=num_bead_populations,
                                       covariance_type='diag',
                                       n_init=bead_population_init)
        
        gmm.fit(gmm_data)
        
        # Mark outlier data that will be ignored in calibration
        outlier_cut_frame = gmm_data.copy()
        outlier_cut_frame["score"] = gmm.score_samples(outlier_cut_frame)
        cutoff = outlier_cut_frame["score"].quantile(outlier_quantile)
        if fit_VL1:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
        else:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
        
        # Re-run GMM fit with outliers excluded
        gmm = nist_gmm.GaussianMixture(n_components=num_bead_populations,
                                       covariance_type='diag',
                                       n_init=10,
                                       means_init=gmm.means_,
                                       weights_init=gmm.weights_)
        
        gmm.fit(gmm_data_trimmed);
        
        if manuscript_style:
            ret_fig_1 = (None, None)
        
    # If pop_init_means is specified, use pop_init_means and pop_init_cov to initialize GMM
    else:
        fixed_means = pop_init_means
        fixed_covars = pop_init_cov
        #precisions_init = [1/cov for cov in pop_init_cov]
        
        # First fit iteration: keep means and covariances fixed, only allow weights to vary.
        #     The result is used in plots to compare initialization with actual data
        gmm_fixed = nist_gmm.GaussianMixture(n_components=num_bead_populations,
                                             covariance_type='diag',
                                             n_init=bead_population_init,
                                             fixed_means=fixed_means,
                                             fixed_covars=fixed_covars)
        
        gmm_fixed.fit(gmm_data)
        
        # Mark outlier data
        outlier_cut_frame = gmm_data.copy()
        outlier_cut_frame["score"] = gmm_fixed.score_samples(outlier_cut_frame)
        cutoff = outlier_cut_frame["score"].quantile(outlier_quantile)
        if fit_VL1:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
        else:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
        
        # Plot comparison between initialization and actual fit results
        if fit_VL1:
            plt.rcParams["figure.figsize"] = [18, 12]
            init_fig, init_axs = plt.subplots(2, 3)
            init_axs = init_axs.flatten()
        else:
            plt.rcParams["figure.figsize"] = [18, 6]
            init_fig, init_axs = plt.subplots(1, 3)
        if manuscript_style:
            ret_fig_1 = (init_fig, init_axs)
        else:
            init_fig.suptitle('Manual initializatiion of GMM fit',
                              y=0.9, verticalalignment='bottom', size=16)
        
        bead_intensities_b = np.sort(gmm_fixed.means_[:,0])
        bead_intensities_y = np.sort(gmm_fixed.means_[:,1])
        
        x = gmm_data[fluoro_channel_1]
        y = gmm_data[fluoro_channel_2]
        x_out = gmm_data_outliers[fluoro_channel_1]
        y_out = gmm_data_outliers[fluoro_channel_2]
        
        x_sd_max = np.sqrt(gmm_fixed.covariances_[:,0].max())
        y_sd_max = np.sqrt(gmm_fixed.covariances_[:,1].max())
        x_sd_min = np.sqrt(gmm_fixed.covariances_[:,0].min())
        y_sd_min = np.sqrt(gmm_fixed.covariances_[:,1].min())
        
        labels = gmm_fixed.predict(gmm_data)
        probs = gmm_fixed.predict_proba(gmm_data)
        size = 10 * probs.max(1) ** 2  # square emphasizes differences
        
        #init_axs[0].hist2d(x, y, bins=[x_bins2, y_bins2], norm=colors.LogNorm(), rasterized=True);
        for i, (ax, sub) in enumerate(zip(init_axs[:3], ['A', 'B', 'C'])):
        #for ax in init_axs[:3]:
            if manuscript_style:
                shift = 0.025
                t_x = -0.15
                t_y = 1.05
                box = ax.get_position()
                box.x0 = box.x0 + shift*i
                box.x1 = box.x1 + shift*i
                ax.set_position(box)
                if sub_fig_lower:
                    sub = sub.to_lower()
                ax.text(x=t_x, y=t_y, s=sub, horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=24, fontweight="bold")
            ax.set_xlabel(fluoro_channel_1)
            ax.set_ylabel(fluoro_channel_2)
            ax.scatter(x, y, c=labels, cmap='viridis', s=size, rasterized=True);
            ax.plot(x_out, y_out, 'x', c='red', fillstyle='none', rasterized=True);
            for m, cv in zip(fixed_means, fixed_covars):
                ax.plot([ m[0]-np.sqrt(cv[0]), m[0]+np.sqrt(cv[0]) ], [ m[1]-np.sqrt(cv[1]), m[1]+np.sqrt(cv[1]) ], color='orange')
                ax.plot([ m[0]-np.sqrt(cv[0]), m[0]+np.sqrt(cv[0]) ], [ m[1]+np.sqrt(cv[1]), m[1]-np.sqrt(cv[1]) ], color='orange')
        
        init_axs[0].set_xlim(-3*x_sd_min, bead_intensities_b[-4] + 3*x_sd_min)
        init_axs[0].set_ylim(-3*y_sd_min, bead_intensities_y[-4] + 3*y_sd_min)
        
        init_axs[1].set_xlim(-x_sd_max, bead_intensities_b[-3] + 3*x_sd_max)
        init_axs[1].set_ylim(-y_sd_max, bead_intensities_y[-3] + 3*y_sd_max)
        
        init_axs[2].set_xlim(-x_sd_max, bead_intensities_b[-1] + 5*x_sd_max)
        init_axs[2].set_ylim(-y_sd_max, bead_intensities_y[-1] + 5*y_sd_max)
        
        w_factor = 0.2 / gmm_fixed.weights_.max()
        for pos, covar, w in zip(gmm_fixed.means_, gmm_fixed.covariances_, gmm_fixed.weights_):
            pos = pos[:2]
            covar = covar[:2]
            draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[0])
            draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[0])
            draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[1])
            draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[1])
            draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[2])
            draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[2])
            
        if fit_VL1:
            bead_intensities_v = np.sort(gmm_fixed.means_[:,2])
            x_v = gmm_data[fluoro_channel_3]
            x_out_v = gmm_data_outliers[fluoro_channel_3]
            x_sd_max_v = np.sqrt(gmm_fixed.covariances_[:,2].max())
            x_sd_min_v = np.sqrt(gmm_fixed.covariances_[:,2].min())
            
            for ax in init_axs[3:]:
                ax.set_xlabel(fluoro_channel_3)
                ax.set_ylabel(fluoro_channel_2)
                ax.scatter(x_v, y, c=labels, cmap='viridis', s=size*0.5, rasterized=True);
                ax.scatter(x_out_v, y_out, c='red', rasterized=True);
                for m, cv in zip(fixed_means, fixed_covars):
                    ax.plot([ m[2]-np.sqrt(cv[2]), m[2]+np.sqrt(cv[2]) ], [ m[1]-np.sqrt(cv[1]), m[1]+np.sqrt(cv[1]) ], color='orange')
                    ax.plot([ m[2]-np.sqrt(cv[2]), m[2]+np.sqrt(cv[2]) ], [ m[1]+np.sqrt(cv[1]), m[1]-np.sqrt(cv[1]) ], color='orange')
            
            init_axs[3].set_xlim(-3*x_sd_min_v, bead_intensities_v[-4] + 3*x_sd_min_v)
            init_axs[3].set_ylim(-3*y_sd_min, bead_intensities_y[-4] + 3*y_sd_min)
            
            init_axs[4].set_xlim(-x_sd_max_v, bead_intensities_v[-3] + 3*x_sd_max_v)
            init_axs[4].set_ylim(-y_sd_max, bead_intensities_y[-3] + 3*y_sd_max)
            
            init_axs[5].set_xlim(-x_sd_max_v, bead_intensities_v[-1] + 5*x_sd_max_v)
            init_axs[5].set_ylim(-y_sd_max, bead_intensities_y[-1] + 5*y_sd_max)
            
            w_factor = 0.2 / gmm_fixed.weights_.max()
            for pos, covar, w in zip(gmm_fixed.means_, gmm_fixed.covariances_, gmm_fixed.weights_):
                pos = pos[-1:-3:-1]
                covar = covar[-1:-3:-1]
                draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[3])
                draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[3])
                draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[4])
                draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[4])
                draw_ellipse(pos, covar, alpha=0.24, edgecolor='k', ax=init_axs[5])
                draw_ellipse(pos, covar, alpha=0.24, facecolor='none', edgecolor='k', ax=init_axs[5])
        
        # Re-run fit, allowing all parameters to vary, initialized from previous result
        gmm = nist_gmm.GaussianMixture(n_components=num_bead_populations,
                                       covariance_type='diag',
                                       n_init=10,
                                       means_init=gmm_fixed.means_,
                                       weights_init=gmm_fixed.weights_)
        gmm.fit(gmm_data_trimmed);
        
        # Re-detirmine outlier data that will be ignored in calibration
        outlier_cut_frame = gmm_data.copy()
        outlier_cut_frame["score"] = gmm.score_samples(outlier_cut_frame)
        cutoff = outlier_cut_frame["score"].quantile(outlier_quantile)
        if fit_VL1:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2,fluoro_channel_3]].copy()
        else:
            gmm_data_trimmed = outlier_cut_frame[outlier_cut_frame["score"]>cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
            gmm_data_outliers = outlier_cut_frame[outlier_cut_frame["score"]<=cutoff].loc[:, [fluoro_channel_1, fluoro_channel_2]].copy()
        
        # Re-run GMM fit with new outliers excluded
        gmm = nist_gmm.GaussianMixture(n_components=num_bead_populations,
                                       covariance_type='diag',
                                       n_init=10,
                                       means_init=gmm.means_,
                                       weights_init=gmm.weights_)
        
        gmm.fit(gmm_data_trimmed);
    
    bead_population_fit = gmm
    
    bead_intensities_b = np.sort(bead_population_fit.means_[:,0])
    bead_intensities_y = np.sort(bead_population_fit.means_[:,1])
    
    var_pos_b = [list(bead_population_fit.means_[:,0]).index(inten) for inten in bead_intensities_b]
    var_pos_y = [list(bead_population_fit.means_[:,1]).index(inten) for inten in bead_intensities_y]
    #prec_b = bead_population_fit.precisions_[:,0][var_pos_b]
    fit_cov = bead_population_fit.covariances_
    var_b = fit_cov[:,0][var_pos_b]
    var_y = fit_cov[:,1][var_pos_y]
    bead_sigmas_b = np.sqrt(var_b)
    bead_sigmas_y = np.sqrt(var_y)
    if fit_VL1:
        bead_intensities_v = np.sort(bead_population_fit.means_[:,2])
        var_pos_v = [list(bead_population_fit.means_[:,2]).index(inten) for inten in bead_intensities_v]
        var_v = fit_cov[:,2][var_pos_v]
        bead_sigmas_v = np.sqrt(var_v)
    
    for i, (intensity, sigma, var, pos) in enumerate(zip(bead_intensities_b, bead_sigmas_b, var_b, var_pos_b)):
        if ((intensity + 3.5*sigma) > max_cytometer_signal):
            labels = bead_population_fit.predict(gmm_data)
            trunc_gmm_data = gmm_data[labels==pos]
            intensity, sigma = estimate_upper_truncated_normal(data=trunc_gmm_data[fluoro_channel_1], truncation=max_cytometer_signal)
            bead_intensities_b[i] = intensity
            bead_sigmas_b[i] = sigma
            var_b[i] = sigma**2
    
    for i, (intensity, sigma, var, pos) in enumerate(zip(bead_intensities_y, bead_sigmas_y, var_y, var_pos_y)):
        if ((intensity + 3.5*sigma) > max_cytometer_signal):
            labels = bead_population_fit.predict(gmm_data)
            trunc_gmm_data = gmm_data[labels==pos]
            intensity, sigma = estimate_upper_truncated_normal(data=trunc_gmm_data[fluoro_channel_2], truncation=max_cytometer_signal)
            bead_intensities_y[i] = intensity
            bead_sigmas_y[i] = sigma
            var_y[i] = sigma**2
            
    if fit_VL1:
        for i, (intensity, sigma, var, pos) in enumerate(zip(bead_intensities_v, bead_sigmas_v, var_v, var_pos_v)):
            if ((intensity + 3.5*sigma) > max_cytometer_signal):
                labels = bead_population_fit.predict(gmm_data)
                trunc_gmm_data_v = gmm_data[labels==pos]
                intensity, sigma = estimate_upper_truncated_normal(data=trunc_gmm_data_v[fluoro_channel_3], truncation=max_cytometer_signal)
                bead_intensities_v[i] = intensity
                bead_sigmas_v[i] = sigma
                var_v[i] = sigma**2
    
    #sns.set()

    plt.rcParams["figure.figsize"] = [18, 18]
    fig, axs = plt.subplots(3, 3)
    if not manuscript_style:
        fig.suptitle('Final GMM fit results with fluorescence signals to find different beads',
                     y=0.89, verticalalignment='bottom', size=16)
    if manuscript_style:
        ret_fig_2 = (fig, axs)
    axs_2d = axs[0]
    axs_1d = axs[1:]
        
    x = gmm_data_trimmed[fluoro_channel_1]
    y = gmm_data_trimmed[fluoro_channel_2]
    x_out = gmm_data_outliers[fluoro_channel_1]
    y_out = gmm_data_outliers[fluoro_channel_2]
        
    x_sd_max = np.sqrt(bead_population_fit.covariances_[:,0].max())
    y_sd_max = np.sqrt(bead_population_fit.covariances_[:,1].max())

    #x_bins2 = np.linspace(x.min(), x.max(), 200)
    #y_bins2 = np.linspace(y.min(), y.max(), 200)

    labels = bead_population_fit.predict(gmm_data_trimmed)
    probs = bead_population_fit.predict_proba(gmm_data_trimmed)
    size = 10 * probs.max(1) ** 2  # square emphasizes differences
    
    if not manuscript_style:
        for ax in axs_1d.flatten():
            ax.set_yscale("log")

    if num_bead_populations>4:
        axs_2d[0].set_xlim(-x_sd_max, bead_intensities_b[-5] + 3*x_sd_max)
        axs_2d[0].set_ylim(-y_sd_max, bead_intensities_y[-5] + 3*y_sd_max)
        axs_1d[0][0].set_xlim(-x_sd_max, bead_intensities_b[-5] + 3*x_sd_max)
        axs_1d[1][0].set_xlim(-y_sd_max, bead_intensities_y[-5] + 3*y_sd_max)
    else:
        axs_2d[0].set_xlim(-x_sd_max, bead_intensities_b[0] + 3*x_sd_max)
        axs_2d[0].set_ylim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)
        axs_1d[0][0].set_xlim(-x_sd_max, bead_intensities_b[0] + 3*x_sd_max)
        axs_1d[1][0].set_xlim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)

    if num_bead_populations>2:
        axs_2d[1].set_xlim(-x_sd_max, bead_intensities_b[-3] + 3*x_sd_max)
        axs_2d[1].set_ylim(-y_sd_max, bead_intensities_y[-3] + 3*y_sd_max)
        axs_1d[0][1].set_xlim(-x_sd_max, bead_intensities_b[-3] + 3*x_sd_max)
        axs_1d[1][1].set_xlim(-y_sd_max, bead_intensities_y[-3] + 3*y_sd_max)
    else:
        axs_2d[1].set_xlim(-x_sd_max, bead_intensities_b[0] + 3*x_sd_max)
        axs_2d[1].set_ylim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)
        axs_1d[0][1].set_xlim(-x_sd_max, bead_intensities_b[0] + 3*x_sd_max)
        axs_1d[1][1].set_xlim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)
        
    axs_2d[2].set_xlim(-x_sd_max, bead_intensities_b[-1] + 5*x_sd_max)
    axs_2d[2].set_ylim(-y_sd_max, bead_intensities_y[-1] + 5*y_sd_max)
    axs_1d[0][2].set_xlim(-x_sd_max, bead_intensities_b[-1] + 5*x_sd_max)
    axs_1d[1][2].set_xlim(-y_sd_max, bead_intensities_y[-1] + 5*y_sd_max)

    for ax, ax_1d in zip(axs_2d, np.transpose(axs_1d)):
        ax.scatter(x, y, c=labels, cmap='viridis', s=size, rasterized=True);
        ax.plot(x_out, y_out, 'x', c='red', fillstyle='none', rasterized=True);
        ax.set_xlabel(fluoro_channel_1)
        ax.set_ylabel(fluoro_channel_2)
        xlim = ax_1d[0].get_xlim()
        bins = np.linspace(xlim[0], xlim[1], 70)
        ax_1d[0].hist(x, bins=bins, color='royalblue', zorder=-1)
        ax_1d[0].set_xlabel(fluoro_channel_1)
        ax_1d[0].set_ylabel('Count')
        xlim = ax_1d[1].get_xlim()
        bins = np.linspace(xlim[0], xlim[1], 70)
        ax_1d[1].hist(y, bins=bins, color='goldenrod', zorder=-1)
        ax_1d[1].set_xlabel(fluoro_channel_2)
        ax_1d[1].set_ylabel('Count')
        
        
    if manuscript_style:
        shift = 0.025
        t_x = -0.15
        t_y = 1.05
        for j, (ax_row, sub_list) in enumerate(zip(axs, [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']])):
            for i, (ax, sub) in enumerate(zip(ax_row, sub_list)):
                box = ax.get_position()
                box.x0 = box.x0 + shift*i
                box.x1 = box.x1 + shift*i
                box.y0 = box.y0 - shift*j
                box.y1 = box.y1 - shift*j
                ax.set_position(box)
                if sub_fig_lower:
                    sub = sub.to_lower()
                ax.text(x=t_x, y=t_y, s=sub, horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=24, fontweight="bold")
        ret_fig_2 = (fig, axs)

    w_factor = 0.2 / bead_population_fit.weights_.max()
    for pos, covar, w in zip(bead_population_fit.means_, bead_population_fit.covariances_, bead_population_fit.weights_):
        pos = pos[:2]
        covar = covar[:2]
        draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[0])
        draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[0])
        draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[1])
        draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[1])
        draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[2])
        draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[2])
        
    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)
    
    if fit_VL1:
        x_v = gmm_data_trimmed[fluoro_channel_3]
        x_out_v = gmm_data_outliers[fluoro_channel_3]
        x_sd_max_v = np.sqrt(bead_population_fit.covariances_[:,2].max())
        
        #sns.set()
        plt.rcParams["figure.figsize"] = [18, 12]
        fig, axs = plt.subplots(2, 3)
        if not manuscript_style:
            fig.suptitle('Final GMM fit results with Vl1-A signal',
                         y=0.91, verticalalignment='bottom', size=16)
        axs = axs.flatten()
        axs_2d = axs[:3]
        
        for ax in axs[3:]:
            ax.set_yscale("log")
    
        if num_bead_populations>4:
            axs_2d[0].set_xlim(-x_sd_max_v, bead_intensities_v[-5] + 3*x_sd_max_v)
            axs_2d[0].set_ylim(-y_sd_max, bead_intensities_y[-5] + 3*y_sd_max)
            axs[3].set_xlim(-x_sd_max_v, bead_intensities_v[-5] + 3*x_sd_max_v)
        else:
            axs[3].set_xlim(-x_sd_max_v, bead_intensities_v[0] + 3*x_sd_max_v)
            axs_2d[0].set_xlim(-x_sd_max_v, bead_intensities_v[0] + 3*x_sd_max_v)
            axs_2d[0].set_ylim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)
    
        if num_bead_populations>2:
            axs_2d[1].set_xlim(-x_sd_max_v, bead_intensities_v[-3] + 3*x_sd_max_v)
            axs_2d[1].set_ylim(-y_sd_max, bead_intensities_y[-3] + 3*y_sd_max)
            axs[4].set_xlim(-x_sd_max_v, bead_intensities_v[-3] + 3*x_sd_max_v)
        else:
            axs_2d[1].set_xlim(-x_sd_max_v, bead_intensities_v[0] + 3*x_sd_max_v)
            axs_2d[1].set_ylim(-y_sd_max, bead_intensities_y[0] + 3*y_sd_max)
            axs[4].set_xlim(-x_sd_max_v, bead_intensities_v[0] + 3*x_sd_max_v)
            
        axs_2d[2].set_xlim(-x_sd_max_v, bead_intensities_v[-1] + 5*x_sd_max_v)
        axs_2d[2].set_ylim(-y_sd_max, bead_intensities_y[-1] + 5*y_sd_max)
        axs[5].set_xlim(-x_sd_max_v, bead_intensities_v[-1] + 5*x_sd_max_v)
        
        for ax in axs[3:]:
            xlim = ax.get_xlim()
            bins = np.linspace(xlim[0], xlim[1], 70)
            ax.hist(x_v, bins=bins, color='blueviolet')
            ax.set_xlabel(fluoro_channel_3)
            ax.set_ylabel('Count')
            
        for ax in axs_2d:
            ax.scatter(x_v, y, c=labels, cmap='viridis', s=size*0.5, rasterized=True);
            ax.scatter(x_out_v, y_out, c='red', rasterized=True);
            ax.set_xlabel(fluoro_channel_3)
            ax.set_ylabel(fluoro_channel_2)
            
        for pos, covar, w in zip(bead_population_fit.means_, bead_population_fit.covariances_, bead_population_fit.weights_):
            pos = pos[-1:-3:-1]
            covar = covar[-1:-3:-1]
            draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[0])
            draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[0])
            draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[1])
            draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[1])
            draw_ellipse(pos, covar, alpha=0.4, edgecolor='k', ax=axs_2d[2])
            draw_ellipse(pos, covar, alpha=0.4, facecolor='none', edgecolor='k', ax=axs_2d[2])
        
        pdf.savefig(fig)
        if not show_plots:
            plt.close(fig)
    
    calibration_data_b = np.array(bead_calibration_frame.calibration_data_b)[skip_low_beads:]
    calibration_data_y = np.array(bead_calibration_frame.calibration_data_y)[skip_low_beads:]
    calibration_data_v = np.array(bead_calibration_frame.calibration_data_v)[skip_low_beads:]
    
    popt_b, pcov_b = curve_fit(bead_func1, bead_intensities_b, calibration_data_b[:len(bead_intensities_b)], sigma=bead_sigmas_b)
    popt_y, pcov_y = curve_fit(bead_func1, bead_intensities_y, calibration_data_y[:len(bead_intensities_y)], sigma=bead_sigmas_y)
    if fit_VL1:
        popt_v, pcov_v = curve_fit(bead_func1, bead_intensities_v, calibration_data_v[:len(bead_intensities_v)], sigma=bead_sigmas_v)

    if num_1D_clusters is None:
        num_1D_clusters = len(calibration_data_b[inv_bead_func1(calibration_data_b, *popt_b) + 4*bead_sigmas_b[-1]<channel_max_b])
    
    if (num_bead_populations<num_1D_clusters and auto_1D_fit):
        if update_progress: print('1D fit:')
        gmm_data_1D = singlet_beads_frame[fluoro_channel_1].copy()        
        gmm_data_1D = gmm_data_1D[gmm_data_1D < channel_max_b]
        gmm_data_1D = np.array(gmm_data_1D)
        
        gmm_data_1D = gmm_data_1D.reshape(-1, 1)
        
        gmm_init_1D_means = inv_bead_func1(calibration_data_b, *popt_b)[:num_1D_clusters]
        gmm_init_1D_means = gmm_init_1D_means.reshape(-1, 1)
        
        gmm_init_1D_precisions = np.append(1/var_b, [1/var_b[-1] for i in range(num_1D_clusters - num_bead_populations)])
        gmm_init_1D_precisions = gmm_init_1D_precisions.reshape(-1, 1, 1)
        
        gmm_init_1D_weights = [1/num_1D_clusters for i in range(num_1D_clusters)]
        
        gmm_1D = nist_gmm.GaussianMixture(n_components=num_1D_clusters,
                                          covariance_type='full', n_init=1,
                                          weights_init=gmm_init_1D_weights,
                                          means_init=gmm_init_1D_means,
                                          precisions_init=gmm_init_1D_precisions)
        
        gmm_1D.fit(gmm_data_1D)
        
        bead_intensities_b = np.sort(gmm_1D.means_.flatten())
        
        var_pos_b = [list(gmm_1D.means_.flatten()).index(inten) for inten in bead_intensities_b]
        var_b = gmm_1D.covariances_.flatten()[var_pos_b]
        bead_sigmas_b = np.sqrt(var_b)
        
        for i, (intensity, sigma, var, pos) in enumerate(zip(bead_intensities_b, bead_sigmas_b, var_b, var_pos_b)):
            if ((intensity + 3.5*sigma) > max_cytometer_signal):
                if update_progress:
                    print('using estimates for truncated Gaussian...')
                labels = gmm_1D.predict(gmm_data_1D)
                trunc_gmm_data = gmm_data_1D[labels==pos]
                intensity, sigma = estimate_upper_truncated_normal(data=trunc_gmm_data, truncation=max_cytometer_signal)
                bead_intensities_b[i] = intensity
                bead_sigmas_b[i] = sigma
                var_b[i] = sigma**2
        
        
        popt_b, pcov_b = curve_fit(bead_func1,
                                   bead_intensities_b,
                                   calibration_data_b[:len(bead_intensities_b)],
                                   sigma=bead_sigmas_b)
        
        plt.rcParams["figure.figsize"] = [18, 6]
        fig, axs = plt.subplots(1, 1)
        if not manuscript_style:
            fig.suptitle('Final 1D GMM fit results',
                         y=0.92, verticalalignment='bottom', size=16)
        
        bin_values = axs.hist(gmm_data_1D, bins=200)[0];
        cmap = color_maps.get_cmap('jet')
        y_max = bin_values.max()
        for i, (m, s) in enumerate(zip(bead_intensities_b, bead_sigmas_b)):
            c_i = cmap(i/len(bead_intensities_b))
            axs.plot([m, m], [0, y_max], color = c_i);
            axs.plot([m + s, m + s], [0, y_max], color = c_i, linewidth=0.5);
            axs.plot([m - s, m - s], [0, y_max], color = c_i, linewidth=0.5);
            
            #c=labels, cmap='viridis'
            
            
        x_min = bead_intensities_b[0] - 5*bead_sigmas_b[0]
        x_max = bead_intensities_b[-1] + 5*bead_sigmas_b[-1]
        axs.set_xlim(x_min, x_max)
            
        pdf.savefig(fig)
        if not show_plots:
            plt.close(fig)
    
    #sns.set()
    
    plt.rcParams["figure.figsize"] = [18, 12]
    if fit_VL1:
        fig, axs = plt.subplots(3, 3)
    else:
        fig, axs = plt.subplots(2, 3)
        
    
    if not manuscript_style:
        fig.suptitle('Bead calibration curves',
                     y=0.9, verticalalignment='bottom', size=16)
        
    for ax in axs.flatten():
        ax.set_xlabel('Measured Bead Brightness')
        ax.set_ylabel('Bead Calibration Value')
    
    if num_bead_populations>2:
        x_plot_data_b = np.linspace(0, 1.2*bead_intensities_b[2], 20)
        x_plot_data_y = np.linspace(0, 1.2*bead_intensities_y[2], 20)
            
        x_b = bead_intensities_b[:3]
        y_b = calibration_data_b[:3]
        x_y = bead_intensities_y[:3]
        y_y = calibration_data_y[:3]
        
        if fit_VL1:
            x_plot_data_v = np.linspace(0, 1.2*bead_intensities_v[2], 20)
            x_v = bead_intensities_v[:3]
            y_v = calibration_data_v[:3]
    else:
        x_plot_data_b = np.linspace(0, 1.2*bead_intensities_b[-1], 20)
        x_plot_data_y = np.linspace(0, 1.2*bead_intensities_y[-1], 20)
            
        x_b = bead_intensities_b
        y_b = calibration_data_b[:len(x_b)]
        x_y = bead_intensities_y
        y_y = calibration_data_y[:len(x_y)]
        
        if fit_VL1:
            x_v = bead_intensities_v
            y_v = calibration_data_v[:len(x_v)]
            x_plot_data_v = np.linspace(0, 1.2*bead_intensities_v[-1], 20)
    
    log_x_plot_data_b = np.linspace(3, 6, 100)
    log_x_plot_data_y = np.linspace(3, 6, 100)
                
    log_x_b = np.log10(bead_intensities_b[1:])
    log_y_b = np.log10(calibration_data_b[1:len(bead_intensities_b)])
    log_x_y = np.log10(bead_intensities_y[1:])
    log_y_y = np.log10(calibration_data_y[1:len(bead_intensities_y)])
    
    if fit_VL1:
        log_x_plot_data_v = np.linspace(3, 6, 100)
        log_x_v = np.log10(bead_intensities_v[1:])
        log_y_v = np.log10(calibration_data_v[1:len(bead_intensities_v)])
    
    for j in range(3):
        axs[0,j].text(0.5, 0.9, fluoro_channel_1, horizontalalignment='center',
                      verticalalignment='center',
                      transform=axs[0,j].transAxes, size=16)
        axs[1,j].text(0.5, 0.9, fluoro_channel_2, horizontalalignment='center',
                      verticalalignment='center',
                      transform=axs[1,j].transAxes, size=16)
        if fit_VL1:
            axs[2,j].text(0.5, 0.9, fluoro_channel_3, horizontalalignment='center', verticalalignment='center', transform=axs[2,j].transAxes)
        
    axs[0,0].scatter(x_b, y_b, s=90);
    axs[0,0].plot(x_plot_data_b, bead_func1(x_plot_data_b, *popt_b), 'g-')

    axs[0,1].scatter(log_x_b, log_y_b, s=90);
    axs[0,1].plot(log_x_plot_data_b, bead_func2(log_x_plot_data_b, *popt_b), 'g-')
    axs[0,2].scatter(log_x_b, log_y_b - bead_func2(log_x_b, *popt_b), s=90, c='g');
    
    axs[1,0].scatter(x_y, y_y, s=90);
    axs[1,0].plot(x_plot_data_y, bead_func1(x_plot_data_y, *popt_y), 'r-')

    axs[1,1].scatter(log_x_y, log_y_y, s=90);
    axs[1,1].plot(log_x_plot_data_y, bead_func2(log_x_plot_data_y, *popt_y), 'r-')
    axs[1,2].scatter(log_x_y, log_y_y - bead_func2(log_x_y, *popt_y), s=90, c='r');
    
    if fit_VL1:
        axs[2,0].scatter(x_v, y_v, s=90);
        axs[2,0].plot(x_plot_data_v, bead_func1(x_plot_data_v, *popt_v), 'm-')
        axs[2,1].scatter(log_x_v, log_y_v, s=90);
        axs[2,1].plot(log_x_plot_data_v, bead_func2(log_x_plot_data_v, *popt_v), 'm-')
        axs[2,2].scatter(log_x_v, log_y_v - bead_func2(log_x_v, *popt_v), s=90, c='m');
    pdf.savefig(fig)
    if not show_plots:
        plt.close(fig)
    
    pdf.close()
    
    try:
        bead_data.metadata._bead_calibration_params[fluoro_channel_1] = popt_b
    except AttributeError:
        bead_data.metadata._bead_calibration_params = {fluoro_channel_1:popt_b}
    
    try:
        bead_data.metadata._bead_calibration_params[fluoro_channel_2] = popt_y
    except AttributeError:
        bead_data.metadata._bead_calibration_params = {fluoro_channel_2:popt_y}
    
    if fit_VL1:
        try:
            bead_data.metadata._bead_calibration_params[fluoro_channel_3] = popt_v
        except AttributeError:
            bead_data.metadata._bead_calibration_params = {fluoro_channel_3:popt_v}
    
    with open(bead_file, 'wb') as f:
        pickle.dump(bead_data, f)
        
    if manuscript_style:
        return [ret_fig_1, ret_fig_2]
    elif debug:
        return ( gmm, gmm_data )
    else:
        return (bead_func1(bead_intensities_b, *popt_b),
                bead_func1(bead_intensities_y, *popt_y),
                bead_intensities_b, bead_intensities_y,
                var_b, var_y, fit_cov)
    

def bead_func1(x, a, b):
    return a + b * x
    

def inv_bead_func1(y, a, b):
    return (y - a)/b


def bead_func2(x, a, b):
    return np.log10(a + b * (10**x))
    

def findcommonstart(strlist):
    return strlist[0][:([min([x[0]==elem for elem in x]) for x in zip(*strlist)]+[0]).index(0)]

    
def find_sample_names(coli_files):
    start_string = findcommonstart(coli_files)
    sample_names = [s[len(start_string):s.rfind('.')] if len(s)>17 else s[:s.rfind('.')] for s in coli_files]
    #remove_chars = ' _.'
    #fixed_names = []
    #for name in sample_names:
    #    while name[0] in remove_chars:
    #        name = name[1:]
    #    fixed_names.append(name)
        
    return sample_names, start_string

    
    