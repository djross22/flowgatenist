
import cmdstanpy
import pickle
import numpy as np
import pandas as pd
import os


def check_all_diagnostics(fit):
    print(fit.diagnose())


def compile_model(filename, model_name=None, force_recompile=False, verbose=True):

    return_directory = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Stan models'))
    
    sm = cmdstanpy.CmdStanModel(stan_file=filename)
    
    os.chdir(return_directory)

    return sm


def rhat_from_dataframe(df, split_chains=True):
    df = df.copy()
    if split_chains and ('draw' in list(df.columns)):
        chains = np.unique(df.chain)
        add_chain = max(chains) + 1
        cut_draw = (df.draw.max()+1)/2
        new_chain = []
        new_draw = []
        for c, d in zip(df.chain, df.draw):
            if d >= cut_draw:
                new_chain.append(c + add_chain)
                new_draw.append(d - cut_draw)
            else:
                new_chain.append(c)
                new_draw.append(d)
        df['chain'] = new_chain
        df['draw'] = new_draw
        
    chains = np.unique(df.chain)
    num_chains = len(chains)
    num_samples = len(df)/num_chains
    
    columns = list(df.columns)
    ignore_columns = ['chain', 'draw', 'warmup'] + [x for x in columns if x[-2:]=='__']
    columns = [x for x in columns if x not in ignore_columns]
    
    chain_mean = [df[df.chain==n][columns].mean() for n in chains]
    
    chain_mean = pd.concat(chain_mean, axis=1)
    
    grand_mean = chain_mean.mean(axis=1)
    
    x3 = chain_mean.sub(grand_mean, axis=0)**2
    between_chains_var = num_samples/(num_chains-1)*x3.sum(axis=1)
    
    within_chains_var = []
    for n in chains:
        d2 = df[df.chain==n]
        d2 = d2[columns]
        x = ((d2 - chain_mean[n])**2).sum()
        within_chains_var.append(1/(num_samples-1)*x)
    
    within_chains_var = pd.concat(within_chains_var, axis=1).mean(axis=1)
    
    rh1 = ((num_samples-1)/num_samples)*within_chains_var
    rh2 = between_chains_var/num_samples
    
    return np.sqrt((rh1 + rh2)/within_chains_var)