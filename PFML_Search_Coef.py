# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:04:08 2025

@author: Patrick
"""

import pandas as pd
import numpy as np
from functools import reduce
import pickle
from tqdm import tqdm

"""
Compute beta as in (26) at the end of each year starting from 1971. 

For each beta_t in (26), ALL data until the first date of the sample is used.

The code outputs coef_dict which contains the computation of beta_t for each
end of the year for every hyperparameter combination
"""

#%% Libraries

#Run specific functions
from General_functions import *

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout
settings, pf_set = get_settings()

#%% Functions

def denom_sum_fun(train):
    """
    Loops over all values in the dictionary train, extracting the 'denom' DataFrame from each.
    Uses reduce() to sum all these DataFrames element-wise.
    
    Returns the result: a single DataFrame representing the summed 'denom' matrices.
    """
    denom_list = [entry['denom'] for entry in train.values()]
    denom_sum = reduce(lambda x, y: x + y, denom_list)
    return denom_sum

#%% Overall inputs

#Years for which the optimisation (26) is computed for
hp_years = np.arange(settings['pf']['dates']['start_year'], settings['pf']['dates']['end_yr']+1)

# Hyperparameters
g_vec = settings['pf_ml']['g_vec']
p_vec = settings['pf_ml']['p_vec']
l_vec = settings['pf_ml']['l_vec']

#%% Compute beta Coefficients in (26)

for g_index, g in enumerate(g_vec):
    #Initialise dictionary for output
    coef_dict = {}
    
    # Read in Data
    with open(path + f'/Data/pfml_input_{g_index}.pkl', "rb") as f:
        pfml_input = pickle.load(f)[g_index]

    d_all = pd.Series(sorted(list(pfml_input['reals'].keys()))) 
    end_bef = min(hp_years)-1
    
    #Get the burn-in period
    mask = d_all < pd.to_datetime(f"{end_bef-1}-12-31")
    train_bef = {k: pfml_input['reals'][k] for k in list(d_all[mask])}
    
    ### --- Compute r_tilde_sum, i.e. the sum of all r_tilde for all dates of the
    #       burn-in period
    r_tilde_series_list = [inner_dict['r_tilde'] for inner_dict in train_bef.values()]
    # Concatenate the Series into a DataFrame. Each Series becomes a column.
    r_tilde_df = pd.concat(r_tilde_series_list, axis=1)
    
    # Calculate the sum across rows (axis=1 mimics R's rowSums)
    r_tilde_sum = r_tilde_df.sum(axis=1)
    
    #Sum denom for all dates of the burn-in period
    denom_raw_sum = denom_sum_fun(train_bef)
    
    #Number of Dates in burn-in period (used for computed the average utility)
    n = len(train_bef)
    
    #Save Workspace
    del train_bef
    
    # Make sure that hp_years is sorted
    hp_years = sorted(hp_years)
    
    """
    The following computes beta_t as in (26) for all hyperparameter combinations.
    It uses an expanding window such that it updates the sum in (25) with each new
    year.
    """
    
    for i, year in tqdm(enumerate(hp_years), desc="Computing Beta for all Hyperparameter Combinations"):
        
        # Get the next 12 months
        start_date = pd.to_datetime(f"{year - 2}-12-31")
        end_date = pd.to_datetime(f"{year - 1}-11-30")
        
        # Filter training data
        train_new = dict([(k, v) for k, v in pfml_input['reals'].items() if start_date <= k <= end_date])
        
        # Update n ( T in (25) )
        n += len(train_new)
        
        # Update r_tilde_sum (running sum)
        r_tilde_series_list = [inner_dict['r_tilde'] for inner_dict in train_new.values()]
        r_tilde_df = pd.concat(r_tilde_series_list, axis=1)
        r_tilde_sum += r_tilde_df.sum(axis=1)
        
        # Update denom
        denom_raw_new = denom_sum_fun(train_new)
        denom_raw_sum += denom_raw_new
    
        # Compute beta coefficients
        coef_by_hp = {}
        for p in p_vec:
            feat_p = pfml_feat_fun(p=p)
            r_tilde_sub = r_tilde_sum.loc[feat_p] / n
            denom_sub = denom_raw_sum.loc[feat_p,feat_p] / n
            
            results = {}
            for j, l in enumerate(l_vec):
                reg_matrix = denom_sub.to_numpy() + l * np.identity(p + 1)
                results[j] = pd.Series(np.linalg.solve(reg_matrix, r_tilde_sub.to_numpy()),index = feat_p)
            
            coef_by_hp[p] = results
        
        coef_dict[year] = coef_by_hp

    
    #Save Results
    coef_dict = {g_index: coef_dict}
    with open(path + f'/Data/coef_dict_{g_index}.pkl', 'wb') as f:
        pickle.dump(coef_dict, f)
