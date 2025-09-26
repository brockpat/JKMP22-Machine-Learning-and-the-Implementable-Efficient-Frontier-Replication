# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:39:51 2025

@author: Patrick

This file computes the utility at each month, 
i.e. \tilde{r}_{t+1} beta - 0.5 beta' \tilde{\Sigma}_t beta,
for every hyperparameter combination.

Based on this, the cumulative utility up to date, i.e. (25), for each
hyperparameter combination is computed.

Lastly, each utility is ranked in order to be able to pick the best hyperparameter
combination. The results are exported as a dataframe called 'validation'

"""

#%% Libraries

import pandas as pd
import numpy as np
import pickle

#Run specific functions
from General_functions import *

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout
settings, pf_set = get_settings()

hp_years = np.arange(settings['pf']['dates']['start_year'], settings['pf']['dates']['end_yr'] + 1)

# Investor
mu = pf_set['mu']
gamma_rel = pf_set['gamma_rel']

# Hyperparameters
rff_feat = True
g_vec = settings['pf_ml']['g_vec']
p_vec = settings['pf_ml']['p_vec']
l_vec = settings['pf_ml']['l_vec']
scale = settings['pf_ml']['scale']
orig_feat = settings['pf_ml']['orig_feat']

# Other parameters
iterations = 10  
hps = None
balanced = False

#%% Function Inputs

#List to Store DataFrame outside of g_vec-Loop
list_validation_df =  []

#List to Store DataFrame within g_vec-Loop
rows = []

for g_index, g in enumerate(g_vec):
    print("g_index: ", g_index)

    with open(path + f'/Data/pfml_input_{g_index}.pkl', "rb") as f:
        pfml_input = pickle.load(f)[g_index]
        
    d_all = pd.Series(sorted(list(pfml_input['reals'].keys()))) 
    
    with open(path + f'/Data/coef_dict_{g_index}.pkl', "rb") as f:
        hp_coef = pickle.load(f)[g_index]

    for end in hp_years: 
        print("     Year: ", end)
        #Subset
        mask = (pd.to_datetime(f"{end-1}-12-31") <= d_all) & (d_all <=  pd.to_datetime(f"{end}-11-30"))
        reals_all = {key: pfml_input['reals'][key] for key in d_all[mask]}
        
        coef_dict_yr = hp_coef[end]
        
        for p in p_vec:
            feat_p = pfml_feat_fun(p = p)
            coef_dict_p = coef_dict_yr[p]
            reals = {date: {'r_tilde': data['r_tilde'].loc[feat_p], 'denom': data['denom'].loc[feat_p,feat_p]} 
                     for date, data in reals_all.items()
                     }
            for i,_ in enumerate(l_vec):
                coef = coef_dict_p[i]
                
                for nm, x in reals.items(): 
                    r_tilde = x['r_tilde']
                    denom = x['denom']
                    
                    r = r_tilde.to_numpy().T @ coef.to_numpy() - 0.5*coef.to_numpy().T @ denom.to_numpy() @ coef.to_numpy()
                    row = {'eom': pd.to_datetime(nm),
                           'eom_ret': nm + pd.offsets.MonthEnd(1),
                           'obj': r,
                           'l': i,
                           'p': p,
                           'hp_end':end,
                           }
                    rows.append(row)
    print("     Creating DataFrame")                
    validation = pd.DataFrame(rows)
    
    validation = validation.sort_values(['p', 'l', 'eom_ret'])
    
    # Cumulative mean by (p, l)
    validation['cum_obj'] = (
        validation
        .groupby(['p', 'l'])['obj']
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
    )
    
    # Rank by -cum_obj within each eom_ret
    validation['rank'] = (
        validation
        .groupby('eom_ret')['cum_obj']
        .rank(ascending=False, method='dense')
    )
    
    #Get g_index
    validation = validation.assign(g = g_index)
    
    list_validation_df.append(validation)

print("Exporting Final DataFrame")                
pd.concat(list_validation_df).to_csv(path + "Data/validation.csv", index=False)