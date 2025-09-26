# -*- coding: utf-8 -*-
"""
Created on Thu May 15 09:32:41 2025

@author: Patrick

Computes the optimal Aim Portfolio from (40). The optimal Aim portfolio picks
the best hyperparameter combination and thus the best beta to compute A_t for
each month.
"""

#%% Libraries

#Set Working Directory
import pandas as pd
import numpy as np
from functools import reduce
import pickle
from tqdm import tqdm

import sqlite3

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

#%% Connect to DataBase
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")

#%% Reading in Data

# =============================================================================
#                           Preprocessed Chars
# =============================================================================

print("Reading in preprocessed Chars (Full Date Range)")
#Build query vector for features:
#   Set Empty String
query_features =""
#   Fill the Empty String with the features
for feature in features:
    query_features = query_features + feature + ", "
#   Delete last comma and space to avoid syntax errors
query_features = query_features[:-2]
#   Build final query vector
query = ("SELECT id, eom, sic, ff49, size_grp, me, crsp_exchcd, dolvol, lambda," 
         " rvol_m, tr_ld0, eom_ret, ret_ld1, tr_ld1, mu_ld0, ff12, valid, "
         + query_features 
         +" FROM Factors_processed " 
         #+f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
         #+"AND CAST(id AS INTEGER) <= 99999" #Filter for CRSP observations (id <= 99999)
         # Not required as I only have CRSP Data in my DataSet anyway
         )

# Read in JKP characteristics data.
chars  = pd.read_sql_query(query, 
                           con=JKP_Factors,
                           parse_dates=['eom', 'eom_ret']
                           )
#Ensure 'valid' flag is of boolean type
chars['valid'] = chars['valid'].astype(bool)
print("    Complete.")
   
# =============================================================================
#                               Important Dates
# =============================================================================
start_oos = settings['pf']['dates']['start_year'] + settings['pf']['dates']['split_years']

settings['split']['test_end'] = pd.to_datetime('2023-12-31')
dates_oos = pd.date_range(
    start=pd.Timestamp(f"{start_oos}-01-01"),
    end=settings['split']['test_end'] + pd.Timedelta(days=1) - pd.DateOffset(months=1),
    freq='MS'
) - pd.DateOffset(days=1)

#%% Inputs
data_tc = chars

#%% Function

#Initialise dictionary to store the aim portfolio
aims = {}

for g_index, g in enumerate(g_vec):
    print('g_index: ', g_index)
    #===================
    # Load Data
    #===================
    #signal_t from pfml_input
    with open(path + f'/Data/pfml_input_{g_index}.pkl', "rb") as f:
        pfml_input = pickle.load(f)[g_index]
    del pfml_input['reals']
    
    #Validation from pfml_hp_reals
    validation = pd.read_csv(path + "Data/validation.csv", parse_dates = ['eom', 'eom_ret'])
    validation = validation[validation['g'] == g_index]
    
    #Estimated Coefficients for all hyperparameter combinations
    with open(path + f'/Data/coef_dict_{g_index}.pkl', "rb") as f:
        coef_dict = pickle.load(f)[g_index]
    #Relabelling
    hp_coef = coef_dict
    
    #Get best performing estimates at the end of the year
    opt_hps = (validation[(validation['eom_ret'].dt.month == 12) & (validation['rank'] == 1)]
               .assign(hp_end = lambda x: x['eom_ret'].dt.year)
               .get(['hp_end','l','p'])
               .reset_index(drop=True)
               )
    
    #Compute optimal aim portfolio for each month
    output = {}
    for d in dates_oos:
        print("    Date: ", d)
        d_ret = d+pd.offsets.MonthEnd(1)
        oos_year = d_ret.year
        hp_year = oos_year-1
        
        #Extract optimal hyperparameter combination
        hps_d = opt_hps[opt_hps['hp_end'] == hp_year]
        
        #Get optimal number of RFFs
        feat = pfml_feat_fun(p = hps_d['p'].values[0])
        #Extract the signals
        s = pfml_input['signal_t'][d][feat]
        
        #Extract the optimal coefficients
        l_no =hps_d['l'].values[0]
        coef = hp_coef[oos_year][hps_d['p'].values[0]][l_no].loc[feat]
        
        #Compute the optimal aim portfolio
        aim_pf = (data_tc.loc[(data_tc['valid']==True) & (data_tc['eom']==d)]
                  .assign(w_aim = s.to_numpy() @ coef.to_numpy())
                  .get(["id","eom","w_aim"])
                  )
        
        #Save Result
        output[d] = {'aim_pf':aim_pf, 'coef':coef}
        
    aims[g_index] = output
    
print("Saving Data")
with open(path + f'/Data/aims.pkl', 'wb') as f:
    pickle.dump(aims, f)
    
