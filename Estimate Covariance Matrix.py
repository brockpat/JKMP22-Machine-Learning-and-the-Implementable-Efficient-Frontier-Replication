# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:38:52 2025

@author: Patrick

This file computes the covariance matrix as in (37).
"""

#%% Libraries

import pickle

import pandas as pd
import sqlite3
from pandas.tseries.offsets import MonthEnd

import numpy as np
from tqdm import tqdm
from numba import njit

#import dask.dataframe as dd
#https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.apply.html
#https://docs.dask.org/en/latest/generated/dask.dataframe.api.GroupBy.apply.html
#import swifter

#Import user-specific functions
from General_functions import *

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout the code
settings, pf_set = get_settings()

#%% Load DataSets

#Connect to DataBases
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")
crsp_daily = sqlite3.connect(database = path + "Data/crsp_daily_SP500.db")

#Select Time Frame (Chunk) to read in
#start_date = "1952-01-01"
#end_date = "2024-12-31"

# =============================================================================
#                           Chars
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
#                               Daily Data
# =============================================================================
print("Reading in Daily Data (Full Date Range)")
#Read in Data
#   Query
query_daily = ("SELECT permno as id, date, ret_excess as ret_exc FROM d_ret_ex "
#+ f"WHERE date BETWEEN '{start_date}' AND '{end_date}'" 
#+  "AND CAST(id AS INTEGER) <= 99999" (only CRSP Data in my Dataset anyway)
)
#   Read in Data
daily = pd.read_sql_query(query_daily,
                          con = crsp_daily,
                          parse_dates = {'date'}
                          )
# Filter
valid_ids = chars[chars['valid'] == True]['id'].unique()
daily = daily[(~daily['ret_exc'].isna())  & (daily['id'].isin(valid_ids))] # & (daily['id'] <= 99999)

# Create end-of-month column
daily['eom'] = daily['date'] + pd.offsets.MonthEnd(0)

print("    Complete.")

# =============================================================================
#                           Additional Data
# =============================================================================
print("Reading in Cluster Labels Data")
cluster_labels = pd.read_csv(path + "Data/cluster_labels_processed.csv")
print("    Complete.")

#%% Cluster Ranks
"""
Compute each stock's rank of a cluster of characteristics in the cross-section
for all valid observations
"""
#Get Valid Observations
chars = chars.loc[chars['valid'], ['id', 'eom', 'size_grp', 'ff12'] + features]

#Rename Object
cluster_data_m = chars
del chars #Note: Modifying Chars & Renaming Saves Memory

#Get Cluster Label Names
clusters = cluster_labels['cluster'].unique().tolist()
print("Cluster Labels are the following\n", clusters)

# Compute each stock's cluster ranks
cluster_ranks = build_cluster_ranks(cluster_data_m, cluster_labels, clusters, features)

#Save Memory
del cluster_labels
"""
Overwrite DataFrame and throw out Characteristics Data for Cluster Ranks
"""
# Add ranks to cluster_data_m
cluster_data_m = cluster_data_m[['id', 'eom', 'size_grp', 'ff12']]

# Calculate "eom_ret" as the last day of the next month
cluster_data_m['eom_ret'] = (pd.to_datetime(cluster_data_m['eom']) + pd.offsets.MonthEnd(1))

# Combine with cluster_ranks_df
cluster_data_m = pd.concat([cluster_data_m, cluster_ranks], axis=1)

"""
Include Industry Dummys
"""
# Add industry or market dummies based on settings
industries = sorted(cluster_data_m['ff12'].dropna().unique())
for ind in industries:
    cluster_data_m[str(ind)] = (cluster_data_m['ff12'] == ind).astype(int)
ind_factors = industries

print("Cluster Ranks Completed.")


#Standardize Cluster Ranks in the cross-section
cluster_data_m[clusters] = (cluster_data_m
                            .groupby('eom')[clusters]
                            .transform(lambda x: (x - x.mean()) / x.std())
                            )

print("Standardisation Completed.")

#%% Prepare DataFrame for Daily Stock Return Regression
"""
Merge 1 month lagged Cluster Ranks to daily Stock Return Data for the 
daily return regression.
"""
#Prepare daily return data
daily = daily[daily['date'] >= cluster_data_m['eom'].min()]

# Define date to merge on
daily['eom_ret'] = daily['eom']

#Merge daily return data with cluster data
#   Merging on eom_ret ==> ranks from previous month are used in the regression, i.e. data was observable
daily = pd.merge(cluster_data_m, daily[['id', 'date', 'ret_exc', 'eom_ret']], 
                          how='inner', on=['id', 'eom_ret'])

#Rename Object
cluster_data_d = daily
del daily #Note: Modifying Chars & Renaming Saves Memory

# Drop rows with any missing data
cluster_data_d.dropna(inplace=True)

print("Daily Stock Return DataFrame Complete.") 

#%% Daily Regression
"""
Conduct the daily stock return regression.
Store the OLS Residual and the OLS-Coefficients, i.e. the daily estimated factor returns
"""
# Prepare column names
factor_cols = ind_factors + clusters
X_cols = factor_cols
y_col = 'ret_exc'

# Extract unique dates
unique_dates = cluster_data_d['date'].unique()

# Prepare result containers
all_residuals = []
all_coefs = []

# Get numerical objects
X_data = cluster_data_d[factor_cols].values
y_data = cluster_data_d[y_col].values
dates = cluster_data_d['date'].values
ids = cluster_data_d['id'].values

#Save Memory
del cluster_data_d

#Compute the Daily Regression
for date in tqdm([date for date in unique_dates], desc="Calculating Daily Return Regression"):
    # Boolean mask for the date
    mask = (dates == date)
    
    #numerical objects
    X = X_data[mask]
    y = y_data[mask]
    id_subset = ids[mask]

    # OLS
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(XtX) @ Xty  # fallback to pseudo-inverse

    # Compute predictions and residuals
    y_pred = X @ coef
    residuals = y - y_pred

    # Save results
    all_residuals.append(pd.DataFrame({
        'id': id_subset,
        'date': date,
        'residual': residuals
    }))
    all_coefs.append(pd.Series(coef, name=date))

#Save Workspace
del X_data, y_data, dates, ids

# DataFrame with Daily OLS-Residuals
spec_risk = pd.concat(all_residuals, axis=0)
spec_risk = spec_risk.sort_values(by=['id', 'date']).reset_index(drop=True)

#Save Workspace
del all_residuals

# Daily Factor Returns DataFrame
fct_ret = (pd.DataFrame(all_coefs).reset_index()
           .set_axis(['date'] + factor_cols, axis=1)
           .sort_values(by = 'date', ignore_index=True)
           )
#Store Dates to be able to delete fct_ret later to save workspace
fct_dates = fct_ret['date']

#Save Workspace
del all_coefs

print("Factor Returns Completed.")

#%% Factor Covariances
"""
Compute the Daily Factor Covariances and the Daily Squared OLS Residuals
(Stock Specific Risk) using exponentially weighted moving averages for every
last trading day of the month.

For monthly conversion, this data must be scaled by 21.
"""

# =============================================================================
#                           Preliminaries
# =============================================================================

# Compute weights for exponential moving average
time_range = np.arange(settings['cov_set']['obs'], 0, -1)
w_cov = (0.5 ** (1 / settings['cov_set']['hl_cor'])) ** time_range
w_var = (0.5 ** (1 / settings['cov_set']['hl_var'])) ** time_range

#Get Dates at which the Barra VarCov is computed (calc_dates)
ref_date = fct_ret['date'].iloc[settings['cov_set']['obs']] #at least 2520 trading days of lookback data
min_date = ref_date.to_period('M').to_timestamp() - pd.offsets.MonthEnd(1)
calc_dates = sorted(cluster_data_m[cluster_data_m['eom'] >= min_date]['eom'].unique()) # monthly dates for the VarCov


# =============================================================================
#                Factor Covariance Matrix V(f_{t+1})
# =============================================================================

# Initialize factor covariance estimation
factor_cov_est = {}

for d in tqdm([date for date in calc_dates], desc="Calculating factor covariances"):
    
    # Determine first observation date
    eligible_dates = fct_ret[fct_ret['date'] <= d]['date']
    first_obs = eligible_dates.iloc[-settings['cov_set']['obs']:].min()
    
    # Filter covariance data
    cov_data = fct_ret[(fct_ret['date'] >= first_obs) & (fct_ret['date'] <= d)]
    
    t = len(cov_data)
    
    """
    I believe it is wrong how JKMP compute the following, as the weights in the correlation
    matrix are mixed with the weights from the variance matrix.
    See also: https://chatgpt.com/share/6808cfe4-1758-8003-9a0f-816c0bde45a4
    """
    #====================
    # JKMP IMPLEMENTATION
    #=====================
    #Weighted Correlation Matrix
    cor_est = weighted_cor_wt(cov_data.drop('date',axis=1),w_cov[-t:])
    #Weighted Variance Matrix
    var_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_var[-t:])    
    
    # Calculate diagonal matrix with standard deviations on diagonal
    sd_diag = np.diag(np.sqrt(np.diag(var_est)))

    # Get Covariance Matrix by multiplying correlation matrix with standard deviation
    cov_est = (sd_diag @ cor_est @ sd_diag).set_index(cov_data.drop('date', axis=1).columns)
    cov_est.columns = cov_data.drop('date', axis=1).columns  
    
    #===================
    # My Implementation
    #===================
    """    
    The Problem with my implementation is that it can give negative variances
    for a stock when computing the barra covariance matrix S @ cov_est @ S' + diag(resid**2)
    
    cov_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_cov[-t:])
    var_est = weighted_cov_wt(cov_data.drop('date',axis=1),w_var[-t:])
    cov_est.values[[range(len(cov_est))], [range(len(cov_est))]] = var_est.values.diagonal()
    
    
    """
    factor_cov_est[d] = cov_est
    
#Save Workspace
del fct_ret
    
# =============================================================================
#               Idiosyncratic Variance diag(epsilon_{t+1})
# =============================================================================

#Define the moving average function
@njit
def ewma_vol(x, lam, start):
    """
    Compute EWMA volatility estimate assuming mean = 0.

    Parameters:
    - x: np.ndarray, 1D array of residuals
    - lam: float, smoothing factor (e.g., 0.94)
    - start: int, number of initial observations for variance estimate

    Returns:
    - vol: np.ndarray, same size as x, with np.nan for first `start` elements
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    vol = np.full(n, np.nan)

    if n <= start:
        return vol

    # Compute initial variance from the first `start` observations (skip NaNs)
    valid_x = x[:start]
    mask = ~np.isnan(valid_x)
    count = np.sum(mask)
    if count <= 1:
        return vol

    initial_var = np.sum(valid_x[mask] ** 2) / (count - 1)
    var = np.empty(n)
    var[:start] = np.nan
    var[start] = initial_var

    # Iteratively Updating EWMA Vol
    for i in range(start + 1, n):
        if np.isnan(x[i - 1]):
            var[i] = var[i - 1]
        else:
            #MA over squared residuals
            var[i] = lam * var[i - 1] + (1 - lam) * x[i - 1] ** 2

    vol = np.sqrt(var)
    return vol

#wrapper that pandas.apply() can call
def ewma_wrapper(series):
    """
    series → numpy → Numba → Series
    """
    arr = series.values.astype(float)        # ensure float array
    out = ewma_vol(arr, 
                   0.5**(1/settings['cov_set']['hl_stock_var']), 
                   settings['cov_set']['initial_var_obs'])          
    return pd.Series(out, index=series.index)

# Compute exponentially weighted moving average to get MONTHLY squared OLS residual
spec_risk['res_vol'] = (
    spec_risk
      .groupby('id')['residual']
      .apply(ewma_wrapper)
      .values
)

#--- Require that the observation has at least 200 non-missing oversations out of the last 252 trading days
# Create td_range dataframe with date and td_252d columns (252-day lag)
td_range = pd.DataFrame({'date': fct_dates,
                         'td_252d': fct_dates.shift(252)
                         })

# Merge spec_risk with td_range on date column
spec_risk = pd.merge(spec_risk, td_range, on='date', how='left')

# Add date_200d column with 200-day lag grouped by id
spec_risk['date_200d'] = spec_risk.groupby('id')['date'].shift(200)

# Filter rows where date_200d >= td_252d and res_vol is not NA
spec_risk = spec_risk[
    (spec_risk['date_200d'] >= spec_risk['td_252d']) & 
    (spec_risk['res_vol'].notna())
][['id', 'date', 'res_vol']]


#--- Extract specific risk by month end (note, these are still daily volatilities, we
#       only keep the end of month observation. Upscaling to monthly volatility happens
#       in the next section )
# Create eom_ret column (end of month date)
spec_risk['eom_ret'] = spec_risk['date'] + MonthEnd(0)

# Create max_date column (maximum date for each id and eom_ret combination)
spec_risk['max_date'] = spec_risk.groupby(['id', 'eom_ret'])['date'].transform('max')

# Filter to keep only rows where date equals max_date, and select specific columns
spec_risk_m = spec_risk[spec_risk['date'] == spec_risk['max_date']][['id', 'eom_ret', 'res_vol']].copy()

# Rename eom_ret to eom
spec_risk_m = spec_risk_m.rename(columns={'eom_ret': 'eom'})

#Save Workspace
del spec_risk

#%% Barra Covariance Matrix
"""
Compute the Barra Covariance Matrix as in (37). Note: Only the relevant objects
to do the final computation in (37) are saved.
"""
#Initialise Object
barra_cov = {}

#Loop over Dates
for d in tqdm(calc_dates, desc="Calculating Barra Cov"):  
    # Filter cluster_data_m for current date
    char_data = cluster_data_m[cluster_data_m['eom'] == d].copy()
    
    # Merge with specific risk data
    char_data = char_data.merge(spec_risk_m, on=['id', 'eom'], how='left')
    
    # Calculate median res_vol by size group and date
    char_data['med_res_vol'] = char_data.groupby(['size_grp', 'eom'])['res_vol'].transform(
        lambda x: x.median(skipna=True)
    )
    
    # Handle edge case where med_res_vol is still NA
    if char_data['med_res_vol'].isna().any():
        char_data['med_res_vol_all'] = char_data.groupby('eom')['res_vol'].transform(
            lambda x: x.median(skipna=True)
        )
        char_data.loc[char_data['med_res_vol'].isna(), 'med_res_vol'] = char_data['med_res_vol_all']
    
    # Impute missing res_vol with median values
    char_data.loc[char_data['res_vol'].isna(), 'res_vol'] = char_data['med_res_vol']
    
    # Get factor covariance matrix and monthly scaling (21 trading days in a month))
    fct_cov = factor_cov_est[d] * 21
    
    # Sort by id to ensure alignment
    char_data = char_data.sort_values('id')
    
    # Create factor loading matrix
    X = char_data[fct_cov.columns].values
    X = pd.DataFrame(X, index=char_data['id'].astype(int), columns=fct_cov.columns)
    
    # Create idiosyncratic variance vector (monthly scaling - 21 trading days in a month)
    ivol_vec = char_data['res_vol']**2 * 21
    ivol_vec.index = char_data['id'].astype(int)
        
    # Store results
    barra_cov[d] = {
        "fct_load": X,
        "fct_cov": fct_cov,
        "ivol_vec": ivol_vec
    }
#%% Save Dictionaries       
with open(path + '/Data/Barra_Cov.pkl', 'wb') as f:
    pickle.dump(barra_cov, f)

