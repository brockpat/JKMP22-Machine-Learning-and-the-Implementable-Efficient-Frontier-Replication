# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:16:31 2025

@author: Patrick

Implement the best portfolio and print results.
"""

#%% Libraries
import pandas as pd
import numpy as np
import pickle

from plotnine import *
from pandas.api.types import CategoricalDtype

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import sqlite3

#Run user specific functions
from General_functions import *

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout
settings, pf_set = get_settings()

# Hyperparameters
g_vec = settings['pf_ml']['g_vec']

#Important Dates
start_oos = settings['pf']['dates']['start_year'] + settings['pf']['dates']['split_years']

dates_oos = pd.date_range(
    start=pd.Timestamp(f"{start_oos}-01-01"),
    end=settings['split']['test_end'] + pd.Timedelta(days=1) - pd.DateOffset(months=1),
    freq='MS'
) - pd.DateOffset(days=1)

#%% Get Data
# =============================================================================
#                           Preprocessed Chars
# =============================================================================
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")

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
         " rvol_m, tr_ld0, eom_ret, ret_ld1, tr_ld1, mu_ld0, ff12, valid"
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

#Relabel
data_tc = chars

print("    Complete.")

# =============================================================================
#                       Preprocessed Wealth Evolution
# =============================================================================
wealth = pd.read_csv(path + "Data/wealth_processed.csv",parse_dates=['eom'])

# =============================================================================
#                           Risk-Free Rate
# =============================================================================
# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])

# Convert to decimal (RF is given in percentage terms)
risk_free["rf"] = risk_free["RF"] / 100

#--- Construct an end-of-month date.
risk_free["eom"] = pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

# Keep only the required columns.
risk_free = risk_free[["eom", "rf"]]

# =============================================================================
#                 Estimated Barra Covariance Matrix
# =============================================================================
with open(path + '/Data/Barra_Cov.pkl', "rb") as barra_cov:
    barra_cov = pickle.load(barra_cov)
cov_list = barra_cov
    
# =============================================================================
#                   Kyle's Lambda for each Stock
# =============================================================================
lambda_dates = chars['eom'].drop_duplicates().sort_values()

#If transaction costs enabled, get Kyle's Lambda
if settings['Transaction_Costs']:
    lambda_list = {
        d: chars[chars['eom'] == d].sort_values('id').set_index('id')['lambda'].to_dict()
        for d in lambda_dates
    }
#Set all entries of Kyle's Lambda to 1e-16, effectively getting rid of transaction costs
else:
    lambda_list = {
        d: {ids: 1e-16 for ids in chars[chars['eom'] == d].sort_values('id')['id']}
        for d in lambda_dates
    }
del lambda_dates


mu = pf_set['mu']
gamma_rel = pf_set['gamma_rel']
iterations = 10

with open(path + '/Data/hps.pkl', "rb") as f:
    hps = pickle.load(f)
    
#%% Functions
def initial_weights_new(data, w_type, udf_weights=None):
    """
    Get Initial Portfolio of the time Series  
    """    
    #Value Weighted Portfolio
    if w_type == "vw":
        pf_w = (
            data.groupby("eom")
            .apply(lambda df: df.assign(w_start=df["me"] / df["me"].sum()))
            .reset_index(drop=True)[["id", "eom", "w_start"]]
        )
    
    #Equal Weighted Portfolio
    elif w_type == "ew":
        pf_w = (
            data.groupby("eom")
            .apply(lambda df: df.assign(w_start=1 / len(df)))
            .reset_index(drop=True)[["id", "eom", "w_start"]]
        )
    else:
        raise ValueError(f"Unknown w_type: {w_type}")
    
    # Set w_start to NaN for all eoms except the earliest one
    min_eom = pf_w["eom"].min()
    pf_w.loc[pf_w["eom"] != min_eom, "w_start"] = np.nan
    
    # Add an empty weight column
    pf_w["w"] = np.nan

    return pf_w

def pfml_w(data, dates, cov_list, lambda_list, gamma_rel, iterations, risk_free, wealth, mu, aims):
    """
    Based on an initially value-weighted portfolio, computes the chosen portfolio
    based on the optimal aim portfolio.
    
    Additional Variable Description
    -------------------------------
    tr_ld1: lead return of the asset (Prepare_Data.py).
    mu:     market return (value-weighted)
    """
    #Initial starting weights at the first trading date
    fa_weights = initial_weights_new(data, w_type = "vw")
    fa_weights = pd.merge(data[["id", "eom", "tr_ld1"]], fa_weights, on=["id", "eom"], how="left")
    fa_weights = pd.merge(wealth[wealth['eom'].isin(fa_weights['eom'].unique())][["eom", "mu_ld1"]], fa_weights, on="eom", how="left")

    for d in dates:
        print(d)
        ids = data.loc[data["eom"] == d, "id"]
        sigma = create_cov(cov_list[d], ids=ids)
        K_Lambda = pd.DataFrame(create_lambda(lambda_list[d], ids=ids),columns = ids, index = ids)
        w = wealth.loc[wealth["eom"] == d, "wealth"].values[0]
        rf = risk_free.loc[risk_free["eom"] == d, "rf"].values[0]
        m = m_func(w=w, mu=mu, rf=rf, sigma_gam=sigma * gamma_rel, gam=gamma_rel, K_Lambda=K_Lambda, iterations=iterations)
        iden = np.eye(m.shape[0])
        
        #Compute portfolio weights chosen at date d based on (17)
        w_cur = fa_weights[fa_weights["eom"] == d].merge(aims, on=["id", "eom"], how="left")
        w_cur['w_opt'] = m @ w_cur['w_start'].to_numpy() + (iden - m) @ w_cur['w_aim'].to_numpy()

        #Compute initial portfolio weights at d+1 (they change based on return & market return)
        next_month = d + pd.offsets.MonthEnd(1)
        w_cur["w_opt_ld1"] = w_cur["w_opt"] * (1 + w_cur["tr_ld1"]) / (1 + w_cur["mu_ld1"])
        
        #Merge to original dataframe
        fa_weights = fa_weights.merge(w_cur[["id", "w_opt", "w_opt_ld1"]], on="id", how="left")
        
        # Set column 'w' equal to 'w_opt'
        fa_weights.loc[(~fa_weights["w_opt"].isna()) & (fa_weights["eom"] == d), "w"] = \
            fa_weights["w_opt"]
        
        # Update next month's initial portfolio weights
        fa_weights.loc[(~fa_weights["w_opt"].isna()) & (fa_weights["eom"] == next_month), "w_start"] = \
            fa_weights["w_opt_ld1"]
        
        # Initialize new stocks with zero weight if not already set
        fa_weights.loc[(fa_weights["eom"] == next_month) & (fa_weights["w_start"].isna()), "w_start"] = 0
        
        # Drop temporary columns
        fa_weights = fa_weights.drop(columns=["w_opt", "w_opt_ld1"])
        
    return fa_weights
        
def compute_stats(group):
    w = group["w"].values
    w_start = group["w_start"].values
    ret_ld1 = group["ret_ld1"].values
    lam = group["lambda"].values
    wlth = group["wealth"].iloc[0]  # unique wealth per eom

    inv = np.sum(np.abs(w))
    shorting = np.sum(np.abs(w[w < 0]))
    turnover = np.sum(np.abs(w - w_start))
    r = np.sum(w * ret_ld1)
    tc = (wlth / 2) * np.sum(lam * ((w - w_start) ** 2))

    return pd.Series({
        "inv": inv,
        "shorting": shorting,
        "turnover": turnover,
        "r": r,
        "tc": tc
    })

def pf_ts_fun(weights, data, wealth, gam):
    # 1. Merge Portfolio weights with data on (id, eom)
    comb = pd.merge(data[["id", "eom", "ret_ld1", "lambda"]],
                    weights, on=["id", "eom"], how="inner")

    # 2. Merge with wealth on eom
    comb = pd.merge(comb, wealth[["eom", "wealth"]], on="eom", how="left")

    # 3. Group by eom and compute metrics

    result = comb.groupby("eom").apply(compute_stats).reset_index()

    # 4. Adjust eom_ret to next month's end-of-month
    result["eom_ret"] = (result["eom"] + pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1)

    # 5. Drop original eom column
    result = result.drop(columns=["eom"])

    return result
#%% Get best Hyperparameter combination & respective aim portfolio

# Extract the 'validation' DataFrame from each element in the hps dict
validation_list = [x["validation"] for x in hps.values()]

# Combine all validation DataFrames into one
best_hps = pd.concat(validation_list, ignore_index=True).drop(columns = 'rank')

#Date-time format
best_hps = (best_hps
            .assign(eom = lambda x: pd.to_datetime(x['eom']))
            .assign(eom_ret = lambda x: pd.to_datetime(x['eom_ret']))
            )

# Rank within each 'eom_ret' group by descending 'cum_obj'
best_hps["rank"] = best_hps.groupby("eom_ret")["cum_obj"].rank(ascending=False, method="first")

# Filter: keep only top-ranked rows where eom_ret is in December
best_hps = best_hps[(best_hps["rank"] == 1) & (best_hps["eom_ret"].dt.month == 12)]

# Plot Selected Hyperparameters (best HPs in terms of utility)
best_hps_long = best_hps.melt(
    id_vars=["eom_ret"], value_vars=["p", "l", "g"], var_name="name", value_name="value"
)
plot = (
    ggplot(best_hps_long, aes(x="eom_ret", y="value")) +
    geom_point(alpha=0.5) +
    geom_line() +
    facet_wrap("~name", scales="free_y", ncol=1) +
    labs(title="Top Hyperparameters Over Time", x="End of Month", y="Value")
)
plot.show()

best_hps_dict = {}
for d in dates_oos:
    d_ret       = d+pd.offsets.MonthEnd(1)
    oos_year    = d_ret.year
    best_g      = best_hps[best_hps['eom_ret'].dt.year==oos_year-1]['g']
    best_p      = best_hps[best_hps['eom_ret'].dt.year==oos_year-1]['p']
    best_aim    = hps[best_g.iloc[0]]['aim_pfs_list'][d]['aim_pf']
    best_coef   = hps[best_g.iloc[0]]['aim_pfs_list'][d]['coef']
    
    best_hps_dict[d] = {"g":best_g, "p":best_p, "aim":best_aim, "coef":best_coef}
    
# Extract all "aim" DataFrames from best_hps_list
aims_list = [x["aim"] for x in best_hps_dict.values()]  # use .values() if it's a dict

# Combine into one DataFrame
aims = pd.concat(aims_list, ignore_index=True)

#%% Get Best Portfolio

#Compute the portfolio weight for each stock
w = pfml_w(data = chars[(chars['eom'].isin(dates_oos)) & (chars['valid'])], 
           dates = dates_oos, cov_list=cov_list, lambda_list=lambda_list, gamma_rel=gamma_rel, 
           iterations=iterations, risk_free=risk_free, wealth=wealth, mu=mu, aims = aims)
w.to_csv(path + "Data/weights.csv",index=False)

pf = pf_ts_fun(weights = w, data=data_tc, wealth = wealth, gam = gamma_rel) 

pf.to_csv(path + "Data/pf.csv",index=False)

#Final Dictionary
pfml = {"hps":hps, "best_hps":best_hps, "best_hps_list":best_hps_dict, "aims":aims, "w":w, "pf":pf}

#%% Summary Statistics
pfs = pd.DataFrame(pfml['pf'])

# Set 'type' as ordered categorical
pfs['type'] = "Portfolio-ML"

# Sort by 'type' and 'eom_ret'
pfs = pfs.sort_values(by=['type', 'eom_ret']).reset_index(drop=True)

# Compute adjusted variance by type
pfs['e_var_adj'] = pfs.groupby('type')['r'].transform(lambda x: (x - x.mean())**2)

# Calculate utility_t
gamma_rel = pf_set['gamma_rel']  # Assuming pf_set is a dictionary or similar
pfs['utility_t'] = pfs['r'] - pfs['tc'] - 0.5 * pfs['e_var_adj'] * gamma_rel

# Portfolio summary statistics
grouped = pfs.groupby('type')

pf_summary = grouped.agg(
    n=('r', 'count'),
    inv=('inv', 'mean'),
    shorting=('shorting', 'mean'),
    turnover_notional=('turnover', 'mean'),
    r=('r', lambda x: x.mean() * 12), #Standard Excess Return! r-r_f
    sd=('r', lambda x: x.std(ddof=1) * np.sqrt(12)),
    sr_gross=('r', lambda x: x.mean() / x.std(ddof=1) * np.sqrt(12)),
    tc=('tc', lambda x: x.mean() * 12),
    r_tc=('r', lambda x: (x - pfs.loc[x.index, 'tc']).mean() * 12),
    sr=('r', lambda x: ((x - pfs.loc[x.index, 'tc']).mean() / x.std(ddof=1)) * np.sqrt(12)),
    obj=('r', lambda x: (x.mean() - 0.5 * x.var(ddof=1) * gamma_rel - pfs.loc[x.index, 'tc'].mean()) * 12)
).reset_index()

pf_summary.to_csv(path + "Data/pf_summary.csv",index=False)


#%% Plots

# Assume pfs is your original DataFrame
# Ensure eom_ret is a datetime column
pfs['eom_ret'] = pd.to_datetime(pfs['eom_ret'])

# Calculate cumulative returns per type
pfs['cumret'] = pfs['r'].cumsum()
pfs['cumret_tc'] = (pfs['r'] - pfs['tc']).cumsum()
pfs['cumret_tc_risk'] = pfs['utility_t'].cumsum()


# Filter to main types
main_types = pfs['type'].unique()  # Adjust as needed
ts_data = pfs[pfs['type'].isin(main_types)].copy()
ts_data = ts_data[['type', 'eom_ret', 'cumret', 'cumret_tc', 'cumret_tc_risk']]

# Convert to long format
ts_data_long = ts_data.melt(id_vars=['type', 'eom_ret'], var_name='name', value_name='value')

# Add 0-row at beginning
start_date = (pfs['eom_ret'].min().to_period('M').to_timestamp('M')) - pd.Timedelta(days=1)
zero_data = pd.DataFrame({
    'eom_ret': [start_date] * 3 * len(main_types),
    'type': list(main_types) * 3,
    'value': [0] * 3 * len(main_types),
    'name': ['cumret'] * len(main_types) + ['cumret_tc'] * len(main_types) + ['cumret_tc_risk'] * len(main_types)
})
ts_data_long = pd.concat([zero_data, ts_data_long], ignore_index=True)

# Pretty names for facets
name_map = {
    'cumret': 'Gross return',
    'cumret_tc': 'Return net of TC',
    'cumret_tc_risk': 'Return net of TC and Risk'
}
ts_data_long['name_pretty'] = ts_data_long['name'].map(name_map)
ts_data_long['name_pretty'] = pd.Categorical(
    ts_data_long['name_pretty'],
    categories=["Gross return", "Return net of TC", "Return net of TC and Risk"],
    ordered=True
)

# Generate plots
plots = []
for name in ts_data_long['name'].unique():
    data = ts_data_long[ts_data_long['name'] == name]
    plot = (
        ggplot(data, aes('eom_ret', 'value', color='type', linetype='type')) +
        geom_line() +
        facet_wrap('~name_pretty', scales='free') +
        coord_cartesian(ylim=(0, None)) +
        labs(y='Cumulative performance', color='Method:', linetype='Method:') +
        scale_x_datetime(date_labels='%Y', breaks=pd.date_range('1980-12-31', '2023-12-31', freq='5Y')) +
        theme(
            axis_title_x=element_blank(),
            strip_background=element_rect(fill='white', color='black'),
            text=element_text(size=11),
            legend_position='none'
        )
    )
    plot.show()