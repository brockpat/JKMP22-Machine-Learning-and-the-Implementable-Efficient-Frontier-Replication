# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:12:45 2025

@author: Patrick
"""

import numpy as np
import pandas as pd
#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from statsmodels.distributions.empirical_distribution import ECDF
import re
from joblib import Parallel, delayed

from scipy.linalg import sqrtm

#%%

# =============================================================================
#                   Baseline Settings & Features used
# =============================================================================

def get_settings():
    settings = {
        "parallel": True,
        "seed_no": 1,
        "months": False,
        "Transaction_Costs":True, 
        "split": {
            "train_end": pd.to_datetime("1970-12-31"),
            "test_end": pd.to_datetime('2023-12-31'), #General Version: chars.eom.max().dt.year -1
            "val_years": 10,
            "model_update_freq": "yearly",  # Options: "once", "yearly", "decade"
            "train_lookback": 1000,
            "retrain_lookback": 1000
        },
        "feat_prank": True,
        "ret_impute": "zero",
        "feat_impute": True,
        "addition_n": 12,
        "deletion_n": 12,
        "screens": {
            "start": pd.to_datetime("1952-01-31"),
            "end": pd.to_datetime("2023-12-31"), #General Version: chars.eom.max().dt.year -1
            "feat_pct": 0.5,
            "nyse_stocks": False 
        },
        "pi": 0.1,  # Price impact for trading 1% of daily volume
        "rff": {
            "p_vec": [2**i for i in range(1, 10)],
            "g_vec": list(np.exp(np.arange(-3, -1))),  
            "l_vec": [0.0] + list(np.exp(np.linspace(-10, 10, 100)))
        },
        "pf": {
            "dates": {
                "start_year": 1971,
                "end_yr": 2023, #General Version: chars.eom.max().dt.year -1
                "split_years": 10
            },
            "hps": {
                "cov_type": "cov_add",
                "m1": {
                    "k": [1, 2, 3],
                    "u": [0.25, 0.5, 1],
                    "g": [0, 1, 2],
                    "K": 12
                },
                "static": {
                    "k": [1.0, 1/3, 1/5],
                    "u": [0.25, 0.5, 1],
                    "g": [0, 1, 2]
                }
            }
        },
        "pf_ml": {
            "g_vec": list(np.exp(np.arange(-3, -1))),  # [-3, -2]
            "p_vec": [2**i for i in range(6, 10)],
            "l_vec": [0.0] + list(np.exp(np.linspace(-10, 10, 100))),
            "orig_feat": False,
            "scale": True
        },
        "ef": {
            "wealth": [1, 1e9, 1e10, 1e11],
            "gamma_rel": [1, 5, 10, 20, 100]
        },
        "cov_set": {
            "industries": True,
            "obs": 252 * 10,
            "hl_cor": int(252 * 3 / 2),
            "hl_var": int(252 / 2),
            "hl_stock_var": int(252 / 2),
            "min_stock_obs": 252,
            "initial_var_obs": 21 * 3
        },
        "factor_ml": {
            "n_pfs": 10
        }
    }
    
    pf_set = {
        "wealth": 1e10,
        "gamma_rel": 10,
        "mu": 0.007,  # Expected monthly portfolio return
        "lb_hor": 11  # Lower bound on horizon for (24)
    }
    return settings,pf_set

#Get the List of Features
# Features ---------------------
def get_features(exclude_poor_coverage = True):
    features = [
      "age",                 "aliq_at",             "aliq_mat",            "ami_126d",           
      "at_be",               "at_gr1",              "at_me",               "at_turnover",        
      "be_gr1a",             "be_me",               "beta_60m",            "beta_dimson_21d",    
      "betabab_1260d",       "betadown_252d",       "bev_mev",             "bidaskhl_21d",       
      "capex_abn",           "capx_gr1",            "capx_gr2",            "capx_gr3",           
      "cash_at",             "chcsho_12m",          "coa_gr1a",            "col_gr1a",           
      "cop_at",              "cop_atl1",            "corr_1260d",          "coskew_21d",         
      "cowc_gr1a",           "dbnetis_at",          "debt_gr3",            "debt_me",            
      "dgp_dsale",           "div12m_me",           "dolvol_126d",         "dolvol_var_126d",    
      "dsale_dinv",          "dsale_drec",          "dsale_dsga",          "earnings_variability",
      "ebit_bev",            "ebit_sale",           "ebitda_mev",          "emp_gr1",            
      "eq_dur",              "eqnetis_at",          "eqnpo_12m",           "eqnpo_me",           
      "eqpo_me",             "f_score",             "fcf_me",              "fnl_gr1a",           
      "gp_at",               "gp_atl1",             "ival_me",             "inv_gr1",            
      "inv_gr1a",            "iskew_capm_21d",      "iskew_ff3_21d",       "iskew_hxz4_21d",     
      "ivol_capm_21d",       "ivol_capm_252d",      "ivol_ff3_21d",        "ivol_hxz4_21d",      
      "kz_index",            "lnoa_gr1a",           "lti_gr1a",            "market_equity",      
      "mispricing_mgmt",     "mispricing_perf",     "ncoa_gr1a",           "ncol_gr1a",          
      "netdebt_me",          "netis_at",            "nfna_gr1a",           "ni_ar1",             
      "ni_be",               "ni_inc8q",            "ni_ivol",             "ni_me",              
      "niq_at",              "niq_at_chg1",         "niq_be",              "niq_be_chg1",        
      "niq_su",              "nncoa_gr1a",          "noa_at",              "noa_gr1a",           
      "o_score",             "oaccruals_at",        "oaccruals_ni",        "ocf_at",             
      "ocf_at_chg1",         "ocf_me",              "ocfq_saleq_std",      "op_at",              
      "op_atl1",             "ope_be",              "ope_bel1",            "opex_at",            
      "pi_nix",              "ppeinv_gr1a",         "prc",                 "prc_highprc_252d",   
      "qmj",                 "qmj_growth",          "qmj_prof",            "qmj_safety",         
      "rd_me",               "rd_sale",             "rd5_at",              "resff3_12_1",        
      "resff3_6_1",          "ret_1_0",             "ret_12_1",            "ret_12_7",           
      "ret_3_1",             "ret_6_1",             "ret_60_12",           "ret_9_1",            
      "rmax1_21d",           "rmax5_21d",           "rmax5_rvol_21d",      "rskew_21d",          
      "rvol_21d",            "sale_bev",            "sale_emp_gr1",        "sale_gr1",           
      "sale_gr3",            "sale_me",             "saleq_gr1",           "saleq_su",           
      "seas_1_1an",          "seas_1_1na",          "seas_11_15an",        "seas_11_15na",       
      "seas_16_20an",        "seas_16_20na",        "seas_2_5an",          "seas_2_5na",         
      "seas_6_10an",         "seas_6_10na",         "sti_gr1a",            "taccruals_at",       
      "taccruals_ni",        "tangibility",         "tax_gr1a",            "turnover_126d",      
      "turnover_var_126d",   "z_score",             "zero_trades_126d",    "zero_trades_21d",    
      "zero_trades_252d",
      "rvol_252d"
    ]
    
    # Exclude features without sufficient coverage
    feat_excl = ["capex_abn", "capx_gr2", "capx_gr3", "debt_gr3", "dgp_dsale",           
                   "dsale_dinv", "dsale_drec", "dsale_dsga", "earnings_variability", "eqnetis_at",          
                   "eqnpo_me", "eqpo_me", "f_score", "iskew_hxz4_21d", "ivol_hxz4_21d",       
                   "netis_at", "ni_ar1", "ni_inc8q", "ni_ivol", "niq_at", "niq_at_chg1", "niq_be", 
                   "niq_be_chg1", "niq_su", "ocfq_saleq_std", "qmj", "qmj_growth", "rd_me", 
                   "rd_sale", "rd5_at", "resff3_12_1", "resff3_6_1", "sale_gr3", "saleq_gr1", 
                   "saleq_su", "seas_16_20an", "seas_16_20na", "sti_gr1a", "z_score"
    ]
    if exclude_poor_coverage:
        # Filter out the excluded features
        features = [feature for feature in features if feature not in feat_excl]
    
    return features

#%%

# Wealth Calculation Function
def wealth_func(wealth_end, end, market, risk_free):
    """
    Compute the wealth trajectory backwards from a given initial wealth
    level. Since backtesting works backwards, for a given initial wealth level
    we need to know how our strategy would have performed in the past to evaluate
    it.
    
    Wealth is assumed to grow exogenously at the market return.
    
    Parameters:
      wealth_end: final wealth value
      end: final date (as a pd.Timestamp)
      market: DataFrame with market returns, containing 'eom_ret' and 'mkt_vw_exc'
      risk_free: DataFrame with risk-free data, containing 'eom' and 'rf'
      
    Returns:
      A DataFrame with columns: 'eom' (end-of-month), 'wealth', and 'mu_ld1' (total return for the period)
    """
    # Rename and merge
    risk_free = risk_free.rename(columns={"eom": "eom_ret"})
    wealth = pd.merge(risk_free[["eom_ret", "rf"]], market, on="eom_ret", how="left")

    # Compute total return
    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    # Filter for dates up to 'end'
    wealth = wealth[wealth["eom_ret"] <= end]

    # Sort descending by date and compute cumulative wealth backward
    wealth = wealth.sort_values("eom_ret", ascending=False)
    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end

    # Final output formatting
    wealth["eom"] = (wealth["eom_ret"].dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1))
    wealth["mu_ld1"] = wealth["tret"]

    result = wealth[["eom", "wealth", "mu_ld1"]].copy()

    # Add one row for the exact 'end' date
    row = pd.DataFrame([{"eom": end, "wealth": wealth_end, "mu_ld1": np.nan}])
    result = pd.concat([result, row], ignore_index=True)

    # Sort chronologically
    result = result.sort_values("eom").reset_index(drop=True)

    return result

def long_horizon_ret(data, h, impute="zero"):
    """
    Compute long-horizon returns.
    
    Parameters:
      data: pandas DataFrame with at least columns 'id', 'eom', 'ret_exc'
      h: int, horizon (number of future periods)
      impute: str, imputation method in {"zero", "mean", "median"}
      
    Returns:
      DataFrame with future return columns ret_ld1, ..., ret_ld{h}.
    """
    # Ensure that the input data is not modified
    data = data.copy()
    
    #Ensure 'eom' is datetime
    data['eom'] = pd.to_datetime(data['eom'])
    
    # Only consider rows where ret_exc is not NA.
    valid_data = data.dropna(subset=["ret_exc"])
    
    # Get unique dates (for merging) present in valid_data.
    dates = valid_data[['eom']].drop_duplicates()
    dates = dates.rename(columns={'eom': 'merge_date'})
    
    # For each id, find the start and end dates where ret_exc is not missing.
    ids = valid_data.groupby('id')['eom'].agg(start='min', end='max').reset_index()
    
    # Create a cross between each security’s valid date range and the unique dates.
    # For each id, select the dates between start and end.
    id_dates = ids.merge(dates, how="cross")
    id_dates = id_dates[(id_dates['merge_date'] >= id_dates['start']) & 
                        (id_dates['merge_date'] <= id_dates['end'])]
    
    # Remove the extra start/end columns.
    id_dates = id_dates.drop(columns=["start", "end"]).rename(columns={"merge_date": "eom"})
    
    # Merge the full panel with the original return data.
    full_ret = pd.merge(id_dates, data[['id', 'eom', 'ret_exc']], on=["id", "eom"], how="left")
    full_ret = full_ret.sort_values(by=["id", "eom"]).reset_index(drop=True)
    
    # Create the lead return columns ret_ld1, ..., ret_ldh within each id group.
    for l in range(1, h+1):
        full_ret[f"ret_ld{l}"] = full_ret.groupby("id")["ret_exc"].shift(-l)
    
    # Drop the original ret_exc column.
    full_ret = full_ret.drop(columns=["ret_exc"])
    
    # Identify rows where all lead returns are missing.
    lead_cols = [f"ret_ld{l}" for l in range(1, h+1)]
    all_missing = full_ret[lead_cols].isna().sum(axis=1) == h
    perc_all_missing = all_missing.mean() * 100
    print(f"All missing excludes {perc_all_missing:.2f}% of the observations")
    
    # Keep only rows where not all lead returns are missing.
    full_ret = full_ret[~all_missing].reset_index(drop=True)
    
    # Impute missing values in the lead columns based on the specified method.
    if impute == "zero":
        full_ret[lead_cols] = full_ret[lead_cols].fillna(0)
    elif impute == "mean":
        # Replace NA with the mean of the column within each month (eom group).
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.mean()))
    elif impute == "median":
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.median()))
    
    return full_ret

#Categories Industries According to Fama-French 12 Industries
#Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
#Categories can also be found in the data folder
def categorize_sic(sic):
    """
    Assigns Fama-French 12 industry classification based on SIC codes.

    Parameters:
    -----------
    chars : pd.DataFrame
        DataFrame containing a 'sic' column with Standard Industrial Classification codes
        
    Returns:
    --------
    pd.Series
        Series containing industry classifications according to Fama-French 12 categories:
        - NoDur: Non-durable Consumer Goods
        - Durbl: Durable Consumer Goods
        - Manuf: Manufacturing
        - Enrgy: Energy
        - Chems: Chemicals
        - BusEq: Business Equipment
        - Telcm: Telecommunications
        - Utils: Utilities
        - Shops: Shops (Wholesale/Retail)
        - Hlth: Healthcare
        - Money: Financial Services
        - Other: All other industries
        
    Note:
    -----
    Follows the standard Fama-French 12 industry classification scheme based on SIC code ranges.
    """
    # Non-Durables
    if ((100 <= sic <= 999) or
        (2000 <= sic <= 2399) or
        (2700 <= sic <= 2749) or
        (2770 <= sic <= 2799) or
        (3100 <= sic <= 3199) or
        (3940 <= sic <= 3989)):
        return "NoDur"
    
    # Durables
    elif ((2500 <= sic <= 2519) or
          (3630 <= sic <= 3659) or
          (sic in [3710, 3711, 3714, 3716, 3750, 3751, 3792]) or
          (3900 <= sic <= 3939) or
          (3990 <= sic <= 3999)):
        return "Durbl"
    
    # Manufacturing
    elif ((2520 <= sic <= 2589) or
          (2600 <= sic <= 2699) or
          (2750 <= sic <= 2769) or
          (3000 <= sic <= 3099) or
          (3200 <= sic <= 3569) or
          (3580 <= sic <= 3629) or
          (3700 <= sic <= 3709) or
          (3712 <= sic <= 3713) or
          (sic in [3715]) or
          (3717 <= sic <= 3749) or
          (3752 <= sic <= 3791) or
          (3793 <= sic <= 3799) or
          (3830 <= sic <= 3839) or
          (3860 <= sic <= 3899)):
        return "Manuf"
    
    # Energy
    elif ((1200 <= sic <= 1399) or
          (2900 <= sic <= 2999)):
        return "Enrgy"
    
    # Chemicals
    elif ((2800 <= sic <= 2829) or
          (2840 <= sic <= 2899)):
        return "Chems"
    
    # Business Equipment
    elif ((3570 <= sic <= 3579) or
          (3660 <= sic <= 3692) or
          (3694 <= sic <= 3699) or
          (3810 <= sic <= 3829) or
          (7370 <= sic <= 7379)):
        return "BusEq"
    
    # Telecommunications
    elif (4800 <= sic <= 4899):
        return "Telcm"
    
    # Utilities
    elif (4900 <= sic <= 4949):
        return "Utils"
    
    # Shops
    elif ((5000 <= sic <= 5999) or
          (7200 <= sic <= 7299) or
          (7600 <= sic <= 7699)):
        return "Shops"
    
    # Health
    elif ((2830 <= sic <= 2839) or
          (sic == 3693) or
          (3840 <= sic <= 3859) or
          (8000 <= sic <= 8099)):
        return "Hlth"
    
    # Money
    elif (6000 <= sic <= 6999):
        return "Money"
    
    # Other
    else:
        return "Other"

def size_screen_fun(chars, type_):
    """
    Applies a screening on stock size based on the passed type.
    This modifies the DataFrame (chars) in place. The supported types are:
      - "all": No additional screening; set valid_size to True for valid_data observations.
      - "topN": Keeps the top N stocks based on market cap (column 'me') within each date 'eom'.
      - "bottomN": Keeps the bottom N stocks.
      - "size_grp_<group>": Keeps stocks in a given size group, given by column 'size_grp'.
      - "perc_low<low>%high<high>%min<min_n>": A percentile-based screen.
    """
    count = 0  # Ensure that at least one screen is applied

    # --- "all" screen: no size screen is applied.
    if type_ == "all":
        print("No size screen")
        #Whenever 'valid_data' == True, 'valid_size' == True
        chars.loc[chars['valid_data'] == True, 'valid_size'] = True
        count += 1

    # --- "top" screen: e.g. "top1000"
    elif "top" in type_:
        # Extract the number after "top"
        top_n = int(type_.replace("top", ""))
        # For valid_data rows, rank stocks within each eom by descending market cap.
        # Using rank(method='first', ascending=False) gives lower rank numbers to higher values.
        chars.loc[chars['valid_data'], 'me_rank'] = chars.loc[chars['valid_data']].groupby('eom')['me']\
            .rank(method='first', ascending=False)
        # Mark valid_size as True if the rank is within top_n
        chars['valid_size'] = (chars['me_rank'] <= top_n) & (~chars['me_rank'].isna())
        # Remove the helper rank column
        chars.drop(columns=['me_rank'], inplace=True)
        count += 1

    # --- "bottom" screen: e.g. "bottom500"
    elif "bottom" in type_:
        bot_n = int(type_.replace("bottom", ""))
        # Rank stocks by ascending order of market cap within each date 'eom'
        chars.loc[chars['valid_data'], 'me_rank'] = chars.loc[chars['valid_data']].groupby('eom')['me']\
            .rank(method='first', ascending=True)
        chars['valid_size'] = (chars['me_rank'] <= bot_n) & (~chars['me_rank'].isna())
        chars.drop(columns=['me_rank'], inplace=True)
        count += 1

    # --- "size_grp_" screen: e.g. "size_grp_small"
    elif "size_grp_" in type_:
        size_grp_screen = type_.replace("size_grp_", "")
        chars['valid_size'] = (chars['size_grp'] == size_grp_screen) & (chars['valid_data'])
        count += 1

    # --- Percentile-based screen: type contains "perc"
    elif "perc" in type_:
        # Extract the lower and upper percentiles and a minimum count from the type string
        # For example: "perc_low20high80min50" would indicate low_p=20, high_p=80, min_n=50.
        low_p = int(re.search(r'(?<=low)\d+', type_).group(0))
        high_p = int(re.search(r'(?<=high)\d+', type_).group(0))
        min_n = int(re.search(r'(?<=min)\d+', type_).group(0))
        print("Percentile-based screening: Range {}% - {}%, min_n: {} stocks".format(low_p, high_p, min_n))

        # For each eom, compute the empirical cumulative distribution function (ECDF) of market cap 'me'
        def ecdf(s):
            # rank data in ascending order and then normalize by count (ecdf never returns 0)
            return s.rank(method='min', pct=True)
        chars.loc[chars['valid_data'], 'me_perc'] = chars.loc[chars['valid_data']].groupby('eom')['me']\
            .transform(ecdf)
        # Define valid_size as stocks with ecdf values strictly above low_p and at or below high_p
        chars['valid_size'] = (chars['me_perc'] > (low_p / 100)) & (chars['me_perc'] <= (high_p / 100))
        # Compute extra counts by group for adjustments
        grp = chars.groupby('eom')
        # Total stocks for which valid_data==True per date
        chars['n_tot'] = grp['valid_data'].transform('sum')
        # Count of stocks that pass the initial valid_size based on percentile
        chars['n_size'] = grp['valid_size'].transform('sum')
        # Count of stocks with percentile below the lower bound
        chars['n_less'] = grp.apply(lambda df: ((df['valid_data']) & (df['me_perc'] <= (low_p / 100))).sum()).reindex(chars.index, method='ffill')
        # Count of stocks with percentile above the upper bound
        chars['n_more'] = grp.apply(lambda df: ((df['valid_data']) & (df['me_perc'] > (high_p / 100))).sum()).reindex(chars.index, method='ffill')
        # How many stocks are missing from the window to reach min_n:
        chars['n_miss'] = (min_n - chars['n_size']).clip(lower=0)
        # Determine the number of extra stocks to add from below and above the percentile range
        chars['n_below'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_less'])).astype(int)
        chars['n_above'] = np.ceil(np.minimum(chars['n_miss'] / 2, chars['n_more'])).astype(int)
        # Adjust if the sum is less than n_miss by adding the remaining stocks
        # (This correction is applied groupwise in the original code; here we do it rowwise.)
        adjustment = (chars['n_below'] + chars['n_above'] < chars['n_miss'])
        # Adjust by favoring the side with more available stocks.
        chars.loc[adjustment & (chars['n_above'] > chars['n_below']), 'n_above'] += (chars['n_miss'] - chars['n_below'] - chars['n_above'])
        chars.loc[adjustment & (chars['n_above'] < chars['n_below']), 'n_below'] += (chars['n_miss'] - chars['n_below'] - chars['n_above'])
        # Now reassign valid_size with the adjusted bounds.
        chars['valid_size'] = (chars['me_perc'] > (low_p / 100 - chars['n_below'] / chars['n_tot'])) & \
                              (chars['me_perc'] <= (high_p / 100 + chars['n_above'] / chars['n_tot']))

        # Drop helper columns for percentile screening
        chars.drop(columns=['me_perc', 'n_tot', 'n_size', 'n_less', 'n_more', 'n_miss', 'n_below', 'n_above'], inplace=True)
        count += 1

    else:
        raise ValueError("Size screen type not recognized: {}".format(type_))

    if count != 1:
        # Ensure that only one type of screening is applied.
        raise ValueError("Invalid size screen applied!!!!")


def investment_universe(add, delete):
    """
    Computes the inclusion status of a stock in the investment universe based on 
    addition and deletion signals.

    A stock is included in the universe starting from the point where an 'add' signal is received,
    and remains included until a 'delete' signal is triggered. This logic enforces persistence 
    in the investment universe by smoothing transient signals.

    Parameters
    ----------
    add : array-like of bool
        Boolean array indicating when the stock becomes eligible for inclusion 
        (e.g., has met criteria for a defined number of months).
    delete : array-like of bool
        Boolean array indicating when the stock should be excluded due to failing 
        to meet criteria over a defined period.

    Returns
    -------
    np.ndarray of bool
        Boolean array where True indicates that the stock is considered part of 
        the investment universe at that time step.
    """
    add = np.asarray(add, dtype=bool)
    delete = np.asarray(delete, dtype=bool)
    n = len(add)
    included = np.zeros(n, dtype=bool)
    
    if n < 2:
        return included  # Nothing to include for short inputs
    
    state = False
    # Vectorized logic using while-loop mimicry (manual loop still faster than pure loop over full array)
    for i in range(1, n):
        if not state and add[i] and not add[i - 1]:
            state = True
        elif state and delete[i]:
            state = False
        included[i] = state

    return included

def addition_deletion_fun(chars, addition_n, deletion_n):
    """
    Applies a 12-month addition and deletion rule to determine stock inclusion in an investment universe.

    Stocks are added to the universe if they have met specified validity criteria (e.g., data availability and size) 
    for 'addition_n' consecutive months, and removed if they have failed to meet these criteria for 'deletion_n' consecutive months.
    
    The function uses rolling windows to flag additions and deletions, applies a user-defined universe logic via 
    the `investment_universe` function, and calculates turnover statistics with and without the smoothing effect 
    of the addition/deletion rules.

    Parameters:
    ----------
    chars : pd.DataFrame
        Input DataFrame containing columns:
            - 'id': Security identifier
            - 'eom': End-of-month date
            - 'valid_data': Boolean indicator if data is valid
            - 'valid_size': Boolean indicator if size restriction is met
    addition_n : int
        Number of consecutive months required to add a stock to the universe (typically 12).
    deletion_n : int
        Number of consecutive months of failing restrictions to remove a stock from the universe (typically 12).

    Returns:
    -------
    chars: Modified DataFrame with updated 'valid' status and cleaned of temporary columns. 
            Prints turnover statistics before and after applying the addition/deletion rules.
    """
    # Create valid_temp: valid data and valid size
    chars['valid_temp'] = (chars['valid_data'] & chars['valid_size'])
    
    # Sort the DataFrame by 'id' and 'eom'
    chars.sort_values(by=['id', 'eom'], inplace=True)
    
    # Set index to allow for groupby+rolling
    chars = chars.sort_values(by=['id', 'eom']).set_index('id')
    
    # rolling sum with groupby.rolling()
    chars['addition_count'] = (
        chars['valid_temp']
        .groupby(level=0)
        .rolling(window=addition_n, min_periods=addition_n)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    chars['deletion_count'] = (
        chars['valid_temp']
        .groupby(level=0)
        .rolling(window=deletion_n, min_periods=deletion_n)
        .sum()
        .reset_index(level=0, drop=True)
    )
    
    # Restore index sort
    chars = chars.reset_index()

    # Create logical columns 'add' and 'delete'
    # For add: True if the rolling sum equals addition_n (i.e., all values in the window were True)
    # For delete: True if the rolling sum equals 0 (i.e., all values in the window were False)
    chars['add'] = chars['addition_count'] == addition_n
    chars['add'].fillna(False, inplace=True)
    chars['delete'] = chars['deletion_count'] == 0
    chars['delete'].fillna(False, inplace=True)
    
    # Count number of rows per group (by id)
    chars['n'] = chars.groupby('id')['id'].transform('count')
    
    # Apply the investment_universe function for groups with more than one observation
    # We use groupby and then assign the new 'valid' column.
    def process_group_fast(group):
        if len(group) == 1:
            group['valid'] = False
        else:
            group['valid'] = investment_universe(group['add'].to_numpy(), group['delete'].to_numpy())
        return group
    
    groups = [g for _, g in chars.groupby('id')]
    
    for g in groups:
        process_group_fast(g)
    
    results = [process_group_fast(g) for g in groups]
    chars = pd.concat(results, ignore_index=True)
    
    # Ensure that if valid_data is False, valid is also False
    chars.loc[chars['valid_data'] == False, 'valid'] = False
    
    # Check turnover:
    # chg_raw is True when valid_temp changes compared to previous row (per id)
    # chg_adj is True when valid changes compared to previous row (per id)
    chars['valid_temp_shifted'] = chars.groupby('id')['valid_temp'].shift(1)
    chars['chg_raw'] = (chars['valid_temp'] != chars['valid_temp_shifted']).astype(float)
    chars.loc[chars['valid_temp'].isna() | chars['valid_temp_shifted'].isna(), 'chg_raw'] = np.nan

    # Pre-shift the 'valid' column by group
    chars['valid_shifted'] = chars.groupby('id')['valid'].shift(1)
    
    # Vectorized difference check
    chars['chg_adj'] = np.where(
        chars['valid'].isna() | chars['valid_shifted'].isna(),
        np.nan,
        chars['valid'] != chars['valid_shifted']
    )
    
    # Compute aggregated turnover statistics by end of month (eom)
    # STEP 1: Pre-filter out rows where both chg_* and valid_* are NaN (optional but faster)
    valid_raw_mask = chars['chg_raw'].notna() & chars['valid_temp'].notna()
    valid_adj_mask = chars['chg_adj'].notna() & chars['valid'].notna()
    
    # STEP 2: Create condensed dataframe for aggregation
    agg_df = pd.DataFrame({
        'eom': chars['eom'],
        'chg_raw': np.where(valid_raw_mask, chars['chg_raw'], 0.0),
        'chg_adj': np.where(valid_adj_mask, chars['chg_adj'], 0.0),
        'valid_temp': np.where(valid_raw_mask, chars['valid_temp'], 0.0),
        'valid': np.where(valid_adj_mask, chars['valid'], 0.0),
    })
    
    # STEP 3: Group and sum — ultra fast
    sum_df = agg_df.groupby('eom', sort=False).sum().reset_index()
    
    # STEP 4: Compute ratios
    sum_df['raw'] = sum_df['chg_raw'] / sum_df['valid_temp']
    sum_df['adj'] = sum_df['chg_adj'] / sum_df['valid']
    
    # STEP 5: Filter invalids (e.g., division by 0)
    agg = sum_df[(sum_df['adj'].notna()) & (sum_df['adj'] != 0)]
    
    # STEP 6: Compute overall metrics
    overall = {
        'n_months': len(agg),
        'n_raw': agg['valid_temp'].mean(),
        'n_adj': agg['valid'].mean(),
        'turnover_raw': agg['raw'].mean(),
        'turnover_adjusted': agg['adj'].mean()
    }
    
    # Print the turnover values (in percentages)
    print("Turnover wo addition/deletion rule: " + str(round(overall['turnover_raw'] * 100, 2)) + "%")
    print("Turnover w  addition/deletion rule: " + str(round(overall['turnover_adjusted'] * 100, 2)) + "%")
    
    # Clean up: Drop columns that are not needed any more
    chars.drop(columns=["n", "addition_count", "deletion_count", "add",
                        "delete", "valid_temp", "valid_data", "valid_size",
                        "chg_raw", "chg_adj",
                        'valid_temp_shifted', 'valid_shifted'], inplace=True)
    
    return chars

# Define a function that transforms a Series using its ECDF
def ecdf_transform(group):
    # Drop NaN values before computing ECDF
    non_na = group.dropna()
    # If there is no non-NaN data, return the group unchanged
    if non_na.empty:
        return group
    # Create the ECDF function from the non-NaN data
    ecdf_func = ECDF(non_na)
    # Apply the ecdf function to each element; keep NaNs unchanged
    return group.apply(lambda x: ecdf_func(x) if pd.notna(x) else np.nan)


# Compute Cluster Ranks
def build_cluster_ranks(cluster_data_m, cluster_labels, clusters, features):
    
    # Initialize an empty DataFrame to hold the cluster rank columns
    cluster_ranks = pd.DataFrame(index=cluster_data_m.index)
    
    #Iterate over each cluster
    for cl in clusters:
        # Extract Characteristics belonging to this Cluster
        chars_sub = cluster_labels[(cluster_labels['cluster'] == cl) & 
                                   (cluster_labels['characteristic'].isin(features))]
        
        # Get Data of Characteristics belonging to the cluster
        data_sub = cluster_data_m[chars_sub['characteristic'].tolist()].copy()
        
        # Apply direction transformation
        for c in chars_sub['characteristic']:
            #Get Direction
            dir_val = chars_sub.loc[chars_sub['characteristic'] == c, 'direction'].values[0]
            if dir_val == -1:
                data_sub[c] = 1 - data_sub[c]
        
        # Compute row means (each row is a stock at a time point t) 
        #   and assign to a new column named after the cluster
        cluster_ranks[cl] = data_sub.mean(axis=1)
                
    return cluster_ranks

#=============================================================
# Weighted Covariance Matrix
#------------------------------------------------------------
def weighted_cov_wt(df, weights):
    """
    Compute the weighted covariance matrix just like R's cov.wt(..., center=TRUE, method="unbiased").

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of shape (n, p) with numeric data (excluding 'date' or other non-numeric columns).
    weights : array-like
        Length-n vector of weights (e.g. from tail(w_cor, t)). Will be normalized.

    Returns
    -------
    cov_df : pandas.DataFrame
        Weighted covariance matrix as a DataFrame (p x p) with variable names.
    """
    # Convert to NumPy
    X = df.to_numpy()
    w = np.asarray(weights, dtype=float)
    
    # Normalize weights
    w_norm = w / w.sum()
    
    # Weighted mean
    mu = np.average(X, axis=0, weights=w_norm)   
    
    # Centered data
    Xc = X - mu
    
    # Apply sqrt(weights) to each row
    Xw = Xc * np.sqrt(w_norm)[:, np.newaxis]
    
    # Unbiased weighted covariance
    denom = 1.0 - np.sum(w_norm ** 2)
    cov_matrix = (Xw.T @ Xw) / denom  # shape (p, p)

    # Return as DataFrame with labels
    cov_df = pd.DataFrame(cov_matrix, index=df.columns, columns=df.columns)
    
    return cov_df

def weighted_cor_wt(df, weights):
    """
    Compute the weighted correlation matrix like R's cov.wt(..., cor=TRUE, center=TRUE, method="unbiased").

    Parameters
    ----------
    df : pandas.DataFrame
        Numeric data (n x p), excluding any non-numeric columns like 'date'.
    weights : array-like
        Length-n vector of weights (will be normalized).

    Returns
    -------
    cor_df : pandas.DataFrame
        Weighted correlation matrix (p x p) with variable names.
    """
    # Convert inputs
    X = df.to_numpy()
    w = np.asarray(weights, dtype=float)
    
    # Normalize weights
    w_norm = w / w.sum()
    
    # Weighted mean
    mu = np.average(X, axis=0, weights=w_norm)
    
    # Centered and weighted data
    Xc = X - mu
    Xw = Xc * np.sqrt(w_norm)[:, np.newaxis]
    
    # Weighted covariance
    denom = 1.0 - np.sum(w_norm ** 2)
    cov_matrix = (Xw.T @ Xw) / denom

    # Standard deviations
    std_dev = np.sqrt(np.diag(cov_matrix))

    # Outer product of std_dev for normalization
    outer_std = np.outer(std_dev, std_dev)
    
    # Correlation matrix
    cor_matrix = cov_matrix / outer_std
    
    # Ensure diagonals are exactly 1.0 (numerical stability)
    np.fill_diagonal(cor_matrix, 1.0)
    
    # Return as labeled DataFrame
    cor_df = pd.DataFrame(cor_matrix, index=df.columns, columns=df.columns)
    
    return cor_df

def pfml_feat_fun(p):
    """
    Extract the subset of RFFs given the hyperparameter p (# of RFFs)
    """
    feat = ['constant'] \
        + ["rff" + str(x) + "_cos" for x in range(1,p//2+1)] \
            + ["rff" + str(x) + "_sin" for x in range(1,p//2+1)]
    return feat


def create_cov(x, ids=None):
    """
    Create the Barra Covariance Matrix 
    
    Parameters:
    x (dict): Dictionary containing 'fct_load', 'ivol_vec', and 'fct_cov'
    ids (array-like, optional): List of Stock IDs to subset the data
    
    Returns:
    numpy.ndarray: The computed covariance matrix
    """
    ################## Compute the Covariance Matrix ##########################
    # Extract the relevant loadings and ivol
    if ids is None:
        load = x['fct_load']
        ivol = x['ivol_vec']
    else:
        # Convert ids to strings to match R's behavior with as.character()
        load = x['fct_load'].loc[[i for i in ids]]
        ivol = x['ivol_vec'].loc[[i for i in ids]]
    
    # Create the covariance matrix
    sigma = load @ x['fct_cov'] @ load.T + np.diag(ivol) 
    
    ################## Error Correction ##########################
    """
    In case a stock variance is negative. Can happen as (37) is not exactly an
    equality. Due to the EWMA the residuals are not uncorrelated with the fitted values.
    """
    if min(np.diag(sigma)) < 0:
        # Get diagonal indices
        diag_indices = np.arange(len(sigma))
        print("Warning: Negative Variances:", diag_indices)
        
        """
        # Extract the diagonal
        diag_values = sigma.values[diag_indices, diag_indices]
        
        # Find the minimum of the positive diagonal values and use them to replac ethe negatives
        positive_diag = diag_values[diag_values > 0]
        if len(positive_diag) > 0:
            min_positive = positive_diag.min()
        
            # Create a boolean mask where diagonal elements are zero
            mask = diag_values < 0
        
            # Replace zero diagonal elements with min_positive
            sigma.values[diag_indices[mask], diag_indices[mask]] = min_positive
        """
            
    return sigma


def create_lambda(x, ids):
    """
    Create Kyle's Lambda (Diagonal Matrix)
    
    Parameters:
    x: array of each stock's lambda
    
    Returns:
    numpy.ndarray: Diagonal matrix with selected elements
    """
    if isinstance(x, dict):
        selected = [x[i] for i in ids]
    else:
        # If x is array-like, use direct indexing
        selected = x[ids]
    
    return np.diag(selected)


def m_func(w, mu, rf, sigma_gam, gam, K_Lambda, iterations):
    """
    Computes m_tilde from (14) and then m from Lemma 1.
    
    Parameters:
    w (float): Weight parameter
    mu (float): Expected return
    rf (float): Risk-free rate
    sigma_gam (numpy.ndarray): sigma (Barra Cov) * gamma_rel (RA parameter gamma)
    gam (float): Risk aversion parameter gamma
    K_Lambda (numpy.ndarray): Kyle's Lambda (Diagonal Matrix)
    iterations (int): Number of iterations to iterate on (B.19)
    
    Returns:
    numpy.ndarray: The computed matrix m
    """
    
    # Ensure all inputs are numpy arrays
    sigma_gam = np.asarray(sigma_gam)
    K_Lambda = np.asarray(K_Lambda)
    
    n = K_Lambda.shape[0]
    g_bar = np.ones(n)
    mu_bar_vec = np.full(n, 1 + rf + mu)
    
    sigma_gr = (1 / (1 + rf + mu)**2) * (np.outer(mu_bar_vec, mu_bar_vec) + sigma_gam / gam)
    
    
    # Compute K_Lambda_neg05 (diagonal matrix with elements lambda^(-0.5))
    K_Lambda_neg05 = np.diag(np.diag(K_Lambda) ** (-0.5) )
    
    # Compute x and y
    x = (w ** -1) * K_Lambda_neg05 @ sigma_gam @ K_Lambda_neg05
    y = np.diag(1 + np.diag(sigma_gr))
    
    # Initialize sigma_hat and m_tilde
    sigma_hat = x + np.diag(1 + g_bar)
    m_tilde = 0.5 * (sigma_hat - np.real(sqrtm(sigma_hat @ sigma_hat - 4 * np.eye(n))))
    
    # Iterate on F
    for i in range(iterations):
        m_tilde = np.linalg.inv(x + y - m_tilde * sigma_gr)
    
    # Final output
    return K_Lambda_neg05 @ m_tilde @ np.sqrt(K_Lambda)