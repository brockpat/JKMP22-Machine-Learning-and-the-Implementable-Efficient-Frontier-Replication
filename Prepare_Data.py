# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:37:41 2025

@author: Patrick
"""

"""
This file prepares and processes the characteristics data for U.S. CRSP stocks 
It performs the following key tasks:

1. **Setup and Initialization**: Loads feature definitions and global settings used across the script.
2. **Data Acquisition**: Connects to local databases and reads in data on stock characteristics (JKP factors), market returns, and the risk-free rate.
3. **Data Enrichment**: Enhances the dataset with factor labels, directionality, and derived metrics such as Kyle’s Lambda and scaled volatility.
4. **Return Construction**: Computes lead and realized total returns.
5. **Wealth Evolution**: Calculates wealth accumulation over time given total returns and a chosen benchmark date.
6. **Screening**: Applies multiple data filters to ensure quality and consistency, such as exchange listing, valid return data, liquidity, and completeness of features.
7. **Feature Processing**: Standardizes feature values via percentile ranking in the cross-section and imputes missing data to ensure completeness.
8. **Industry Classification**: Adds industry classification Strings based on SIC codes.
9. **Investment Universe Selection**: Flags valid and investable stocks using lookback logic, size-based screens, and addition/deletion rules.
10. **Diagnostics and Output**: Provides summary statistics and plots to evaluate data validity and the evolution of the investable universe over time.

The output is a cleaned and enriched DataFrame (`chars`) ready for downstream analysis or portfolio modeling.
"""

#%% Libraries

#Run specific functions
from General_functions import *

#DataFrame Libraries
import pandas as pd
import sqlite3
from pandas.tseries.offsets import MonthEnd

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout
settings, pf_set = get_settings()

#%% Connect to DataBases
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")
crsp_daily = sqlite3.connect(database = path + "Data/crsp_daily_SP500.db")

#%% Risk Free Rate
"""
Load Risk-Free Rate Data from Kenneth French's Website.
"""
# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])

# Convert to decimal (RF is given in percentage terms)
risk_free["rf"] = risk_free["RF"] / 100

#--- Construct an end-of-month date.
risk_free["eom"] = pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

# Keep only the required columns.
risk_free = risk_free[["eom", "rf"]]

print("Risk-free Rate Data Complete.")
#%% Market Returns 
"""
Load Market Return Data

Data Source: #https://www.dropbox.com/scl/fo/zxha6i1zcjzx8a3mb2372/AN9kkos5H5UjjXUOqW3EuDs?rlkey=i3wkvrjbadft6hld863571dol&e=1&dl=0
taken from: https://github.com/theisij/ml-and-the-implementable-efficient-frontier/tree/main?tab=readme-ov-file
"""
# Read in market returns data.
market = (pd.read_csv(path + "Data/market_returns.csv", dtype={"eom": str})
          #Only get US
          .pipe(lambda df: df[df['excntry'] == 'USA'])
          #Format Date
          .assign(eom_ret = lambda df: pd.to_datetime(df["eom"], format="%Y-%m-%d"))
          #Get relevant columns
          .get(["eom_ret", "mkt_vw_exc"])
          )
print("Market Data Complete.")

#%% Factor Labels
"""
Load in the Factor labels, i.e. the cluster to which each JKP Factor belongs to.

Create a DataFrame containing the name of the characteristic, its cluster
and its direction.

Data obtained from: https://github.com/theisij/GlobalFactor/tree/master/GlobalFactors
"""
# Load cluster labels data.
cluster_labels = pd.read_csv(path + "Data/Cluster Labels.csv")

# Variable Name Editting: lower case and replace spaces/hyphens with underscores.
cluster_labels["cluster"] = cluster_labels["cluster"].str.lower().str.replace(r"[\s-]", "_", regex=True)

# Load factor signs (factor details) from data.
factor_signs = pd.read_excel(path + "Data/Factor Details.xlsx")

# Select and filter columns:
#   rename 'abr_jkp' to 'characteristic' (abr_jkp are the abbreviations JKP use)
factor_signs = factor_signs[["abr_jkp", "direction"]].dropna(subset=["abr_jkp"]).copy()
#   filter rows where characteristic is not missing.
factor_signs = factor_signs.rename(columns={"abr_jkp": "characteristic"})

# Ensure the 'direction' column is numeric.
factor_signs["direction"] = pd.to_numeric(factor_signs["direction"], errors="coerce")

# Merge factor_signs with cluster_labels
cluster_labels = pd.merge(factor_signs, cluster_labels, on="characteristic", how="right")

# Manually append a row to assign the 'rvol_252d' characteristic to a specific cluster.
extra_row = pd.DataFrame({
    "characteristic": ["rvol_252d"],
    "direction": [-1],
    "cluster": ["low_risk"]
})
cluster_labels = pd.concat([cluster_labels, extra_row], ignore_index=True)

print("Factor Labels Complete.")

#%% Characteristics Data
"""
Read in monthly data of Stock Characteristics.
Convert numerical features to numerical type.
Compute Kyle's lambda for each stock.
Scale monthly return volatility for each stock.
"""
#Select Time Frame (Chunk) to Read in
#start_date = "1952-01-01"
#end_date = "2024-12-31"
#print(f"Reading in Chars Data. Date Range: {start_date} to {end_date}")
print("Reading in Chars Data (Full Date Range)")

#Build query vector for features:
#   Set Empty String
query_features =""
#   Fill the Empty String with the features
for feature in features:
    query_features = query_features + feature + ", "
#   Delete last comma and space to avoid syntax errors
query_features = query_features[:-2]
#   Build final query vector
query = ("SELECT id, eom, sic, ff49, size_grp, me, crsp_exchcd, ret_exc, "
         +query_features 
         +" FROM Factors " 
         #+f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
         #+"AND CAST(id AS INTEGER) <= 99999" #Filter for CRSP observations (id <= 99999)
         # Not required as I only have CRSP Data in my DataSet anyway
         )

# Read in JKP characteristics data.
chars  = pd.read_sql_query(query, 
                           con=JKP_Factors,
                           parse_dates={'eom'}
                           )

#Convert features to numeric
for feature in features:
    chars[feature] = pd.to_numeric(chars[feature], errors = 'coerce')
#Convert id to int
chars['id'] = chars['id'].astype('int64')
#Convert sic to integer
chars['sic'] = pd.to_numeric(chars['sic'], errors = 'coerce', downcast='integer')

#Add additional columns:
#   dollar volume
chars["dolvol"] = chars["dolvol_126d"]
#   Kyle's Lambda (pi = price impact)
chars["lambda"] = 2 / chars["dolvol"] * settings["pi"]
#   Monthly version of rvol by annualizing daily volatility.
chars["rvol_m"] = chars["rvol_252d"] * (21 ** 0.5)

print("Chars Data Complete")
#%% Monthly Returns
"""
Get leaded monthly excess returns (ret_ldx).
Construct date eom_ret which is the 1 month leaded date eom.
Compute total return, i.e. the return of the asset (tr_ld1).

Merge Columns to chars DataFrame
"""
#Extract Monthly excess returns
monthly = chars[['id', 'eom', 'ret_exc']]
monthly.loc[:,'ret_exc'] = pd.to_numeric(monthly['ret_exc'], errors = 'coerce')
monthly = monthly.dropna()

#Lead Returns: ret_ld1t = ret_exc(t+1)
#   If lead return doesn't exist (Time Series exhausted), lead return is set to 0
data_ret_ld1 = long_horizon_ret(monthly, h=settings['pf']['hps']['m1']['K'])[['id', 'eom', 'ret_ld1']]

#'eom_ret' leads eom. Thus, 'ret_ld1' is the realized return at date eom_ret and 1 month leaded return at date eom
data_ret_ld1['eom_ret'] = data_ret_ld1['eom'] + MonthEnd(1)

#Merge with risk_free data on 'eom' column
data_ret_ld1 = pd.merge(risk_free, data_ret_ld1, on='eom', how='right')

#Calculate total return (tr_ld1 = ret_ld1 + rf)
data_ret_ld1['tr_ld1'] = data_ret_ld1['ret_ld1'] + data_ret_ld1['rf']

#Drop the rf column
data_ret_ld1 = data_ret_ld1.drop(columns=['rf'])

#Create DataFrame
data_ret_ld1 = data_ret_ld1.merge(
    data_ret_ld1[['id', 'eom', 'tr_ld1']]
    .rename(columns={'tr_ld1': 'tr_ld0'})
    #Date = Last Day of the Next Month
    .assign(eom=lambda df: df['eom'] + MonthEnd(1)),
    on=['id', 'eom'],
    how='left'
    #At date eom_ret: 'tr_ld1' is the realized total return
    #At date eom: 'tr_ld1' is the lead total return, 'tr_ld0' is the realized total return
)[['id','eom','tr_ld0','eom_ret','ret_ld1','tr_ld1']]

#Save Memory
del monthly

# Merge chars with total return data computed above.
chars = pd.merge(chars, data_ret_ld1, on=["id", "eom"], how="left")

#Save Memory
del data_ret_ld1

print("Leading Returns Complete")
#%% Wealth (AUM) Evolution
"""
For a given initial wealth level at time settings['split']['test_end'], compute
the past wealth levels if they had evolved exactly according to the realized
total return.

Merge Wealth to chars
"""
#Get the Wealth Evolution 
wealth = wealth_func(pf_set['wealth'], settings['split']['test_end'], market, risk_free)

#Merge with chars
wealth_shifted = wealth.copy()
wealth_shifted["eom"] = wealth_shifted["eom"] + pd.offsets.MonthEnd(1)
#Since date is shifted 1 month ahead, lead mu is now contemporary mu, so it is renamed to mu_ld0
wealth_shifted = wealth_shifted[["eom", "mu_ld1"]].rename(columns={"mu_ld1": "mu_ld0"})

# Step 2: Left join with chars on 'eom'
chars = pd.merge(chars, wealth_shifted, on="eom", how="left")
del wealth_shifted

print("Wealth Evolution Complete.")
#%% Data Screens
"""
Remove unwanted observations depending on which screening parameters were chosen.
"""

#==================================================
#              Screens (Data Removal)
#--------------------------------------------------

# 1. Screen by Exchange code if NYSE only is desired.
if settings["screens"]["nyse_stocks"]:
    #Variable EXCHCD in CRSP: 1 = NYSE, 2 = AMEC, 3 = NASDAQ
    pct_excluded = (chars["crsp_exchcd"] != 1).mean() * 100
    print(f"   NYSE stock screen excludes {pct_excluded:.2f}% of the observations")
    chars = chars[chars["crsp_exchcd"] == 1]

#Save key metrics before screening which clears out Data
n_start = len(chars)
me_start= np.sum(chars['me'].dropna())

# 2. Date screen.
pct_date_excluded = ((chars["eom"] < settings["screens"]["start"]) | (chars["eom"] > settings["screens"]["end"])).mean() * 100
print(f"   Date screen excludes {pct_date_excluded:.2f}% of the observations")
chars = chars[(chars["eom"] >= settings["screens"]["start"]) & (chars["eom"] <= settings["screens"]["end"])]


# 3. Require non-missing market equity 'me'.
pct_me_missing = chars["me"].isna().mean() * 100
print(f"   Non-missing me excludes {pct_me_missing:.2f}% of the observations")
chars = chars[~chars["me"].isna()]

# 4. Require non-missing returns at t and t+1.
pct_return_invalid = (chars["tr_ld1"].isna() | chars["tr_ld0"].isna()).mean() * 100
print(f"   Valid return req excludes {pct_return_invalid:.2f}% of the observations")
chars = chars[(~chars["tr_ld1"].isna()) & (~chars["tr_ld0"].isna())]

# 5. Require non-missing and non-zero dollar volume.
pct_dolvol_invalid = ((chars["dolvol"].isna()) | (chars["dolvol"] == 0)).mean() * 100
print(f"   Non-missing/non-zero dolvol excludes {pct_dolvol_invalid:.2f}% of the observations")
chars = chars[(~chars["dolvol"].isna()) & (chars["dolvol"] > 0)]

# 6. Require valid SIC code.
pct_sic_invalid = (chars["sic"].isna()).mean() * 100
print(f"   Valid SIC code excludes {pct_sic_invalid:.2f}% of the observations")
chars = chars[~chars["sic"].isna()]

# 7. Feature screens: count the number of non-missing features.
feat_available = chars[features].notna().sum(axis=1)
min_feat = np.floor(len(features) * settings['screens']['feat_pct']) #Minimum features we want per row
print(f"   At least {settings['screens']['feat_pct'] * 100}% of feature excludes {round((feat_available < min_feat).mean() * 100, 2)}% of the observations")
chars = chars[feat_available >= min_feat] #Kick out observations with too many missing values
print(f"In total, the final dataset has {round((len(chars) / n_start) * 100, 2)}% of the observations and {round((chars['me'].sum() / me_start) * 100, 2)}% of the market cap in the post {settings['screens']['start']} data")

#Save Memory
del feat_available

#%% Feature Ranks and Imputation
"""
Computes the Rank of each feature for each stock in the cross section. The stock's
rank of the feature is based on the cross-sectional empirical CDF of that feature.

Next, missing values are imputed.

Original feature values are overwritten
"""
# Compute Feature Rank
if settings["feat_prank"]: #Feature Ranking by percentile ranking
    """
    Performs feature standardization by converting features to percentile ranks within each time period.
    
    For each feature:
    1. Converts the feature values to float type for precision
    2. Calculates empirical CDF (percentile ranks) within each 'eom' group, 
       so values are ranked relative to others in the same time period
    3. Preserves exact zeros by setting them back to 0 after ranking
       (since ECDF would otherwise give them small positive values)
    
    This normalization makes features comparable across different scales and distributions.
    Operates in-place on the DataFrame.
    """    
    ranked = chars.groupby("eom")[features].rank(pct=True) #rank is faster than using custom ecd_transform() and gives the same result
    
    # Restore zeros
    mask = chars[features] == 0.0
    ranked[mask] = 0
    
    #Overwrite feature data
    chars[features] = ranked
    
    #Save Workspace
    del ranked, mask
    
    print("Feature Rank Complete.")

# Impute Missing Features
if settings['feat_impute']:
    """
    Handles missing values (NA) in features using different strategies based on settings.
    
    If percentile ranking was performed:
    - Imputes NAs with 0.5 (midpoint of [0,1] range), preserving the percentile rank interpretation
    
    Otherwise:
    - Imputes NAs with the median value within each 'eom' group (time period), 
      maintaining the original distribution's central tendency
    
    This ensures no missing values remain in the feature columns.
    Operates in-place on the DataFrame.
    """
    if settings['feat_prank']:
        chars[features] = chars[features].apply(lambda x: x.fillna(0.5))
    else:
        # Median imputation by 'eom' group
        for f in features:
            chars[f] = chars.groupby('eom')[f].transform(lambda x: x.fillna(x.median()))
            
    print("Feature Imputation Complete.")

#%% Industry Classification

# Generate Industry Classification String
chars["ff12"] = chars["sic"].apply(categorize_sic).astype(str)
print("Industry Dummy Creation Complete.")

#%% Validity Screening and Lookback Checks
"""
Valid Observation Screening
– All rows are initially marked as valid (i.e. valid_data = True).
– The DataFrame is sorted by id and eom to ensure proper group ordering.
– For each group (by id), the eom column is shifted by a lookback period (lb).
– The month difference between the shifted date and the current date is calculated, and only rows with the correct lookback (month_diff == lb) are kept.

Size-Based Screening (size_screen_fun)
– Depending on the type string, the function applies one of several screens:
• “all”: No screening, all valid data is marked valid_size.
• “topN” or “bottomN”: Ranks the stocks by market cap and selects the top or bottom N stocks per period.
• “size_grp_”: Filters by a specified size group.
• “perc”: Uses an empirical cumulative distribution function (ECDF) to compute size percentiles and then adjusts the screening window if too few stocks are selected.

Addition/Deletion Rule (addition_deletion_fun)
– A temporary flag (valid_temp) is created from the valid_data and valid_size columns.
– For each stock (grouped by id), rolling window counts are calculated over the flag for addition (addition_n) and deletion (deletion_n).
– Based on these counts, flags add and delete are set. For groups with more than one observation, a custom function (investment_universe) is applied to determine the final investable universe.
– Finally, turnover metrics based on the raw and adjusted changes are calculated and printed.

Visualization and Summary
– A scatter plot shows the number of investable stocks (where valid is True) over time (eom).
– A final printout gives a summary of the proportion of valid observations and their market capitalization share.
"""

# =============================================================================
# 1. Check which observations are valid
# -----------------------------------------------------------------------------
# Add a default True 'valid_data' column
chars['valid_data'] = True

# Sort the DataFrame by 'id' and 'eom'
chars.sort_values(by=['id', 'eom'], inplace=True)

# Determine the lookback period
lb = pf_set['lb_hor'] + 1  # Plus 1 to get the last signal of last periods portfolio for portfolio-ML

# For each group defined by 'id', shift 'eom' by lb periods
chars['eom_lag'] = chars.groupby('id')['eom'].shift(lb)

# Calculate the difference in months between 'eom' and the lagged 'eom_lag'
# Here the difference is computed in years multiplied by 12 plus the difference in months.
#   Slower Code: chars['month_diff'] = chars.apply(lambda row: (row['eom'].year - row['eom_lag'].year) * 12 + (row['eom'].month - row['eom_lag'].month), axis=1)
eom = chars['eom'].dt
eom_lag = chars['eom_lag'].dt
chars['month_diff'] = (eom.year - eom_lag.year) * 12 + (eom.month - eom_lag.month)

# Compute the percentage of valid observations that have an improper lookback period.
#   (These are the rows for which month_diff is not equal to lb or is missing)
mask = (chars.loc[chars['valid_data'], 'month_diff'] != lb) | (chars.loc[chars['valid_data'], 'month_diff'].isna())
invalid_pct = round(mask.mean() * 100, 2)
print("   Valid lookback observation screen excludes {}% of the observations".format(invalid_pct))

# Reassign the valid_data flag: keep rows only if month_diff equals lb and is not missing.
chars['valid_data'] = (chars['valid_data'] & (chars['month_diff'] == lb) & (~chars['month_diff'].isna()))

# Remove helper columns 'eom_lag' and 'month_diff'
chars.drop(columns=['eom_lag', 'month_diff'], inplace=True)
del lb   # remove the lb variable


# =============================================================================
# 2. Apply the screens and show output 
# -----------------------------------------------------------------------------
# Apply the size-based screen using the settings provided by settings.screens['size_screen']
# (Assuming "settings" is a dictionary that contains the screen type)
settings['screens']['size_screen'] = 'all' #Nowhere defined in R code
size_screen_fun(chars, type_=settings['screens']['size_screen'])

# Apply the addition/deletion rule with provided parameters
chars = addition_deletion_fun(chars, addition_n=settings['addition_n'], deletion_n=settings['deletion_n'])

# -----------------------------------------------------------------------------
# 3. Plot the investable universe: count of valid stocks by 'eom'
# -----------------------------------------------------------------------------
# Compute the count of stocks where the 'valid' flag is True, grouped by eom.
valid_counts = chars.loc[chars['valid'] == True].groupby('eom').size().reset_index(name='N')

# Generate a scatter plot of number of valid stocks over time.
plt.figure(figsize=(10, 6))
plt.scatter(valid_counts['eom'], valid_counts['N'])
plt.xlabel("eom")
plt.ylabel("Valid stocks")
plt.axhline(y=0, color='grey', linestyle='--')
plt.title("Investable Universe Over Time")
plt.show()

# -----------------------------------------------------------------------------
# 4. Print valid summary: Percentage of valid observations and market cap share
# -----------------------------------------------------------------------------
# Calculate percentage of rows where valid==True
valid_pct = round(chars['valid'].mean() * 100, 2)
# Calculate percentage of market cap (assuming 'me' column represents market cap)
market_cap_pct = round((chars.loc[chars['valid'] == True, 'me'].sum() / chars['me'].sum()) * 100, 2)
print("   The valid_data subset has {}% of the observations and {}% of the market cap".format(valid_pct, market_cap_pct))

#%% Save Objects

#Wealth
wealth.to_csv(path + "Data/wealth_processed.csv", index = False)

#Cluster Labels
cluster_labels.to_csv(path + "Data/cluster_labels_processed.csv", index = False)

#Chars
chars.to_sql(name="Factors_processed", con=JKP_Factors, if_exists="replace", index=False)
JKP_Factors.close()
