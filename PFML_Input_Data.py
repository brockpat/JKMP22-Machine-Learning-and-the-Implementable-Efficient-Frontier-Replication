"""
This file mimics the function pfml_input_fun() from the original R-Code of JKMP22.

This file computes the required inputs for the Machine Learning Algorithm and
exports them. In particular, the data to compute (25) and thus obtain the optimal
beta coefficients from (26) is prepared here. Notice that in order to compute (24),
theta is truncated and runs from 0 to 11. 

Hence, for each month (time period t), the code computes the quantities 
i) 'r_tilde': \tilde{r}_{t+1}, 
ii) 'risk': gamma * \tilde{s}_t' \Sigma_t \tilde{s}_t,
iii) 'tc': \tilde{\Sigma}_t - 'risk',
iv) 'denom': \tilde{Sigma}_t,
v) 'signal_t': Diag(1/sigma)RFF(s) as in (40).

There is one file for each hyperparameter eta, which are stored 
in the vector g_vec, since the RFFs depend on the hyperparameter eta which in turn
through (24) will affect (25) and thus (26).

"""
#%% Libraries

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

import copy
import random
from scipy.stats import multivariate_normal

from tqdm import tqdm
import pickle
import sqlite3

#Run specific functions
from General_functions import *

#%% Connect to DataBase
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")

#%% Get Preliminaries
#List of Stock Features
features = get_features(exclude_poor_coverage = True)

#Settings used throughout
settings, pf_set = get_settings()

#%% Read in & Create Data

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
#                       Preprocessed Wealth Evolution
# =============================================================================
wealth = pd.read_csv(path + "Data/wealth_processed.csv",parse_dates=['eom'])


# =============================================================================
#                 Estimated Barra Covariance Matrix
# =============================================================================
with open(path + '/Data/Barra_Cov.pkl', "rb") as barra_cov:
    barra_cov = pickle.load(barra_cov)
    
# =============================================================================
# Create lambda list: Kyle's Lambda for each Stock
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


# =============================================================================
#                               Important Dates
# =============================================================================
first_cov_date = pd.to_datetime(min(barra_cov.keys()))
hp_years = np.arange(settings['pf']['dates']['start_year'], settings['pf']['dates']['end_yr'] + 1)

start_oos = settings['pf']['dates']['start_year'] + settings['pf']['dates']['split_years']

dates_m2 = pd.date_range(
    start=first_cov_date + pd.DateOffset(months=pf_set['lb_hor'] + 1),
    end=settings['split']['test_end'] + pd.Timedelta(days=1) - pd.DateOffset(months=1),
    freq='MS'
) - pd.DateOffset(days=1)

dates_oos = pd.date_range(
    start=pd.Timestamp(f"{start_oos}-01-01"),
    end=settings['split']['test_end'] + pd.Timedelta(days=1) - pd.DateOffset(months=1),
    freq='MS'
) - pd.DateOffset(days=1)

dates_hp = pd.date_range(
    start=pd.Timestamp(f"{min(hp_years)}-01-01"),
    end=settings['split']['test_end'] + pd.Timedelta(days=1) - pd.DateOffset(months=1),
    freq='MS'
) - pd.DateOffset(days=1)

#%% Functions

#Random Fourier Features
def rff(X, p=None, g=None, W=None):
    """
    Random Fourier Features

    Parameters:
    - X: Input data matrix (NumPy array)
    - p: Number of features (must be divisible by 2)
    - g: Scalar value for the Gaussian kernel (used to scale identity matrix)
    - W: Optional weight matrix (if provided, skips random generation)

    Returns:
    - Dictionary with keys 'W', 'X_cos', 'X_sin'
    """
    #If W is none, then draw random omega vectors. 
    #Else-wise, use pre-specified omega vectors stored in W.
    if W is None:
        k = X.shape[1]
        cov_matrix = g * np.identity(k)
        W = multivariate_normal.rvs(mean=np.zeros(k), cov=cov_matrix, size=p // 2).T  # shape: (k, p//2)
    
    X_new = np.dot(X, W)  # s_{i,t}' w
    
    return {
        'W': W,
        'X_cos': np.cos(X_new),
        'X_sin': np.sin(X_new)
    }

#%% Overall inputs
"""
Label necessary objects. This enhances comparability to the original R-Code
"""
# Data
data_tc = chars
cov_list = barra_cov

# Investor
mu = pf_set['mu']
gamma_rel = pf_set['gamma_rel']

# Dates
dates_full = dates_m2
lb = pf_set['lb_hor']

# Hyperparameters
rff_feat = True
g_vec = settings['pf_ml']['g_vec']
p_vec = settings['pf_ml']['p_vec']
p_max = max(p_vec)
l_vec = settings['pf_ml']['l_vec']
scale = settings['pf_ml']['scale']
orig_feat = settings['pf_ml']['orig_feat']
add_orig = orig_feat


# Other parameters
iterations = 10
hps = None
balanced = False
seed = settings['seed_no']
dates = dates_full

feat_all = pfml_feat_fun(p = max(p_vec))
#%% Compute Inputs for PFML

#Loop over Variance eta of the RFFs
for g_index, g in enumerate(g_vec):
    print("g_index: ", g_index," ", "g: ", g)
    #Initialise the sliced Object to store the output
    pfml_input = {}

    # Lookback Dates
    dates_lb = pd.date_range(start=min(dates) + relativedelta(months=-(lb + 1)), end=dates_m2.max(), freq='M')
    
    #=============================== 
    # Create Random Fourier Features
    #===============================
    
    #Set seed for reproducability
    random.seed(seed)
    
    # Get Omega Weights and RFFs ( sin(w's) & cos(w's) ) for every stock
    # rff_result is a dictionary with:
    #   - 'W': np.ndarray, Omega weights
    #   - 'X_cos': np.ndarray, cosine-transformed RFFs
    #   - 'X_sin': np.ndarray, sine-transformed RFFs 
    rff_result = rff(data_tc[features], p=p_max, g=g, W = pd.read_csv(path + "Data/rff_w.csv").drop(columns = 'Unnamed: 0').to_numpy())
    
    #Extract Omega Weights
    rff_w = rff_result['W']
    
    #Get the pandas dataframe for RFF transformation of chars
    #rff_x is a pandas DataFrame with
    #   - Each Row is a Stock
    #   - Columns = feat, i.e. the RFF transformed features
    rff_x = np.concatenate([rff_result['X_cos'], rff_result['X_sin']], axis=1)
    rff_x = pd.DataFrame(rff_x, columns=[f"rff{i+1}_cos" if i < p_max // 2 else f"rff{i+1 - p_max//2}_sin" for i in range(p_max)])
    rff_cols = [f"rff{i+1}_cos" for i in range(p_max//2)] + [f"rff{i+1}_sin" for i in range(p_max//2)]
    
    #Save Workspace
    del rff_result
    
    #Get list of new column names (RFFs + constant)
    feat_new = rff_x.columns.tolist()   #RFFs only
    feat_cons = feat_new + ['constant'] #RFFs + constant
    
    #Assemble Information in a DataFrame
    data = pd.concat([data_tc[['id', 'eom', 'valid', 'ret_ld1', 'tr_ld0', 'mu_ld0']], rff_x], axis=1)
    
    #Save Workspace
    del rff_x
    
    #=============================== 
    # Volatility Scaling as in (40)
    #===============================
    scale_frames = []
    for d in dates_lb:
        
        #Create Barra Covariance Matrix S V(f) S' + diag(epsilon^2)
        sigma = create_cov(cov_list[d])
       
        #Get Std.Dev of each stock
        diag_vol = np.sqrt(np.diag(sigma))

        #Get Stock ids
        ids = sigma.index if isinstance(sigma, pd.DataFrame) else np.arange(len(diag_vol))
        
        #Create DataFrame
        temp = pd.DataFrame({'id': ids, 'eom': d, 'vol_scale': diag_vol})
        
        #Save Dataframe
        scale_frames.append(temp)
    
    #Create DataFrame of each stock's Std Dev
    scales = pd.concat(scale_frames, ignore_index=True)
    scales.id = scales.id.astype(int)
    
    #Save Workspace
    del scale_frames, temp
    
    #Merge Stock's Std Devs to data
    data = data.merge(scales, on=['id', 'eom'], how='left')
    
    #Fill Missing Values
    data['vol_scale'] = data.groupby('eom')['vol_scale'].transform(
        lambda x: x.fillna(x.median(skipna=True))
    )
    #Save Workspace
    del scales
    
# =============================================================================
# Construct Inputs 
# =============================================================================

    #Initialise Objects to Store Data
    reals_output = {}
    signal_t_output = {}
        
    #loop over dates
    for d in tqdm(dates_m2, desc="Computing PFML Inputs"):
        
        #Get Returns Data for valid stocks at date d
        data_ret = data[(data['valid']) & (data['eom'] == d)][['id', 'ret_ld1']]
                
        #Get valid ids at date d (fix stock universe from date d for past 12 monthss)
        ids = data_ret['id'].values.astype(int)
        
        #Number of Stocks at date d
        n = len(ids)
        
        #Extract LEAD returns
        r = data_ret['ret_ld1'].values
        
        #Get Barra Covariance Matrix for the subset of Stocks
        sigma = create_cov(cov_list[d], ids=ids)
        
        #Get Kyle's Lambda Matrix
        K_Lambda = pd.DataFrame(create_lambda(lambda_list[d], ids=ids),columns = ids, index = ids)

        #Get AUM in $ of Investor at date d
        w = wealth.loc[wealth['eom'] == d, 'wealth'].values[0]
        
        #Get the riskree rate
        rf = risk_free.loc[risk_free['eom'] == d, 'rf'].values[0]
        
        #Compute m
        m = m_func(w=w, mu=mu, rf=rf, sigma_gam=sigma * gamma_rel, gam=gamma_rel, K_Lambda=K_Lambda, iterations=iterations)
    
        #================================================================
        #                          Compute (24)
        #================================================================
        # Subset of data from current date d going back the past 12 months. This
        # is required to compute (24) where theta runs from 0 to 11.
        #   Important: We are fixing the stock universe from date d for all the
        #               past 12 months. This is why we must do standardisation
        #               in the loop
        # Notice that stocks must have at least 12 months of past return data
        # in order to compute (24)
        data_sub = data[
            (data['id'].isin(ids)) & 
            (data['eom'] >= d - relativedelta(months=lb+1)) & 
            (data['eom'] <= d)
        ]
        
        #Add constant
        data_sub = data_sub.assign(constant = 1)
    
        #Standardise RFFs (de-meaning and sum of squares = 1)
        #   Reason: the sum of a column is unaffected by the number of stocks (otherwise leverage over time would be dependent on the number of stocks)
        data_sub[feat_new] = data_sub.groupby('eom')[feat_new].transform(lambda x: x-x.mean()) #don't de-mean constant (or else its zero)
        data_sub[feat_cons] = data_sub.groupby('eom')[feat_cons].transform(lambda x: x* np.sqrt(1/(x**2).sum())) # Sum of Squares = 1
    
        #Store Results in a Dictionary
        data_sub = data_sub.sort_values(by=['eom', 'id'], ascending=[False, True])
        data_sub = {g: df for g, df in data_sub.groupby('eom')}
        
        #======================================
        # I) Scaling Signals:  Diag(1/sigma)RFF(s)
        #======================================
        # Define Object to store the Result
        signals = {}
        for eom_date, df in data_sub.items():
            #Get the Signals (Standardized RFFs)
            s = data_sub[eom_date][feat_cons].to_numpy()
            #Get 1/Std.Dev of each stock as a diagonal matrix
            scales_mat = np.diag(1 / data_sub[eom_date]['vol_scale'].values)
            #Scale each Stock's RFFs by the Std.Dev
            s = scales_mat @ s            
            #Store Results
            signals[eom_date] = pd.DataFrame(s,columns = feat_cons, index=ids)
            
        #Save Workspace
        del scales_mat,s
        
        #Save more workspace: this data is now included in signals dictionary and can be removed from data_sub
        for key in data_sub.keys():
            data_sub[key].drop(feat_cons,axis=1,inplace=True)
    
        # ===============================
        # II) Compute gtm (m @ g_t) as in (24) 
        # ===============================
        gtm = {} #m @ np.diag(gt) for each dates in d
        eom_keys = sorted(list(data_sub.keys()),reverse = True)
        for eom_date in eom_keys:
            df = data_sub[eom_date]
            gt = (1 + df['tr_ld0'].values) / (1 + df['mu_ld0'].values) #g_t as in (6)
            gt = np.nan_to_num(gt, nan=1.0) #Set Missing Values to 1
            gtm[eom_date] = m @ np.diag(gt) #Matrix Product
        
        
        # ============================================
        # III) Compute Cumulative Product of gtm as in (24)
        # ============================================
        
        #Define initial objects
        gtm_agg = {}  #gtm_agg = \prod_{\tau=1}^\theta m g_{t-\tau+1} where \theta is the dict key
        gtm_agg_l1 = {} #Lagged gtm_agg
        
        # Get the keys (names) from gtm, excluding the last one
        #keys = list(gtm.keys())[:-1] if isinstance(gtm, dict) else range(len(gtm)-1)
        latest_date = eom_keys[0]
        gtm_agg = {latest_date: np.eye(len(ids))}
        
        gtm_agg_l1 = copy.deepcopy(gtm_agg)
    
        for i in range(0, lb):
            prev = eom_keys[i + 1]
            curr = eom_keys[i]
            gtm_agg[prev] = gtm_agg[curr] @ gtm[curr]
            #Lagged cumulative product
            gtm_agg_l1[prev] = gtm_agg_l1[curr] @ gtm[prev]
            
        #==============================================================
        # IV) Compute Weighted Signals \tilde{s}_t from (24)
        #==============================================================
        
        ##!!!!! Note: \tilde{s}_t is called omega in the code
    
        # Compute omega and omega_l1
        omega = 0 # \tilde{s}_t
        const = 0 # I do not know what this is or why it is being used to compute \pi_t in (24)
        omega_l1 = 0 #= \tilde{s}_{t-1}
        const_l1 = 0
    
        for theta in range(lb + 1):
            d_new = d + pd.offsets.MonthEnd(-theta)
            d_new_l1 = d_new + pd.offsets.MonthEnd(-1)
            s = signals[d_new]
            s_l1 = signals[d_new_l1]
            
            omega += gtm_agg[d_new] @ s.to_numpy() #s = f(s[d_new])
            const += gtm_agg[d_new]
            
            omega_l1 += gtm_agg_l1[d_new] @ s_l1
            const_l1 += gtm_agg_l1[d_new]
        #omega = \sum_{\theta = 0}^lb [ (\pi_{\tau = 1}^{\theta} m g_{t-\tau+1})  f(s_{t-\theta})) ] #=\tilde{s_t}'
        omega = np.linalg.solve(const, omega)
        omega_l1 = np.linalg.solve(const_l1, omega_l1)
    
        gt = np.diag((1 + data_sub[d]['tr_ld0'].values) / (1 + data_sub[d]['mu_ld0'].values))
        omega_chg = omega - gt @ omega_l1
        
        omega=pd.DataFrame(omega, index = ids, columns = feat_cons)
        omega_chg = pd.DataFrame(omega_chg, index = ids, columns = feat_cons)
    
        #================================================
        # Compute (25): \tilde{r}_{t+1} & \tilde{Sigma}_t
        #================================================
        ###### Compute a single summand for date t in (25)
        
        #r_tilde
        r_tilde = omega.T @ r
        
        #risk = gamma * \tilde{s}_t' \Sigma \tilde{s}_t
        risk = gamma_rel * omega.to_numpy().T @ sigma.to_numpy() @ omega.to_numpy()
        risk = pd.DataFrame(risk,index = feat_cons, columns = feat_cons)
        
        #Transaction costs: risk + tc = \tilde{\Sigma}
        tc = w * omega_chg.to_numpy().T @ K_Lambda.to_numpy() @ omega_chg.to_numpy()
        tc = pd.DataFrame(tc, index = feat_cons, columns = feat_cons)
        
        #denom = \tilde{\Sigma}
        denom = risk + tc
        
        #Summarise results
        reals = {'r_tilde':r_tilde[feat_all], 'denom':denom.loc[feat_all,feat_all], 'risk':risk.loc[feat_all,feat_all], 'tc':tc.loc[feat_all,feat_all]}
        signal_t = signals[d]
    
        #Save Outputs
        reals_output[d] = reals
        signal_t_output[d] = signal_t[feat_all]
        
        pfml_input[g_index] = {'reals':reals_output, 'signal_t':signal_t_output, 'rff_w':rff_w}
        
    #Save Object
    with open(path + f"/Data/pfml_input_{g_index}.pkl", 'wb') as f:
        pickle.dump(pfml_input, f)
        
    del pfml_input
