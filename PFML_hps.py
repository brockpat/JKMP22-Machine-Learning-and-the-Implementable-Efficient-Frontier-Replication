# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:56:56 2025

@author: Patrick

Generate the Object hps which merges previously computed results into a dictionary.
"""

import pandas as pd
import numpy as np
from functools import reduce
import pickle
from tqdm import tqdm

#%% Libraries
import sqlite3

#Run specific functions
from General_functions import *

#%% Get Preliminaries

#Settings used throughout
settings, pf_set = get_settings()

g_vec = settings['pf_ml']['g_vec']

#%%
hps = {}
for g_index, _ in enumerate(g_vec):
    print("g_index: ", g_index)
    with open(path + f'/Data/pfml_input_{g_index}.pkl', "rb") as f:
        rff_w = pickle.load(f)[g_index]['rff_w']
        
    with open(path + f'/Data/aims.pkl', "rb") as f:
        aims = pickle.load(f)[g_index]
    
    validation = pd.read_csv(path + "Data/validation.csv")
    validation = validation[validation['g'] == g_index]
    
    hps[g_index] = {'aim_pfs_list':aims, 'validation': validation, 'rff_w': rff_w}
    
print("Saving Data")
with open(path + f'/Data/hps.pkl', 'wb') as f:
    pickle.dump(hps, f)   

