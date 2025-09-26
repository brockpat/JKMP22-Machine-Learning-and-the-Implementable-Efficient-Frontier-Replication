# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:44:49 2025

@author: pbrock
"""

#%% Path
#Set the Path to the replication folder
path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/JKMP_22_Replication/"

#Set Working Directory
import os
os.chdir(path + "Code/")

for file in ['Prepare_Data.py', 'Estimate Covariance Matrix.py', 
             'PFML_Input_Data.py',
             'PFML_Search_Coef.py', 'PFML_hp_reals.py', 'PFML_aim_fun.py', 'PFML_hps.py',
             'PFML_best_hps.py']:
    
    with open(path + "Code/" + file, encoding='utf-8') as script:
        exec(script.read())
