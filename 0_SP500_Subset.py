# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:41:37 2025

@author: Patrick


Only Keep Stocks belonging to the S&P 500
"""
import pandas as pd
import sqlite3
from datetime import date, timedelta



path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/JKMP_22_Replication/"

Constituents = (pd.read_csv(path + "Data/SP500_Historical_Constituents.csv",
                            parse_dates = ['start', 'ending','date'])
                .drop('Unnamed: 0',axis = 1)
                #End of Month
                .assign(eom = lambda x: x.date + pd.offsets.MonthEnd(0))
                )


#Connect to DataBase
db_JKP = sqlite3.connect(database=path + "Data/JKP_US.db")
db_JKP_SP500 = sqlite3.connect(database=path +"Data/JKP_SP500.db")
db_crsp_daily = sqlite3.connect(database=path +"Data/crsp_daily.db")
db_crsp_daily_SP500 = sqlite3.connect(database=path +"Data/db_crsp_daily_SP500.db")




#%% JKP

# Define date range and chunk size
start_date = pd.to_datetime("1952-01-01")
end_date = pd.to_datetime("2024-12-31")
chunk_size = pd.DateOffset(years=5)


current_start = start_date

#-------- Generate Excess Return for each chunk and save data
while current_start < end_date:
    current_end = min(current_start + chunk_size, end_date)

    print(f"Processing chunk: {current_start.date()} to {current_end.date()}")

    # Load chunk of crsp_daily data
    query = ("SELECT * "
             +"FROM Factors " 
             +f"WHERE date BETWEEN '{current_start.date()}' AND '{current_end.date()}' "
             )

    chunk = pd.read_sql_query(query, con=db_JKP, parse_dates=['eom','date'])
    
    df_subset = (Constituents.merge(chunk, left_on = ['permno','eom'], 
                             right_on = ['id','eom'], 
                             suffixes = ('_drop',''), 
                             how = 'inner')
           )
    df_subset.drop(columns = [col for col in df_subset.columns if col.endswith('_drop')] + ['start','ending'], inplace=True)
    
     
    # Save result into a new table (or append if exists)
    df_subset.to_sql(
        'Factors',
        con=db_JKP_SP500,
        if_exists='append',
        index=False
    )
        
    # Move to next chunk
    current_start = current_end + timedelta(days=1)

print("Processing db_JKP_SP500 complete.")

#%% CRSP Daily

del df_subset, chunk

# Define date range and chunk size
start_date = pd.to_datetime("1952-01-01")
end_date = pd.to_datetime("2024-12-31")
chunk_size = pd.DateOffset(years=5)


current_start = start_date

#-------- Generate Excess Return for each chunk and save data
while current_start < end_date:
    current_end = min(current_start + chunk_size, end_date)

    print(f"Processing chunk: {current_start.date()} to {current_end.date()}")

    # Load chunk of crsp_daily data
    query = ("SELECT * "
             +"FROM Factors " 
             +f"WHERE date BETWEEN '{current_start.date()}' AND '{current_end.date()}' "
             )

    # Load chunk of crsp_daily data
    query = ("SELECT * "
             +"FROM d_ret_ex " 
             +f"WHERE date BETWEEN '{current_start.date()}' AND '{current_end.date()}' "
             )
    
    chunk = pd.read_sql_query(query, con=db_crsp_daily, parse_dates=['eom','date'])
    chunk['eom'] = chunk['date'] + pd.offsets.MonthEnd(0)
    
    df_subset = (Constituents.merge(chunk, on = ['eom','permno'], how = 'inner', suffixes = ('_drop',''))
                .drop(['start','ending','date_drop', 'eom'],axis=1)
                )
     
    # Save result into a new table (or append if exists)
    df_subset.to_sql(
        'Factors',
        con=db_crsp_daily_SP500,
        if_exists='append',
        index=False
    )
        
    # Move to next chunk
    current_start = current_end + timedelta(days=1)

print("Processing db_crsp_daily_SP500 complete.")

