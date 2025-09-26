# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:16:09 2025

Download riskfree rate data from Fama-French

Download Daily Stock Return Data from WRDS
"""

#%% Path
path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/"

folder = "JKMP_22_Replication/"

#%% Libraries
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta

#%% WRDS access
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

#Set os directory
os.chdir(path)
load_dotenv(path + folder + "Code/Credentials.env")
#%% Set up WRDS Engine
connection_string = (
  "postgresql+psycopg2://"
 f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)
wrds = create_engine(connection_string, pool_pre_ping=True)

#%% Get Daily Stock Return Data
"""
start_date = "01/01/1952"
end_date = "12/31/2024"

query = (
  "SELECT dsf.permno, dsf.dlycaldt AS date, "
         "dsf.dlyret AS ret, dsf.primaryexch "
    "FROM crsp.dsf_v2 AS dsf "
   f"WHERE dsf.dlycaldt BETWEEN '{start_date}' AND '{end_date}' "
          #US-listed stocks
          "AND sharetype = 'NS' "
          #security type equity
          "AND securitytype = 'EQTY' "  
          #security sub type common stock
          "AND securitysubtype = 'COM' "
          #US Incorporation Flag (Y/N)
          "AND usincflg = 'Y' " 
          #Issuer is a corporation
          "AND issuertype in ('ACOR', 'CORP') " 
          #NYSE, AMEX, NASDAQ Stocks
          "AND primaryexch in ('N', 'A', 'Q') "
          #Stock Prices when or after issuence
          "AND conditionaltype in ('RW', 'NW') "
          #Actively Traded Stocks
          "AND tradingstatusflg = 'A'"
)

crsp_daily = (pd.read_sql_query(
    sql=query,
    con=wrds,
    dtype={"permno": int},
    parse_dates={"date"})
)

#----Save Data

#Create SQLite Database
myDataBase = sqlite3.connect(database=path + folder + "Data/crsp_daily.db")

#Save to DataBase
crsp_daily.to_sql('crsp_daily',myDataBase,if_exists='fail',index=False)
"""
#%% Get Riskfree Rate
#Read in Fama-French daily riskfree-rate data
FF_RF = pd.read_csv(path + folder + "Data/FF_RF_daily.csv").drop(['Mkt-RF','SMB','HML'],axis=1)

#Get proper date format
FF_RF['date'] = pd.to_datetime(FF_RF['date'].astype(str), format='%Y%m%d')

#Drop dates that are not in sample
FF_RF = FF_RF[FF_RF['date'] > '1951-12-31']

#Convert percentage to decimal
FF_RF['RF'] = FF_RF['RF'] / 100  

#%% Generate Excess Return

#Connect to DataBase
myDataBase = sqlite3.connect(database=path + folder +"Data/crsp_daily.db")

"""
I tried direct SQL commands, but it was much, much faster to read the data into
Python in chunks and create a new Table. Although not as elegant, I went for this
quick solution.
"""

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
    query = (f"SELECT * FROM crsp_daily "
             f"WHERE date BETWEEN '{current_start.date()}' AND '{current_end.date()}'")

    chunk = pd.read_sql_query(query, con=myDataBase, parse_dates=['date']).dropna()

    # Ensure ret is float and clean
    chunk['ret'] = pd.to_numeric(chunk['ret'], errors='coerce')
    chunk.dropna(subset=['ret'], inplace=True)

    # Merge with FF_RF to get RF for each date
    merged = pd.merge(chunk, FF_RF[['date', 'RF']], on='date', how='left')
    merged.dropna(subset=['RF'], inplace=True)  # In case RF is missing for a date

    # Compute excess return
    merged['ret_excess'] = merged['ret'] - merged['RF']
    
    #To Save memory, compress columns
    merged['ret'] = merged['ret'].astype('float32')
    merged['ret_excess'] = merged['ret_excess'].astype('float32')
    merged['primaryexch'] = merged['primaryexch'].astype('category')


    # Save result into a new table (or append if exists)
    merged[['permno', 'date', 'ret', 'primaryexch', 'ret_excess']].to_sql(
        'crsp_daily_excess',
        con=myDataBase,
        if_exists='append',
        index=False
    )

    # Move to next chunk
    current_start = current_end + timedelta(days=1)

print("Processing complete.")


#Delte old Table without the excess return
myDataBase.execute("DROP TABLE IF EXISTS crsp_daily;")
myDataBase.commit()
print("Table 'crsp_daily' has been deleted.")


#Rename new Table
myDataBase.execute("ALTER TABLE crsp_daily_excess RENAME TO d_ret_ex;")
myDataBase.commit()
print("Table renamed from 'crsp_daily_excess' to 'd_ret_ex'.")


myDataBase.close()
