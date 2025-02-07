#!/usr/bin/env python
"""
collection of functions for the final case study solution
"""

import os
import sys
import re
import shutil
import time
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

COLORS = ["darkorange","royalblue","slategrey"]

def fetch_data(data_dir):
    """
    laod all json formatted files into a dataframe
    """

    ## input testing
    if not os.path.isdir(data_dir):
        raise Exception("specified data dir does not exist")
    if not len(os.listdir(data_dir)) > 0:
        raise Exception("specified data dir does not contain any files")

    file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search(r"\.json",f)]
    correct_columns = ['country', 'customer_id', 'day', 'invoice', 'month',
                       'price', 'stream_id', 'times_viewed', 'year']

    ## read data into a temp structure
    all_months = {}
    for file_name in file_list:
        df = pd.read_json(file_name)
        all_months[os.path.split(file_name)[-1]] = df

    ## ensure the data are formatted with correct columns
    for f,df in all_months.items():
        cols = set(df.columns.tolist())
        if 'StreamID' in cols:
             df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in cols:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        if 'total_price' in cols:
            df.rename(columns={'total_price':'price'},inplace=True)

        cols = df.columns.tolist()
        if sorted(cols) != correct_columns:
            raise Exception("columns name could not be matched to correct cols")

    ## concat all of the data
    df = pd.concat(list(all_months.values()),sort=True)
    years,months,days = df['year'].values,df['month'].values,df['day'].values 
    dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
    df['invoice_date'] = np.array(dates,dtype='datetime64[D]')
    df['invoice'] = [re.sub(r"\D+","",i) for i in df['invoice'].values]
    
    ## sort by date and reset the index
    df.sort_values(by='invoice_date',inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    return(df)


def convert_to_ts(df_orig, country=None):
    """
    given the original DataFrame (fetch_data())
    return a numerically indexed time-series DataFrame 
    by aggregating over each day
    """

    if country:
        if country not in np.unique(df_orig['country'].values):
            raise Exception("country not found")
    
        mask = df_orig['country'] == country
        df = df_orig[mask]
    else:
        df = df_orig
        
    ## use a date range to ensure all days are accounted for in the data
    invoice_dates = df['invoice_date'].values
    # Calculate first and last month and year in the dataset (i.e. '2017-11' and '2019-07')
    start_month = '{}-{}'.format(df['year'].values[0],str(df['month'].values[0]).zfill(2))
    stop_month = '{}-{}'.format(df['year'].values[-1],str(df['month'].values[-1]).zfill(2))
    df_dates = df['invoice_date'].values.astype('datetime64[D]')
    days = np.arange(start_month,stop_month,dtype='datetime64[D]')
    
    
    purchases = np.array([np.where(df_dates==day)[0].size for day in days])
    invoices = [np.unique(df[df_dates==day]['invoice'].values).size for day in days]
    streams = [np.unique(df[df_dates==day]['stream_id'].values).size for day in days]
    views =  [df[df_dates==day]['times_viewed'].values.sum() for day in days]
    revenue = [df[df_dates==day]['price'].values.sum() for day in days]
    #year_month = ["-".join(re.split("-",str(day))[:2]) for day in days]

    df_time = pd.DataFrame({'date':days,
                            'purchases':purchases,
                            'unique_invoices':invoices,
                            'unique_streams':streams,
                            'total_views':views,
                            'revenue':revenue})
    
    df_time['date'] = pd.to_datetime(df_time['date'])
    
    return(df_time)


def fetch_ts(df, ts_files_dir, clean=False):
    """
    convenience function to read in new data
    uses csv to load quickly
    use clean=True when you want to re-create the files
    """
    
    if clean:
        shutil.rmtree(ts_files_dir)
    if not os.path.exists(ts_files_dir):
        os.mkdir(ts_files_dir)

    ## if files have already been processed load them        
    if len(os.listdir(ts_files_dir)) > 0:
        print("... loading ts data from files")
        return({re.sub(r"\.csv","",cf)[3:]:pd.read_csv(os.path.join(ts_files_dir,cf)) for cf in os.listdir(ts_files_dir)})

    ## get original data
    #print("... processing data for loading")
    #df = fetch_data(data_dir)

    ## find the top ten countries (wrt revenue)
    table = pd.pivot_table(df,index='country',values="price",aggfunc='sum')
    table.columns = ['total_revenue']
    table.sort_values(by='total_revenue',inplace=True,ascending=False)
    top_ten_countries =  np.array(list(table.index))[:10]

    #file_list = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if re.search(r"\.json",f)]
    #countries = [os.path.join(data_dir,"ts-"+re.sub(r"\s+","-",c.lower()) + ".csv") for c in top_ten_countries]

    ## load the data
    dfs = {}
    dfs['all'] = convert_to_ts(df)
    for country in top_ten_countries:
        country_id = re.sub(r"\s+","_",country.lower())
        #file_name = os.path.join(data_dir,"ts-"+ country_id + ".csv")
        dfs[country_id] = convert_to_ts(df,country=country)

    ## save the data as csvs    
    for key, item in dfs.items():
        item.to_csv(os.path.join(ts_files_dir,"ts-"+key+".csv"),index=False)
        
    return(dfs)

#Check whether the dataset was loaded or not
def get_df_shape(df):
    if (df.shape[0] == 0) :
        raise Exception("Something went wrong! No data were the dataframe.")
    else:
        # Display the shape of the data 
        print ("Dataframe created successfully!")
        print("SHAPE OF THE DATA: ")
        print("Number of rows: " + str(df.shape[0]))
        print("Number of columns: " + str(df.shape[1]))

def main(input_data_directory, output_data_directory):
    """
    main function that calls the previous functions 
    Args:
        data_dir (str): source data
        ts_data_dir (str): output data
    """
    #Start counting the delay 
    run_start = time.time() 
    
    #Load data from multiple files in a single dataframe
    df1 = fetch_data(input_data_directory)
         
    #Function call
    get_df_shape(df1)   
       
    #Convert data in the dataframe to time series data and save them in csv files
    fetch_ts(df1, output_data_directory, clean=False)

    #Display the path, including names, of the files created
    file_list = [os.path.join(output_data_directory,f) for f in os.listdir(output_data_directory) if re.search(r"\.csv",f)]
    print(file_list)
    
    #Stop counting the delay and display the total delay from te beginning to the end
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print("load time:", "%d:%02d:%02d"%(h, m, s))

if __name__ == "__main__":
    input_data_directory = os.path.join("data","cs_train")
    output_data_directory = os.path.join("data","ts_data")
    main(input_data_directory, output_data_directory)



