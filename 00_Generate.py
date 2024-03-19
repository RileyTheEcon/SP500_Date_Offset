# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:38:52 2024

@author: RC
"""





# ================================ LIBRARIES ================================ #
import pandas as pd
import yfinance as yf

import warnings
from fredapi import Fred
from time import sleep
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup




dictArgs = {'key_file_path'    : 'fred_api_key.txt', # set local directory
            'fred_source_path' : 'fred_source.csv',  # set location of data dictionary
            'security_sym'     : '^GSPC',            # set security symbol
            'security_name'    : 'SP500',            # set security name
            }

# =========================================================================== #





# ================================ FUNCTIONS ================================ #
def get_api_key (key_file_path) :
    try:
        with open(key_file_path, 'r') as file : 
            key = file.read()
        #   endwith
    except FileNotFoundError : print(f"You're missing {key_file_path}!")
    except Exception as oops : print(f"Something odd happened: {oops}")
    
    return key
####
# I originally developed the below function for a personal project and built
# on it for this assignment: originally took data series names and ID codes as
# List of Tuples, expanded functionality to take table instead and create the
# list of tuples internally
def get_fred_data (fred_key, dfFred, 
                   col_names = {'Name':'Name', 'SeriesID':'SeriesID'},
                   try_limit=5, courtesy_sleep = 0.5
                   ) :
    '''
    Parameters
    ----------
    fred_key : STR
        Valid FRED API as str
    dfFred   : DataFrame-like
        DataFrame-like with an array of desired variable names, and FRED
        series ID codes
    col_names : DICT, optional
        Dictionary matching column names of dfFred column names with the column
        names assumed by the function.
    try_limit : INT, optional
        Function will attempt to access the data associated with a given series
        ID this many times before issuing a warning and continuing. 
        The default is 5.
    courtesy_sleep: FLT, optional
        Wait between making new server requests to avoid flooding the server,
        or if the server is erroring. The default is 0.5 seconds.
    Returns : dfData
    -------
    DATAFRAME
        Returns a dataframe of data requested from FRED server. Each data
        series is in its own column, joined on datetime index, and sorted
        chronologically
    '''
    
    dfFred = pd.DataFrame(dfFred)   # convert to DF object for version control
    dfData = pd.DataFrame()         # create place in memory
    fred   = Fred(fred_key)         # convert to API key object
    
    
    
    #   Version control df names
    col_names = {value:key for key, value in col_names.items()}
    dfFred.rename(columns=col_names, inplace=True)
    
    
    
    #   Remove gaps & warn duplicates
    dfFred = dfFred.dropna()
    
    item_dupe = []
    for name in dfFred.columns :
        item_dupe = dfFred[dfFred.duplicated(name)][name].tolist()
        if len(item_dupe)>0 :
            warnings.warn(f"Duplicated entries found in '{name}': {item_dupe}")
        #   end if
    #   end for
    dfFred = dfFred[~dfFred['Name'].duplicated(keep='first')]
    
    
    
    #   Download data -- using item-wise iter to be nice to hosting server
    for indx, row in dfFred.iterrows() :
        bContinue       = 0
        intErrorCount   = 0
        
        while (bContinue==0)&(intErrorCount<try_limit) : 
            try : # Attempt dl through API
                data = pd.DataFrame(fred.get_series(row['SeriesID'])
                                    ).rename(columns={0:row['Name']})
                data.index.name = 'date'
            except : # Extract data from raw txt page if API fails for any reason
                try: 
                    htmlPage = dlURL('https://fred.stlouisfed.org/data/'+
                                     row['SeriesID']+'.txt')
                    
                    listRows = htmlPage.text.split('\n')
                    listRows = listRows[listRows.index([x for x in listRows 
                                                        if 'DATE' in x][0])+1:]
                    listRows = [[pd.to_datetime(x[:x.index(' ')]),
                                 float(isolate_better(x,' ','\r',b_end=1))
                                 ] 
                                for x in listRows if x!=''
                                ]
                    
                    data = pd.DataFrame(listRows,columns=['index',row['Name']]
                                     ).set_index('index')
                    data.index.name = 'date'
                except : 
                    intErrorCount+=1
                    sleep(1)
                else : bContinue = 1
                #   endtry
            else : bContinue = 1
            #   endtry
        #   endwhile
        
        #   If both approaches above fail - warn user
        if intErrorCount>=try_limit :
            warnings.warn('\nFailure in accessing data from:\n'+
                          f'Name: {row["Name"]}\n'+
                          f'ID:   {row["SeriesID"]}\n'
                          )
            
        #   If the above ran successfully - append along date index
        else : 
            if len(dfData)==0 : dfData = data
            else : dfData = dfData.join(data,how='outer',
                                         )
        #   endif
        
        sleep(courtesy_sleep) # Let's do our best to be polite to the hosting server
    #   endfor
    
    return dfData.sort_index()
####
def get_historic_data (SeriesID, api_key, 
                       series_name = 'value', stale_data = 400
                       ) :
    
    #   Get data
    fred   = Fred(api_key)
    df = fred.get_series_all_releases(SeriesID)
    
    #   Calc gap between reported date and actual date; drop stale data
    df['diff'] = df['realtime_start'] - df['date']
    df = df[df['diff'] <= pd.Timedelta(str(stale_data)+' days')
            ].copy()
    
    #   Get most recent data by actual date
    # Some reports contain original data and revisions, so we grab the most
    # current data from each reporting date
    max_order_indices = (df.sort_values('date')
                           .groupby('realtime_start')['date']
                           .idxmax()
                           )
    df = df.loc[max_order_indices].copy()
    
    #   Drop unneeded columns; set index    
    for col in ['date', 'diff'] : del df[col]
    
    dict_rename = {'realtime_start' : 'date'}
    if series_name!='value' : dict_rename['value'] = series_name
    
    df.rename(columns = dict_rename, 
              inplace = True
              )
    df.set_index('date', inplace = True)
    
    return df
####
def dlURL (url , parser = "html.parser" ) :
    req         = Request(url,headers={'User-Agent':'Mozilla/5.0'})
    urlClient   = urlopen(req)
    pageRough   = urlClient.read()
    urlClient.close()
    pageSoup    = soup(pageRough,parser)

    return pageSoup
#### / ####
# "isolate_better" and its helper function "reverse" are functions I originally
# wrote for a personal project while still teaching myself Python basics.
# Is it a crude and inefficient way to do something that there are probably
# native functions/methods for? Probably, but it works with the other
# pre-existing code I have.
def reverse (stri) :
    x = ""
    for i in stri :
        x = i + x
    return x
####
def isolate_better (stri , start , end, b_end = 0) :
    strShort    = ''
    posStart    = 0
    posEnd      = 0

    if b_end==1 :
        posEnd      = stri.find(end)
        strShort    = stri[:posEnd]
        strShort    = reverse(strShort)
        start       = reverse(start)
        posStart    = posEnd - strShort.find(start)
    #
    else :
        posStart    = stri.find(start)+len(start)
        strShort    = stri[posStart:]
        posEnd      = posStart + strShort.find(end)
    #
    return stri[posStart:posEnd]
#### / ####
def main(key_file_path,     # File path for FRED API key, txt
         fred_source_path,  # File path for variable names & FRED series ID, csv
         security_sym,      # Ticker symbol for security of interest (S&P 500)
         security_name      # Name of security of interest
         ) :
    
    
    
    #   Download YFinance data
    dfFinance = yf.download(security_sym)['Adj Close']
    dfFinance.rename(security_name, inplace=True)
    #
    
    
    
    
    
    #   Get FRED API key from txt doc; Get list of var from cvs
    fred_key = get_api_key(key_file_path)
    
    try :   # Get book of Econ variables
        dfFred = pd.read_csv(fred_source_path)
    except FileNotFoundError : print("Get your own API key!")
    except Exception as oops : print(f"Something odd happened: {oops}")
    #
    
    
    
    
    
    #   Iter thru data series; handle as specified
    dfEcon = pd.DataFrame() # make place in memory
    
    for indx, row in dfFred.iterrows() :
        
        # Handle where there is no delta between reported date & reporting date
        if row['Handling'] == 'Daily' :
            df = get_fred_data(fred_key, pd.DataFrame(row).T)
            
        #   end if
        
        elif row['Handling'] in ['Monthly', 'GDP'] :
            
            df = get_historic_data(row['SeriesID'], 
                                   fred_key
                                   )
            df.rename(columns = {'value': row['Name']},
                      inplace = True
                      )
            
            # Handle where reporting delta is roughly one month 
            # (assuming no interm revisions)
            if row['Handling'] == 'Monthly' :
                df[row['Name']+'_M/M'] = (df[row['Name']]
                                          .pct_change(1, fill_method=None)
                                          )
                
            # Handle where reporting delta is roughly one quarter (allowing revisions)
            elif row['Handling'] == 'GDP' :
                df[row['Name']+'_new-release'] = [int(x in [1, 4, 7, 10]) 
                                                  for x in df.index.month
                                                  ]
                df[row['Name']+'_Q/Q'] = (df[row['Name']]
                                          .pct_change(3, fill_method=None)
                                          )
                
            #   endif
            df[row['Name']+'_Y/Y'] = (df[row['Name']]
                                      .pct_change(12, fill_method=None)
                                      )
            
        #   endif
        else : raise ValueError(f'Handling method {row["Handling"]} not specified!')
        
        dfEcon = dfEcon.join(df, how = 'outer')
    #   end for
    
    
    
    
    
    #   Combine & export
    dfData = pd.DataFrame(dfFinance).join(dfEcon, how = 'left')
    dfData.ffill(inplace=True)
    dfData = dfData[dfData['GDP'].notnull()]
    
    dfData.to_csv('SP500_Offset.csv')
    #
    
####
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    print(__doc__)
    main(**dictArgs)
#   endif
# =========================================================================== #



'''
key_file_path    = dictArgs['key_file_path']
fred_source_path = dictArgs['fred_source_path']
security_sym     = dictArgs['security_sym']
security_name    = dictArgs['security_name']
'''





