# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:17:02 2024

@author: RC
"""





# ================================ LIBRARIES ================================ #
import numpy as np
import pandas as pd
import yfinance as yf
import datasets
from typing import List
import csv
import json
import logging

import warnings
from fredapi import Fred
from time import sleep
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup




dictArgs = {'key_file_path'    : 'fred_api_key.txt',      # set local directory
            'fred_source_path' : 'fred.csv',              # set location of data dictionary
            'security_sym'     : '^GSPC',                 # set security symbol
            'security_name'    : 'SP500',                 # set security name
            'export_path'      : 'SP500_Date_Offset.csv'  # set export destination
            }

# =========================================================================== #





# ================================== INFO =================================== #
_CITATION = """\
@online{BEA_GDP,
  author       = {{U.S. Bureau of Economic Analysis}},
  title        = {Real Gross Domestic Product [GDPC1]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/GDPC1},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{Consumer_Sentiment,
  author       = {{Surveys of Consumers, University of Michigan}},
  title        = {University of Michigan: Consumer Sentiment © [UMCSENT]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/UMCSENT},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{CPI_All_Items,
  author       = {{U.S. Bureau of Labor Statistics}},
  title        = {Consumer Price Index for All Urban Consumers: All Items in U.S. City Average [CPIAUCSL]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/CPIAUCSL},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{CPI_All_Items_Less_Food_Energy,
  author       = {{U.S. Bureau of Labor Statistics}},
  title        = {Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average [CPILFESL]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/CPILFESL},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{Fed_Funds_Rate,
  author       = {{Board of Governors of the Federal Reserve System (US)}},
  title        = {Federal Funds Effective Rate [DFF]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/DFF},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-20}
}
@online{New_Housing_Units_Started,
  author       = {{U.S. Census Bureau and U.S. Department of Housing and Urban Development}},
  title        = {New Privately-Owned Housing Units Started: Total Units [HOUST]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/HOUST},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-19}
}
@online{New_One_Family_Houses_Sold,
  author       = {{U.S. Census Bureau and U.S. Department of Housing and Urban Development}},
  title        = {New One Family Houses Sold: United States [HSN1F]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/HSN1F},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{PCE_Chain_Price_Index,
  author       = {{U.S. Bureau of Economic Analysis}},
  title        = {Personal Consumption Expenditures: Chain-type Price Index [PCEPI]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/PCEPI},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{PCE_Excluding_Food_Energy,
  author       = {{U.S. Bureau of Economic Analysis}},
  title        = {Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index) [PCEPILFE]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/PCEPILFE},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{SP500,
  author       = {{S&P Dow Jones Indices LLC}},
  title        = {S\&P 500 [SP500]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/SP500},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-20}
}
@online{Total_Construction_Spending,
  author       = {{U.S. Census Bureau}},
  title        = {Total Construction Spending: Total Construction in the United States [TTLCONS]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/TTLCONS},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{Total_Nonfarm_Employees,
  author       = {{U.S. Bureau of Labor Statistics}},
  title        = {All Employees, Total Nonfarm [PAYEMS]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/PAYEMS},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
@online{Unemployment_Rate,
  author       = {{U.S. Bureau of Labor Statistics}},
  title        = {Unemployment Rate [UNRATE]},
  year         = {2024},
  url          = {https://fred.stlouisfed.org/series/UNRATE},
  organization = {FRED, Federal Reserve Bank of St. Louis},
  urldate      = {2024-03-13}
}
"""

# You can copy an official description
_DESCRIPTION = """\
The S&P 500 Date Offset project seeks to offer an alternative way of modeling
financial trends from economic conditions.

Due to the rigorous tabulation process, the gap between when economic data is
reported and the time which it is meant to describe can be months. Moreover,
when this data is released, it is usually backdated to correspond with the date
of the first day of the time period it reflects. That said, if the data causes
a correction in financial markets, that change will be reflected in the data
for the day of the release (and not the back dated day!).

That prompts the immediate question: would data offset to reflect investors'
knowledge in the moment provide a better model for the markets than the
traditionally structured data?

In addition to the S&P 500 daily close price--which is used here to represent
the stock market overall--variables were chosen from the list of Leading,
Lagging and Coincident Indicators as maintained by the Conference Board.
Those variables and their transformations are:
(M/M = Month-over-month percent change, 
 Q/Q = Quarter-over-quarter percent change, 
 Y/Y = Year-over-year percent change
 )
    
 - Consumer Sentiment, University of Michigan
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - Consumer Price Index
  - All Items
  - All Items less Food & Energy
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - Federal Funds Rate
 Freq: Daily
 Tran: None
 
 - Gross Domestic Product
 Freq: Quarterly
 Tran: Q/Q, Y/Y
 
 - New Housing Units Started
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - New One Family Houses Sold
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - Personal Consumption Expenditure: Chain-type Price Index
  - All Items
  - All Items excluding Food & Energy
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - Total Construction Spending
 Freq: Monthly 
 Tran: M/M, Y/Y
 
 - Total Nonfarm Employment
 Freq: Monthly
 Tran: M/M, Y/Y
 
 - Unemployment Rate
 Freq: Monthly
 Tran: M/M, Y/Y
     
"""

# Homepage
_HOMEPAGE = "https://github.com/RileyTheEcon/SP500_Date_Offset"

# License is a mix of Public Domain and Creative Commons
# Sourcing the data so that it is all Public Domain is a longer term goal for
# this project
_LICENSE = ""

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://huggingface.co/datasets/rc9494/SP500_Date_Offset/dataset/"
_URLS = {
    "dev": _URL + "blob/main/SP500_Date_Offset.csv"
}
# =========================================================================== #





# ================================ FUNCTIONS ================================ #
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
                       series_name = 'value',
                       stale_data = 500
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
####
def check_data (dfFred, fred_key) :
    #   Check to make sure sufficient data is available
    df = pd.DataFrame() # create space in memory
    
    for i,r in dfFred[~dfFred['Freq'].isin(['Daily', 'Weekly'])].iterrows() :
        # Download data
        df = get_historic_data(r['SeriesID'], 
                               fred_key, 
                               r['Name']
                               )
        
        # Report series statistics
        print(r['Name'],'\n',
              'First Obs.: ', df.first_valid_index(), '\n',
              'Count Obs.: ', len(df), '\n',
              '\n'
              )
    #   end for i,r
#### / ####
def main(key_file_path,     # File path for FRED API key, txt
         fred_source_path,  # File path for variable names & FRED series ID, csv
         security_sym,      # Ticker symbol for security of interest (S&P 500)
         security_name,     # Name of security of interest
         export_path        # File path to save data
         ) :
    
    
    
    #   Seek API key; Prompt user if not found; access from repo if not given
    bDownload = False # Bool: Dl from repo or generate fresh?
                      # true = download pre-generated data from repo ; false = gen new
    
    try :   
        # try to get key from file
        with open(key_file_path, 'r') as file : 
            fred_key = file.read()
        #   endwith
    
    except FileNotFoundError :
        print('FRED api key not found!\n'+
              'Please enter api key or hit enter to download static dataset from repo:'
              )
        fred_key = input()
        
        if len(fred_key)==0 : bDownload = True
        else :
            pass # test validity of api key
        #   end if len
        
    except Exception as oops : print(f"Something odd happened: {oops}")
    #
    
    
    
    
    
    #   Import list of variables if it exists ; else download from repo
    if not bDownload :  # skip chunk if we're dl'ing from repo
        try : 
            # import list of variable to pull
            dfFred = pd.read_csv(fred_source_path)
            
        except FileNotFoundError :
            print('Could not find list of variables to generate: '+
                  fred_source_path+'\n'+
                  'Switching to download static dataset from repo instead!\n'
                  )
            bDownload = True
            
        #   end try/except
    #   end if bDownload
    
    #
    
    
    
    
    
    #   If above checks fail, then download from existing repo
    if bDownload :
        dfData = pd.read_csv('https://raw.githubusercontent.com/RileyTheEcon/'+
                             'SP500_Date_Offset/main/SP500_Offset.csv',
                             index_col='Date'
                             )
        
    #   If all above checks pass, generate fresh data from FRED api
    else :
        
        #   Download YFinance data
        dfFinance = yf.download(security_sym)['Adj Close']
        dfFinance.rename(security_name, inplace=True)
        #
        
        
        
        
        
        #   Iter thru data series; handle as specified
        dfEcon = pd.DataFrame() # make place in memory
        
        for i,r in dfFred.iterrows() :
            if not pd.notnull(r['SeriesID']) :    # skip if info missing
                continue
            #   end if
            
            # Create space in memory
            df = pd.DataFrame()
            
            # Import data
            if r['Freq'] in ['Daily', 'Weekly'] :
                # Dl data for daily/ weekly freq
                df = get_fred_data(fred_key, 
                                   pd.DataFrame(r).T[['Name','SeriesID']]
                                   )
            
            else :
                # Dl data for daily/ weekly freq
                df = get_historic_data(r['SeriesID'], 
                                       fred_key
                                       )
                df.rename(columns = {'value': r['Name']},
                          inplace = True
                          )
                
                #   Indicate report date
                df[r['Name']+'_release'] = 1
            
            #   end if import
            
            #   Attach to full dataframe
            dfEcon = dfEcon.join(df, how='outer')
            
        #   end for iterrows
        #
        
        
        
        
        
        #   Combine & fill numeric vars & export
        # Ffill numeric vars & fillna(0) indicators
        # left append to stock data
        dfData = (pd.DataFrame(dfFinance)
                  .join(dfEcon[[x for x in dfEcon.columns 
                                if len(dfEcon[x].unique())>3]
                               ].ffill(),
                        how='left'
                        )
                  .join(dfEcon[[x for x in dfEcon.columns 
                                if len(dfEcon[x].unique())<=3]
                               ].fillna(0),
                        how='left'
                        )
                  )
        
        # Export
        if len(export_path)>0 :
            dfData.to_csv(export_path)
        #   end if
        #
        
    #   end if bDownload
    
    return dfData
    #
    
####
class SP500_Date_Offset(datasets.GeneratorBasedBuilder):
    """ . """

    _URLS = _URLS
    VERSION = datasets.Version("1.1.0")

    def _info(self):
      raise ValueError('woops!')
      return datasets.DatasetInfo(
          description=_DESCRIPTION,
          features=datasets.Features(
              {
                  "Date": datasets.Value("datetime"),
                  "SP500": datasets.Value("float"),
                  "Fed-Rate": datasets.Value("float"),
                  "Yield-10Y": datasets.Value("float"),
                  "Yield-1M": datasets.Value("float"),
                  "Yield-1Y": datasets.Value("float"),
                  "Yield-20Y": datasets.Value("float"),
                  "Yield-2Y": datasets.Value("float"),
                  "Yield-30Y": datasets.Value("float"),
                  "Yield-3M": datasets.Value("float"),
                  "Yield-3Y": datasets.Value("float"),
                  "Yield-5Y": datasets.Value("float"),
                  "Yield-6M": datasets.Value("float"),
                  "Yield-7Y": datasets.Value("float"),
                  "Bus-Apps": datasets.Value("float"),
                  "Loans-CI": datasets.Value("float"),
                  "Loans-Cons": datasets.Value("float"),
                  "Loans-RE": datasets.Value("float"),
                  "Unemp-Claims": datasets.Value("float"),
                  "Con-Sentim": datasets.Value("float"),
                  "Con-Sentim_release": datasets.Value("bool"),
                  "Con-Spends": datasets.Value("float"),
                  "Con-Spends_release": datasets.Value("bool"),
                  "CPI": datasets.Value("float"),
                  "CPI_release": datasets.Value("bool"),
                  "CPI-Core": datasets.Value("float"),
                  "CPI-Core_release": datasets.Value("bool"),
                  "CPI-Services": datasets.Value("float"),
                  "CPI-Services_release": datasets.Value("bool"),
                  "Home-Sales": datasets.Value("float"),
                  "Home-Sales_release": datasets.Value("bool"),
                  "Home-Starts": datasets.Value("float"),
                  "Home-Starts_release": datasets.Value("bool"),
                  "Income-Trans": datasets.Value("float"),
                  "Income-Trans_release": datasets.Value("bool"),
                  "Indust-Prod": datasets.Value("float"),
                  "Indust-Prod_release": datasets.Value("bool"),
                  "Inventory-Sales": datasets.Value("float"),
                  "Inventory-Sales_release": datasets.Value("bool"),
                  "Manu-Hours": datasets.Value("float"),
                  "Manu-Hours_release": datasets.Value("bool"),
                  "MT-Sales": datasets.Value("float"),
                  "MT-Sales_release": datasets.Value("bool"),
                  "NO-Capital": datasets.Value("float"),
                  "NO-Capital_release": datasets.Value("bool"),
                  "NO-Consumer": datasets.Value("float"),
                  "NO-Consumer_release": datasets.Value("bool"),
                  "NO-Durables": datasets.Value("float"),
                  "NO-Durables_release": datasets.Value("bool"),
                  "NO-Unfilled": datasets.Value("float"),
                  "NO-Unfilled_release": datasets.Value("bool"),
                  "PCE": datasets.Value("float"),
                  "PCE_release": datasets.Value("bool"),
                  "PCE-Core": datasets.Value("float"),
                  "PCE-Core_release": datasets.Value("bool"),
                  "PPI-Architect": datasets.Value("float"),
                  "PPI-Architect_release": datasets.Value("bool"),
                  "Total-Emp": datasets.Value("float"),
                  "Total-Emp_release": datasets.Value("bool"),
                  "Unemploy": datasets.Value("float"),
                  "Unemploy_release": datasets.Value("bool"),
                  "Unemp-Weeks": datasets.Value("float"),
                  "Unemp-Weeks_release": datasets.Value("bool"),
                  "Delinq-CreditC": datasets.Value("float"),
                  "Delinq-CreditC_release": datasets.Value("bool"),
                  "GDP": datasets.Value("float"),
                  "GDP_release": datasets.Value("bool"),
              }
          ),
          # No default supervised_keys (as we have to pass both question
          # and context as input).
          supervised_keys=None,
          homepage="https://github.com/RileyTheEcon/SP500_Date_Offset",
          citation=_CITATION,
      )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
      urls_to_download = self._URLS
      downloaded_files = dl_manager.download_and_extract(urls_to_download)

      return [
          datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]})
      ]
      
    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        
        
        dictArgs = {'key_file_path'    : 'fred_api_key.txt',      # set local directory
                    'fred_source_path' : 'fred.csv',              # set location of data dictionary
                    'security_sym'     : '^GSPC',                 # set security symbol
                    'security_name'    : 'SP500',                 # set security name
                    'export_path'      : 'SP500_Date_Offset.csv'  # set export destination
                    }
        
        dfData = main(**dictArgs)
        
        for i,r in dfData.iteritems() :
            # Features currently used are "context", "question", and "answers".
            # Others are extracted here for the ease of future expansions.
            yield i, {
                'Date': i,
                "SP500": r["SP500"],
                "Fed-Rate": r["Fed-Rate"],
                "Yield-10Y": r["Yield-10Y"],
                "Yield-1M": r["Yield-1M"],
                "Yield-1Y": r["Yield-1Y"],
                "Yield-20Y": r["Yield-20Y"],
                "Yield-2Y": r["Yield-2Y"],
                "Yield-30Y": r["Yield-30Y"],
                "Yield-3M": r["Yield-3M"],
                "Yield-3Y": r["Yield-3Y"],
                "Yield-5Y": r["Yield-5Y"],
                "Yield-6M": r["Yield-6M"],
                "Yield-7Y": r["Yield-7Y"],
                "Bus-Apps": r["Bus-Apps"],
                "Loans-CI": r["Loans-CI"],
                "Loans-Cons": r["Loans-Cons"],
                "Loans-RE": r["Loans-RE"],
                "Unemp-Claims": r["Unemp-Claims"],
                "Con-Sentim": r["Con-Sentim"],
                "Con-Sentim_release": r["Con-Sentim_release"],
                "Con-Spends": r["Con-Spends"],
                "Con-Spends_release": r["Con-Spends_release"],
                "CPI": r["CPI"],
                "CPI_release": r["CPI_release"],
                "CPI-Core": r["CPI-Core"],
                "CPI-Core_release": r["CPI-Core_release"],
                "CPI-Services": r["CPI-Services"],
                "CPI-Services_release": r["CPI-Services_release"],
                "Home-Sales": r["Home-Sales"],
                "Home-Sales_release": r["Home-Sales_release"],
                "Home-Starts": r["Home-Starts"],
                "Home-Starts_release": r["Home-Starts_release"],
                "Income-Trans": r["Income-Trans"],
                "Income-Trans_release": r["Income-Trans_release"],
                "Indust-Prod": r["Indust-Prod"],
                "Indust-Prod_release": r["Indust-Prod_release"],
                "Inventory-Sales": r["Inventory-Sales"],
                "Inventory-Sales_release": r["Inventory-Sales_release"],
                "Manu-Hours": r["Manu-Hours"],
                "Manu-Hours_release": r["Manu-Hours_release"],
                "MT-Sales": r["MT-Sales"],
                "MT-Sales_release": r["MT-Sales_release"],
                "NO-Capital": r["NO-Capital"],
                "NO-Capital_release": r["NO-Capital_release"],
                "NO-Consumer": r["NO-Consumer"],
                "NO-Consumer_release": r["NO-Consumer_release"],
                "NO-Durables": r["NO-Durables"],
                "NO-Durables_release": r["NO-Durables_release"],
                "NO-Unfilled": r["NO-Unfilled"],
                "NO-Unfilled_release": r["NO-Unfilled_release"],
                "PCE": r["PCE"],
                "PCE_release": r["PCE_release"],
                "PCE-Core": r["PCE-Core"],
                "PCE-Core_release": r["PCE-Core_release"],
                "PPI-Architect": r["PPI-Architect"],
                "PPI-Architect_release": r["PPI-Architect_release"],
                "Total-Emp": r["Total-Emp"],
                "Total-Emp_release": r["Total-Emp_release"],
                "Unemploy": r["Unemploy"],
                "Unemploy_release": r["Unemploy_release"],
                "Unemp-Weeks": r["Unemp-Weeks"],
                "Unemp-Weeks_release": r["Unemp-Weeks_release"],
                "Delinq-CreditC": r["Delinq-CreditC"],
                "Delinq-CreditC_release": r["Delinq-CreditC_release"],
                "GDP": r["GDP"],
                "GDP_release": r["GDP_release"],
            }
        #   end for
    #   end def
#   end class
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    print(__doc__)
    main(**dictArgs)
#   endif
# =========================================================================== #



''' DEBUG
key_file_path    = dictArgs['key_file_path']
fred_source_path = dictArgs['fred_source_path']
security_sym     = dictArgs['security_sym']
security_name    = dictArgs['security_name']
export_path      = dictArgs['export_path']
'''





