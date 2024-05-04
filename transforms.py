# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:59:03 2024

@author: RC
"""





# ================================ LIBRARIES ================================ #
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from statsmodels.tsa.stattools import adfuller



dictArgs = {'dirDataSource'      : 'SP500_Date_Offset.csv', # dir for data source
            'dirTransformSource' : 'fred.csv',              # dir for data transforms guide
            'bDelNonStat'        : True,            # bool: delete non-stationary variables
            'bCharts'            : True,            # bool: generate charts
            'dirCharts'          : 'charts',        # dir for charts to be exported
            'dirExportData'      : 'SP500_transformed.csv' # dir for transformed data export
            }

# =========================================================================== #





# ================================ FUNCTIONS ================================ #
def get_time_delta (freq, gap) :
    '''
    Parameters
    ----------
    freq : STR
        Release frequency of variable
    gap : STR
        Desired delta. D, W, M, Q, Y.

    Returns
    -------
    Int : time-step to be used in .diff() or .pct_chage() methods
    '''
    
    delta = {'Daily'     : {'D' : 1,
                            'W' : 5,
                            'M' : 20,
                            'Q' : 65,
                            'Y' : 250
                            },
             'Weekly'    : {'W' : 1,
                            'M' : 4,
                            'Q' : 13,
                            'Y' : 52
                            },
             'Monthly'   : {'M' : 1,
                            'Q' : 3,
                            'Y' : 12
                            },
             
             'Quarterly' : {'Q' : 1,
                            'Y' : 4
                            }
             }
    
    return delta[freq][gap[0]]
####
def verify_dir (dirFile) :
    '''
        Given a file directory, this function checks if the relevant folder 
    exists. If it does not, then it creates the folder. Once done, it returns
    the original file directory.
    Intended use:
        df.to_csv(verify_dir('dir/that_might/not/exist.csv'))
    '''
    
    dirFolder = dirFile[:dirFile.rfind('/')]
    
    if ('/' in dirFile)&('..' != dirFolder) : # verify dir goes to a different folder
        if not os.path.exists(dirFolder) : os.makedirs(dirFolder)
    #   end verify
    
    return dirFile
####
def create_pca (df) :
    
    #Step1: We calculate the mean of the scaled data.
    df_normalized_mean = pd.DataFrame(scale(df))
    
    #Step2: Remove any missing data points
    df_normalized_mean.dropna(inplace=True)
    instruments = df.columns
    
    #Step3: We calculate the PCA using fit_transform
    pca = PCA(n_components=len(instruments))
    YC_PCA = pca.fit_transform(df_normalized_mean)
    
    #Step4: Create a SCREE plot to check the weights of each component
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    #
    
    
    #Here we look at the cumulative variance explained
    plt.plot(labels, pca.explained_variance_ratio_.cumsum()*100)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Cumulative Explained Variance');
    #
    
    #   Return
    return YC_PCA
    #
####
def test_stationarity (x) :     # return p-value for Dickey-Fuller test
    return adfuller(x)[1]
####
#### / ####
def main(dirDataSource,         # dir for data source
         dirTransformSource,    # dir for data transforms guide
         bDelNonStat,           # bool: delete non-stationary variables
         bCharts,               # bool: generate charts
         dirCharts,             # dir for charts to be exported
         dirExportData          # dir for transformed data export
         ) :
    
    
    
    #   Import existing data & transforms guide
    dfData = pd.read_csv(dirDataSource,
                         index_col='Date'
                         )
    
    dfFred = pd.read_csv(dirTransformSource)
    #
    
    
    
    
    
    #   Generate PCA vectors
    dictPCA = {'unrestri' : ['Fed-Rate', 'Yield-1M',  'Yield-3M',  'Yield-6M', 
                             'Yield-1Y', 'Yield-2Y',  'Yield-3Y',  'Yield-5Y', 
                             'Yield-7Y', 'Yield-10Y', 'Yield-20Y', 'Yield-30Y'
                             ],
               'restrict' : ['Fed-Rate', 'Yield-1Y', 'Yield-3Y', 
                             'Yield-5Y', 'Yield-10Y'
                             ]
               }
    
    for key in dictPCA.keys() :
        
        #   Create subset
        df = dfData[dictPCA[key]]
        df = df[df.notnull().all(axis=1)]
        #
        
        #   Create PCA
        dfPCA = pd.DataFrame(create_pca(df)[:,:3],
                             columns=['PCA_'+key+'_1',
                                      'PCA_'+key+'_2',
                                      'PCA_'+key+'_3'
                                      ]
                             )
        dfPCA.index = df.index
        #
        
        #   Add PCA to dataframe
        dfData = dfData.join(dfPCA)
        #
        
        #   Add PCA vars to transforms guide (for experimental purposes)
        for i in ['_1', '_2', '_3'] :
            dictRow = {'Name'  : ['PCA_'+key+i], 'Freq' : ['Daily'],
                       'As-Is' : ['Y'], 'D-D' : ['Y'], 'W-W' : ['Y'],
                       'M-M'   : ['Y'], 'Q-Q' : ['Y'], 'Y-Y' : ['Y']
                       }
            dfFred  = pd.concat([dfFred,
                                 pd.DataFrame(dictRow)
                                 ], ignore_index=True)
        #   end for i
        #
    #   end for key
    #
    
    
    
    
    
    #   Make transform from guide
    for i,r in dfFred.iterrows() :
        
        #   For variable, get list of transforms from guide frame
        listTrans = [k for k in r.keys() if r[k]=='Y']
        
        #   Iter thru transforms
        for gap in listTrans :
            
            #   Create subset
            if r['Freq'] == 'Daily' :
                df = dfData[r['Name']].copy()
            elif r['Freq'] == 'Weekly' :
                df = dfData[r['Name']].iloc[::5].copy()
            else :
                df = dfData[r['Name']][dfData[r['Name']+'_release']==1].copy()
            #   end if subset
            
            #   Difference
            if ('-' in gap)&(gap != 'As-Is') :
                dfData[r['Name']+'_'+gap
                       ] = (df.diff(get_time_delta(r['Freq'], gap))
                            )
                #   Ffill nans
                dfData[r['Name']+'_'+gap].ffill(inplace=True)
            
            #   Percent-change
            elif '/' in gap :
                dfData[r['Name']+'_'+gap
                       ] = (df.pct_change(get_time_delta(r['Freq'], gap))
                            )
                #   Ffill nans
                dfData[r['Name']+'_'+gap].ffill(inplace=True)
                
            #   end if gap
            
        #   end for gap
        #   
        
        #   Keep/Drop As-Is
        if not 'As-Is' in listTrans : del dfData[r['Name']]
        
    #   end for
    #
    
    
    
    
    
    #   Make Dickey-Fuller tests
    sig_level = 0.05
    
    intCount = len(dfData.columns)
    for var in [x for x in dfData.columns if len(dfData[x].unique())>3] :
        p_value = test_stationarity(dfData[var][dfData[var].notnull()])
        if p_value > sig_level :  
            print(f"Variable '{var}' is non-stationary (p-value = {p_value}). Dropping...")
            dfData.drop(columns=var, inplace=True)
        #   end if
    #   end for
    print('Number of columns dropped:', 
          intCount-len(dfData.columns),
          '\t;\t',
          'Number of columns remaining:',
          len(dfData.columns)
          )
    #
    
    
    
    
    
    #   Get lags
    
    #
    
    
    
    
    
    #   Generate plots
    dfData['day_release'] = ((dfData[[x for x in dfData.columns 
                                      if '_release' in x]]==1
                              ).any(axis=1)
                               .astype(int)
                               )
    
    if bCharts :
        dictVars = {'As-Is'    : [x for x in dfFred['Name'] if x in dfData.columns],
                    'Days'     : [x for x in dfData.columns if ('D-D' in x)|('D/D' in x)],
                    'Weeks'    : [x for x in dfData.columns if ('W-W' in x)|('W/W' in x)],
                    'Months'   : [x for x in dfData.columns if ('M-M' in x)|('M/M' in x)],
                    'Quarters' : [x for x in dfData.columns if ('Q-Q' in x)|('Q/Q' in x)],
                    'Years'    : [x for x in dfData.columns if ('Y-Y' in x)|('Y/Y' in x)]
                    }
        
        for vs in dictVars.keys() :
            shape_map = {0: 'o', 1: 's'}  # set marker shape
            sns.pairplot(dfData, 
                         vars=dictVars[vs], 
                         hue='day_release', 
                         palette='colorblind', 
                         markers=shape_map
                         ) # make scatter matrix
            
            if len(dirCharts)>0 :
                plt.savefig(verify_dir(dirCharts+'/'+vs+'.jpg'), dpi=300) # export
            #   end if
            plt.show()
        #   end for variables list
    #   end if bCharts
    #
    
    
    
    
    
    #   Export transformed and tested data
    if len(dirExportData)>0 :
        dfData.to_csv(verify_dir(dirExportData))
    #   end if export
    #
    
####
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    print(__doc__)
    main(**dictArgs)
#   endif
# =========================================================================== #



''' DEBUG
dirDataSource      = 'SP500_Date_Offset.csv' # dir for data source
dirTransformSource = 'fred.csv'              # dir for data transforms guide
bDelNonStat        = True            # bool: delete non-stationary variables
bCharts            = True            # bool: generate charts
dirCharts          = 'charts'        # dir for charts to be exported
dirExportData      = 'SP500_transformed.csv' # dir for transformed data export
'''





