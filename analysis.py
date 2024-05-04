# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:59:23 2024

@author: RC
"""






# ================================ LIBRARIES ================================ #
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from multiprocessing import cpu_count # 6 max

import statsmodels.api as sm 
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



dictArgs = {'data_source'   : 'SP500_transformed.csv', #
            'k_fold_cv'     : 5, # 
            'output_name'   : 'analysis_log.csv', #
            'use_cores'     : 0, # 1=regular ; 0=all ; else=# of cores to use in mp
            'set_seed'      : 20240424
            }

# =========================================================================== #





# ================================ FUNCTIONS ================================ #

#### / ####
def main(data_source,
         k_fold_cv,
         output_name,
         use_cores,
         set_seed
         ) :
    
    
    
    #   Import data
    dfData = pd.read_csv(data_source, index_col='Date')
    
    dictMem = dict()
    #
    
    
    
    
    
    #   Set variable lists
    contin_vars = [x for x in dfData.columns 
                   if ('SP500' in x)&(not '_l' in x)]
    catego_vars = [x for x in dfData.columns 
                   if ('Recession' in x)&(not '_l' in x)]
    predictors  = [x for x in dfData.columns
                   if (x not in contin_vars+catego_vars)&
                      (not '_release' in x)
                   ]
    #
    
    
    
    
    
    #   Set cores
    if use_cores==0 : n_jobs = cpu_count()
    else : n_jobs = use_cores
    #
    
    
    
    
    
    ####    Recession Binary Models
    df = dfData[dfData.notnull().all(axis=1)].copy()
    for Y in catego_vars :
        
        ####    Binom LASSO
        # Set random seed, CV k-folds & hyperparameter space
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        alphas = np.logspace(-5, 1, 100)  
        
        # Set model & grid_search objects
        modelLasso  = LogisticRegression(penalty='l1', 
                                         solver='liblinear', 
                                         max_iter=1000
                                         )
        grid_search = GridSearchCV(estimator=modelLasso, 
                                   param_grid={'C': 1/alphas}, 
                                   scoring='neg_log_loss', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Fit grid_search ; find best hyper-param
        grid_search.fit(sm.add_constant(df[predictors]), 
                        df[Y])
        
        # 
        # Get the best LASSO model
        best_model = grid_search.best_estimator_
        coefficients = best_model.coef_     # Retrieve the coefficients
        variable_importance = dict(zip(predictors, 
                                       coefficients[0])
                                   )
        sorted_variable_importance = sorted(variable_importance.items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True
                                            )
        
        # Print variable importance
        dictMem[Y+'_lasso_alpha'] = grid_search.best_params_['C']
        dictMem[Y+'_lasso_ll'] = -grid_search.best_score_
        dictMem[Y+'_lasso_vars'] = sorted_variable_importance
        
        print(Y+': LASSO:\n',
              'Best alpha:  ',dictMem[Y+'_lasso_alpha'],'\n'
              'Best l-loss: ',dictMem[Y+'_lasso_ll'],'\n',
              sep=''
              )
        print("Variable Importance (absolute coefficients):")
        for var, coef in sorted_variable_importance:
            print(f"{var}: {coef}")
        
        plt.barh(*zip(*sorted_variable_importance))
        plt.xlabel('Absolute Coefficient')
        plt.ylabel('Variable')
        plt.title(Y+': Variable Importance (absolute coefficients)')
        plt.show()
        #
        
        
        
        
        
        ####    Binom E.Net
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        alphas = np.logspace(-5, 1, 100)  
        l1_ratios = np.linspace(0.1, 0.9, 9)
        
        # Create Elastic Net logistic regression model
        modelElastic = LogisticRegression(penalty='elasticnet', 
                                          solver='saga', 
                                          max_iter=1000
                                          )
        
        # Set up grid search with cross-validation
        param_grid = {'C'        : 1/alphas,
                      'l1_ratio' : l1_ratios
                      }
        grid_search = GridSearchCV(estimator=modelElastic, 
                                   param_grid=param_grid, 
                                   scoring='neg_log_loss', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Perform grid search to find best alpha and l1_ratio
        grid_search.fit(sm.add_constant(df[predictors]), 
                        df[Y])
        
        # Retrieve the best Elastic Net model
        best_model_elastic_net = grid_search.best_estimator_
        coefficients_elastic_net = best_model_elastic_net.coef_
        variable_importance_elastic_net = dict(zip(predictors, 
                                                   coefficients_elastic_net[0])
                                               )
        
        sorted_variable_importance_elastic_net = sorted(variable_importance_elastic_net.items(), 
                                                        key=lambda x: abs(x[1]), 
                                                        reverse=True
                                                        )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_elast_alpha'] = grid_search.best_params_['C']
        dictMem[Y+'_elast_l1rat'] = grid_search.best_params_['l1_ratio']
        dictMem[Y+'_elast_ll']    = -grid_search.best_score_
        dictMem[Y+'_elast_vars'] = sorted_variable_importance_elastic_net
        
        print(Y+': ELASTIC-NET:\n',
              'Best alpha:  ',dictMem[Y+'_elast_alpha'],'\n',
              'Best L1:     ',dictMem[Y+'_elast_l1rat'],'\n',
              'Best l-loss: ',dictMem[Y+'_elast_ll'],'\n',
              sep=''
              )
        print("Variable Importance (absolute coefficients) for Elastic Net:")
        for var, coef in sorted_variable_importance_elastic_net:
            print(f"{var}: {coef}")
        #   end for
        
        plt.barh(*zip(*sorted_variable_importance_elastic_net))
        plt.xlabel('Absolute Coefficient')
        plt.ylabel('Variable')
        plt.title('Variable Importance (absolute coefficients) for Elastic Net')
        plt.show()
        #
        
        
        
        
        
        ####    Tree
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        param_grid = {'max_depth':         [None]+list(range(1,12+1)),
                      'min_samples_leaf' : [None]+list(range(1,12+1))
                      }
        
        # Create decision tree classifier
        modelTree = DecisionTreeClassifier()
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(estimator=modelTree, 
                                   param_grid=param_grid, 
                                   scoring='neg_log_loss', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        grid_search.fit(df[predictors], 
                        df[Y]
                        )
        
        
        # Retrieve the best decision tree model
        best_model_tree = grid_search.best_estimator_
        feature_importances_tree = best_model_tree.feature_importances_
        feature_importance_mapping = dict(zip(predictors, 
                                              feature_importances_tree)
                                          )
        sorted_feature_importance = sorted(feature_importance_mapping.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True
                                           )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_tree_depth']  = grid_search.best_params_['max_depth']
        dictMem[Y+'_tree_leaf']   = grid_search.best_params_['min_samples_leaf']
        dictMem[Y+'_tree_l-loss'] = -grid_search.best_score_
        dictMem[Y+'_tree_vars'] = sorted_feature_importance
        
        print(Y+': CLASSIFICATION TREE:\n',
              'Best Max-Depth:  ',dictMem[Y+'_tree_depth'],'\n',
              'Best Min-Sample: ',dictMem[Y+'_tree_leaf'],'\n',
              'Best l-loss:     ',dictMem[Y+'_tree_l-loss'],'\n',
              sep=''
              )
        
        # Print feature importances
        print("Feature Importance for Decision Tree:")
        for feature, importance in sorted_feature_importance:
            print(f"{feature}: {importance}")
        #   end for

        plt.barh(*zip(*sorted_feature_importance))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Decision Tree')
        plt.show()
        #
        
        
        
        
        
        ####    Random Forest
        # Example DataFrame with features (X) and target variable (y)
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        param_grid = {
            'n_estimators':     list(range(50,400,50)),
            'max_depth':        [None]+list(range(1,12+1)),
            'min_samples_leaf': [None]+list(range(1,12+1))
        }
        
        # Create Random Forest classifier
        modelForest = RandomForestClassifier()
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(estimator=modelForest, 
                                   param_grid=param_grid, 
                                   scoring='neg_log_loss', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Perform grid search to find best hyperparameters
        grid_search.fit(df[predictors], 
                        df[Y])
        
        
        # Retrieve the best Random Forest model
        best_model_forest = grid_search.best_estimator_
        feature_importances_forest = best_model_forest.feature_importances_
        feature_importance_mapping_forest = dict(zip(predictors, 
                                                     feature_importances_forest)
                                                 )
        sorted_feature_importance_forest = sorted(feature_importance_mapping_forest.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True
                                                  )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_forest_n']      = grid_search.best_params_['n_estimators']
        dictMem[Y+'_forest_depth']  = grid_search.best_params_['max_depth']
        dictMem[Y+'_forest_leaf']   = grid_search.best_params_['min_samples_leaf']
        dictMem[Y+'_forest_l-loss'] = -grid_search.best_score_
        dictMem[Y+'_forest_vars'] = sorted_feature_importance_forest
        
        print(Y+': CLASSIFICATION FOREST:\n',
              'Best Forest-Size: ',dictMem['forest_n'],'\n',
              'Best Max-Depth:   ',dictMem['forest_depth'],'\n',
              'Best Min-Sample:  ',dictMem['forest_leaf'],'\n',
              'Best l-loss:      ',dictMem['forest_l-loss'],'\n',
              sep=''
              )
        print("Feature Importance for Random Forest:")
        for feature, importance in sorted_feature_importance_forest:
            print(f"{feature}: {importance}")
        #   end for
        
        plt.barh(*zip(*sorted_feature_importance_forest))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Random Forest')
        plt.show()
        #
    #   end for Y
    
    
    
    
    
    ####    SP500 Price Change Models
    df = dfData[dfData.notnull().all(axis=1)].copy()
    for Y in contin_vars :
        
        ####    LASSO
        # Set random seed, CV k-folds & hyperparameter space
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        alphas = np.logspace(-5, 1, 100)  
        
        # Set model & grid_search objects
        modelLasso  = Lasso()
        grid_search = GridSearchCV(estimator=modelLasso, 
                                   param_grid={'alpha': alphas}, 
                                   scoring='neg_mean_squared_error', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Fit grid_search ; find best hyper-param
        grid_search.fit(sm.add_constant(df[predictors]), 
                        df[Y])
        
        # 
        # Get the best LASSO model
        best_model = grid_search.best_estimator_
        coefficients = best_model.coef_     # Retrieve the coefficients
        variable_importance = dict(zip(predictors, 
                                       coefficients)
                                   )
        sorted_variable_importance = sorted(variable_importance.items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True
                                            )
        
        # Print variable importance
        dictMem[Y+'_lasso_alpha'] = grid_search.best_params_['alpha']
        dictMem[Y+'_lasso_mse'] = -grid_search.best_score_
        dictMem[Y+'_lasso_vars'] = sorted_variable_importance
        
        print(Y+': LASSO:\n',
              'Best alpha:  ',dictMem[Y+'_lasso_alpha'],'\n'
              'Best l-loss: ',dictMem[Y+'_lasso_mse'],'\n',
              sep=''
              )
        print("Variable Importance (absolute coefficients):")
        for var, coef in sorted_variable_importance:
            print(f"{var}: {coef}")
        
        plt.barh(*zip(*sorted_variable_importance))
        plt.xlabel('Absolute Coefficient')
        plt.ylabel('Variable')
        plt.title(Y+': Variable Importance (absolute coefficients)')
        plt.show()
        #
        
        
        
        
        
        ####    Binom E.Net
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        alphas = np.logspace(-5, 1, 100)  
        l1_ratios = np.linspace(0.1, 0.9, 9)
        
        # Create Elastic Net logistic regression model
        modelElastic = ElasticNet(max_iter=1000)
        
        # Set up grid search with cross-validation
        param_grid = {'alpha'    : 1/alphas,
                      'l1_ratio' : l1_ratios
                      }
        grid_search = GridSearchCV(estimator=modelElastic, 
                                   param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Perform grid search to find best alpha and l1_ratio
        grid_search.fit(sm.add_constant(df[predictors]), 
                        df[Y])
        
        # Retrieve the best Elastic Net model
        best_model_elastic_net = grid_search.best_estimator_
        coefficients_elastic_net = best_model_elastic_net.coef_
        variable_importance_elastic_net = dict(zip(predictors, 
                                                   coefficients_elastic_net)
                                               )
        
        sorted_variable_importance_elastic_net = sorted(variable_importance_elastic_net.items(), 
                                                        key=lambda x: abs(x[1]), 
                                                        reverse=True
                                                        )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_elast_alpha'] = grid_search.best_params_['alpha']
        dictMem[Y+'_elast_l1rat'] = grid_search.best_params_['l1_ratio']
        dictMem[Y+'_elast_mse']    = -grid_search.best_score_
        dictMem[Y+'_elast_vars'] = sorted_variable_importance_elastic_net
        
        print(Y+': ELASTIC-NET:\n',
              'Best alpha:  ',dictMem[Y+'_elast_alpha'],'\n',
              'Best L1:     ',dictMem[Y+'_elast_l1rat'],'\n',
              'Best MSE: ',dictMem[Y+'_elast_mse'],'\n',
              sep=''
              )
        print("Variable Importance (absolute coefficients) for Elastic Net:")
        for var, coef in sorted_variable_importance_elastic_net:
            print(f"{var}: {coef}")
        #   end for
        
        plt.barh(*zip(*sorted_variable_importance_elastic_net))
        plt.xlabel('Absolute Coefficient')
        plt.ylabel('Variable')
        plt.title('Variable Importance (absolute coefficients) for Elastic Net')
        plt.show()
        #
        
        
        
        
        
        ####    Tree
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        param_grid = {'max_depth':         [None]+list(range(1,12+1)),
                      'min_samples_leaf' : [None]+list(range(1,12+1))
                      }
        
        # Create decision tree classifier
        modelTree = DecisionTreeRegressor()
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(estimator=modelTree, 
                                   param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        grid_search.fit(df[predictors], 
                        df[Y]
                        )
        
        
        # Retrieve the best decision tree model
        best_model_tree = grid_search.best_estimator_
        feature_importances_tree = best_model_tree.feature_importances_
        feature_importance_mapping = dict(zip(predictors, 
                                              feature_importances_tree)
                                          )
        sorted_feature_importance = sorted(feature_importance_mapping.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True
                                           )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_tree_depth']  = grid_search.best_params_['max_depth']
        dictMem[Y+'_tree_leaf']   = grid_search.best_params_['min_samples_leaf']
        dictMem[Y+'_tree_mse'] = -grid_search.best_score_
        dictMem[Y+'_tree_vars'] = sorted_feature_importance
        
        print(Y+': CLASSIFICATION TREE:\n',
              'Best Max-Depth:  ',dictMem[Y+'_tree_depth'],'\n',
              'Best Min-Sample: ',dictMem[Y+'_tree_leaf'],'\n',
              'Best mse:     ',dictMem[Y+'_tree_mse'],'\n',
              sep=''
              )
        
        # Print feature importances
        print("Feature Importance for Decision Tree:")
        for feature, importance in sorted_feature_importance:
            print(f"{feature}: {importance}")
        #   end for

        plt.barh(*zip(*sorted_feature_importance))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Decision Tree')
        plt.show()
        #
        
        
        
        
        
        ####    Random Forest
        # Example DataFrame with features (X) and target variable (y)
        np.random.seed(set_seed)
        k_folds = KFold(n_splits=k_fold_cv, shuffle=True)
        param_grid = {
            'n_estimators':     list(range(50,400,50)),
            'max_depth':        [None]+list(range(1,12+1)),
            'min_samples_leaf': [None]+list(range(1,12+1))
        }
        
        # Create Random Forest classifier
        modelForest = RandomForestRegressor()
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(estimator=modelForest, 
                                   param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', 
                                   cv=k_folds,
                                   n_jobs=n_jobs
                                   )
        
        # Perform grid search to find best hyperparameters
        grid_search.fit(df[predictors], 
                        df[Y])
        
        
        # Retrieve the best Random Forest model
        best_model_forest = grid_search.best_estimator_
        feature_importances_forest = best_model_forest.feature_importances_
        feature_importance_mapping_forest = dict(zip(predictors, 
                                                     feature_importances_forest)
                                                 )
        sorted_feature_importance_forest = sorted(feature_importance_mapping_forest.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True
                                                  )
        
        # Get the best hyperparameters and corresponding log loss score
        dictMem[Y+'_forest_n']      = grid_search.best_params_['n_estimators']
        dictMem[Y+'_forest_depth']  = grid_search.best_params_['max_depth']
        dictMem[Y+'_forest_leaf']   = grid_search.best_params_['min_samples_leaf']
        dictMem[Y+'_forest_mse'] = -grid_search.best_score_
        dictMem[Y+'_forest_vars'] = sorted_feature_importance_forest
        
        print(Y+': CLASSIFICATION FOREST:\n',
              'Best Forest-Size: ',dictMem[Y+'_forest_n'],'\n',
              'Best Max-Depth:   ',dictMem[Y+'_forest_depth'],'\n',
              'Best Min-Sample:  ',dictMem[Y+'_forest_leaf'],'\n',
              'Best mse:      ',dictMem[Y+'_forest_mse'],'\n',
              sep=''
              )
        print("Feature Importance for Random Forest:")
        for feature, importance in sorted_feature_importance_forest:
            print(f"{feature}: {importance}")
        #   end for
        
        plt.barh(*zip(*sorted_feature_importance_forest))
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Random Forest')
        plt.show()
        #
        
    #   end for Y--continuous
    
####
# =========================================================================== #





# =================================== MAIN ================================== #
if __name__ == "__main__" :
    print(__doc__)
    main(**dictArgs)
#   endif
# =========================================================================== #



''' DEBUG
data_source   = 'SP500_transformed.csv' #
k_fold_cv     = 5 # 
output_name   = 'analysis_log.csv' #
use_cores     = 0 # 1=regular ; 0=all ; else=# of cores to use in mp
set_seed      = 20240424
'''





