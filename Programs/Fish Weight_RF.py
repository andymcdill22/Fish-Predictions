#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:14:34 2023

@author: andrewmcdill
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
sns.set()


#Import processed dataset
fish_mod = pd.read_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/Fish_Preprocessed.csv')


#Keep outliers but remove incorrect data point
fish_mod = fish_mod.drop(fish_mod[fish_mod['Weight'] == 0].index)
fish_mod = fish_mod.drop(['Remove'], axis=1)
fish_mod = fish_mod.reset_index(drop=True)


#Train/Test Split
X = fish_mod.drop(['Weight', 'Species'], axis=1)
y = pd.DataFrame(fish_mod.loc[:, fish_mod.columns == 'Weight'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)


#Grid Search to find the best parameters for RandomForestRegressor
regressor = RandomForestRegressor(random_state=42, max_features=8, max_leaf_nodes=45)
param_grid = {'max_depth': np.arange(1, 16, 1),
              'n_estimators': np.arange(1, 21, 1)}
grid = GridSearchCV(regressor, param_grid).fit(X_train, y_train)
print(grid.best_params_)


#Create model with best parameters
rf_mod = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'],
                               max_depth = grid.best_params_['max_depth'],
                               max_features=8, max_leaf_nodes=45, 
                               random_state=42)
rf_mod.fit(X_train, y_train)
yhat_train = pd.DataFrame(rf_mod.predict(X_train))
yhat_train.columns = ['Pred_Weight']
y_train = y_train.reset_index(drop=True)
eval_res = pd.concat([y_train, yhat_train], axis=1)
eval_res['Residuals'] = eval_res['Weight'] - eval_res['Pred_Weight'] 
eval_res.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/RF_Residuals_Train.csv')
#Training visual
sns.scatterplot(eval_res['Weight'], eval_res['Pred_Weight'])
#R2
rf_mod.score(X_train, y_train)
#MSE/MAE
mse = mean_squared_error(eval_res['Weight'], eval_res['Pred_Weight'])
mae = mean_absolute_error(eval_res['Weight'], eval_res['Pred_Weight'])


#Verify model on test data
yhat_test = pd.DataFrame(rf_mod.predict(X_test))
yhat_test.columns = ['Pred_Weight']
y_test = y_test.reset_index(drop=True)
eval_res_test = pd.concat([y_test, yhat_test], axis=1)
eval_res_test['Residuals'] = eval_res_test['Weight'] - eval_res_test['Pred_Weight']
eval_res_test.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/RF_Residuals_Test.csv')
##Testing metrics and visuals
sns.scatterplot(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
#MSE/MAE
mse_test = mean_squared_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
mae_test = mean_absolute_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])