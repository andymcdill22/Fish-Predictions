#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:14:34 2023

@author: andrewmcdill
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from xgboost import XGBRegressor
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
X = fish_mod.drop(['Weight','Species'], axis=1)
y = pd.DataFrame(fish_mod.loc[:, fish_mod.columns == 'Weight'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)


#GridSearch to find best parameters for XGB Regressor
regressor = XGBRegressor(eval_metric='rmse', subsample=0.7)
param_grid = {'max_depth': np.arange(1, 16, 1),
              'n_estimators': np.arange(1, 21, 1),
              'learning_rate': np.arange(0, 1, 0.1)}
grid = GridSearchCV(regressor, param_grid).fit(X_train, y_train)
print(grid.best_params_)


#Create model on best number of trees and depth
xgb_mod = XGBRegressor(learning_rate = grid.best_params_['learning_rate'],
                       n_estimators = grid.best_params_['n_estimators'],
                       max_depth = grid.best_params_['max_depth'], 
                       subsample=0.5, colsample_bytree=0.6, seed=42)
xgb_mod.fit(X_train, y_train)
yhat_train = pd.DataFrame(xgb_mod.predict(X_train))
yhat_train.columns = ['Pred_Weight']
y_train = y_train.reset_index(drop=True)
eval_res = pd.concat([y_train, yhat_train], axis=1)
eval_res['Residuals'] = eval_res['Weight'] - eval_res['Pred_Weight'] 
eval_res.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/XGB_Residuals_Train.csv')
#Training visual
sns.scatterplot(eval_res['Weight'], eval_res['Pred_Weight'])
#R2
r2 = xgb_mod.score(X_train, y_train)
#MSE/MAE
mse = mean_squared_error(eval_res['Weight'], eval_res['Pred_Weight'])
mae = mean_absolute_error(eval_res['Weight'], eval_res['Pred_Weight'])


#Verify model on test data
yhat_test = pd.DataFrame(xgb_mod.predict(X_test))
yhat_test.columns = ['Pred_Weight']
y_test = y_test.reset_index(drop=True)
eval_res_test = pd.concat([y_test, yhat_test], axis=1)
eval_res_test['Residuals'] = eval_res_test['Weight'] - eval_res_test['Pred_Weight']
eval_res_test.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/XGB_Residuals_Test.csv')
##Testing metrics and visuals
sns.scatterplot(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
#MSE/MAE
mse_test = mean_squared_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
mae_test = mean_absolute_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])