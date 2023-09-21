#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:58:48 2023

@author: andrewmcdill
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
sns.set()


#Import processed dataset
fish_mod = pd.read_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/Fish_Preprocessed.csv')


#Drop outliers
fish_mod = fish_mod.drop(fish_mod[fish_mod['Remove'] == 'Y'].index)
fish_mod = fish_mod.drop(['Remove'], axis=1)
fish_mod = fish_mod.reset_index(drop=True)


#Normalize
X = fish_mod.drop(['Weight','Species'], axis=1)
y = pd.DataFrame(fish_mod.loc[:, fish_mod.columns == 'Weight'])
scaler = preprocessing.MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns = X.columns)


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)


#Search grid for best KNN parameters
KNN_mod_grid = neighbors.KNeighborsRegressor()
param_grid = {'n_neighbors': np.arange(2, 21, 1)}
grid = GridSearchCV(KNN_mod_grid, param_grid, scoring='neg_mean_absolute_error').fit(X_train, y_train)
print(grid.best_params_)


#Create model on best K from loop
KNN_mod = neighbors.KNeighborsRegressor(n_neighbors=grid.best_params_['n_neighbors'])
KNN_mod.fit(X_train, y_train)
yhat_train = pd.DataFrame(KNN_mod.predict(X_train))
yhat_train.columns = ['Pred_Weight']
y_train = y_train.reset_index(drop=True)
eval_res = pd.concat([y_train, yhat_train], axis=1)
eval_res['Residuals'] = eval_res['Weight'] - eval_res['Pred_Weight']
eval_res.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/KNN_Residuals_Train.csv')
##Training metrics and visuals
sns.scatterplot(eval_res['Weight'], eval_res['Pred_Weight'])
#R2
r2 = KNN_mod.score(X_train, y_train)
#MSE/MAE
mse = mean_squared_error(eval_res['Weight'], eval_res['Pred_Weight'])
mae = mean_absolute_error(eval_res['Weight'], eval_res['Pred_Weight'])


#Verify model on test data
yhat_test = pd.DataFrame(KNN_mod.predict(X_test))
yhat_test.columns = ['Pred_Weight']
y_test = y_test.reset_index(drop=True)
eval_res_test = pd.concat([y_test, yhat_test], axis=1)
eval_res_test['Residuals'] = eval_res_test['Weight'] - eval_res_test['Pred_Weight']
eval_res_test.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/KNN_Residuals_Test.csv')
##Testing metrics and visuals
sns.scatterplot(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
#MSE/MAE
mse_test = mean_squared_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])
mae_test = mean_absolute_error(eval_res_test['Weight'], eval_res_test['Pred_Weight'])