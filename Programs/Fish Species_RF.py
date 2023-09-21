#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:31:33 2023

@author: andrewmcdill
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from mlxtend.plotting import plot_confusion_matrix
import imblearn


#Import dataset
fish_mod = pd.read_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/Fish_Preprocessed.csv')


#Keep outliers but remove incorrect data point
fish_mod = fish_mod.drop(fish_mod[fish_mod['Weight'] == 0].index)
fish_mod = fish_mod.drop(['Remove'], axis=1)
fish_mod = fish_mod.reset_index(drop=True)


#Split data into train and test
X = fish_mod.iloc[:, 0:6]
y = pd.DataFrame(fish_mod.loc[:, fish_mod.columns == 'Species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Implement over sampling method SMOTE to account for imbalanced data 
oversample = imblearn.over_sampling.SMOTE(k_neighbors=2, random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)


#Search grid for best RandomForestClassifier parameters
numlab = preprocessing.LabelEncoder()
y_train_num = pd.DataFrame(numlab.fit_transform(y_train), columns = ['Species_Num'])
rf_mod_grid = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': np.arange(1, 16, 1),
              'max_depth': np.arange(1, 16, 1),
              'max_features': np.arange(1, len(X_train.columns), 1)}
grid = GridSearchCV(rf_mod_grid, param_grid, scoring='f1_micro').fit(X_train, y_train_num)
print(grid.best_params_)


#Create and fit RandomForestClassifier model with best parameters
rf_mod = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'], 
                                max_depth=grid.best_params_['max_depth'],
                                max_features=grid.best_params_['max_features'],
                                random_state=42)
rf_mod.fit(X_train, y_train)
yhat_train = pd.DataFrame(rf_mod.predict(X_train))
yhat_train.columns = ['Pred_Species']
eval_class = pd.concat([y_train, yhat_train], axis=1)
eval_class.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/RF_Classification_Train.csv')
#Accuracy and F1-Score
f1_train = f1_score(eval_class['Species'], eval_class['Pred_Species'], average='micro')
#Confusion matrix
plot_confusion_matrix(confusion_matrix(eval_class['Species'], eval_class['Pred_Species']),
                      class_names=['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish'])


#Verify model on test data
yhat_test = pd.DataFrame(rf_mod.predict(X_test))
yhat_test.columns = ['Pred_Species']
y_test = y_test.reset_index(drop=True)
eval_class_test = pd.concat([y_test, yhat_test], axis=1)
eval_class_test.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/RF_Classification_Test.csv')
#Accuracy and F1-Score
f1_test = f1_score(eval_class_test['Species'], eval_class_test['Pred_Species'], average='macro')
#Confusion matrix
plot_confusion_matrix(confusion_matrix(eval_class_test['Species'], eval_class_test['Pred_Species']),
                      class_names=['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish'])