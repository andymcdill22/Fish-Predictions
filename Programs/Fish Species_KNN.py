#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:46:47 2023

@author: andrewmcdill
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors
import imblearn
from sklearn.metrics import confusion_matrix, f1_score
from mlxtend.plotting import plot_confusion_matrix


#Import processed dataset
fish_mod = pd.read_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/Fish_Preprocessed.csv')


#Drop outliers
fish_mod = fish_mod.drop(fish_mod[fish_mod['Remove'] == 'Y'].index)
fish_mod = fish_mod.drop(['Remove'], axis=1)
fish_mod = fish_mod.reset_index(drop=True)


#Normalize
X = fish_mod.iloc[:, 0:6]
y = pd.DataFrame(fish_mod.loc[:, fish_mod.columns == 'Species'])
scaler = preprocessing.MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns = X.columns)


#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)


#Implement over sampling method SMOTE to account for imbalanced data 
oversample = imblearn.over_sampling.SMOTE(k_neighbors=2, random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)


#Search grid for best KNN parameters
KNN_mod_grid = neighbors.KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(2, 21, 1)}
grid = GridSearchCV(KNN_mod_grid, param_grid, scoring='f1_macro').fit(X_train, y_train)
print(grid.best_params_)


#Create and fit KNN model with best parameters
KNN_mod = neighbors.KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
KNN_mod.fit(X_train, y_train)
yhat_train = pd.DataFrame(KNN_mod.predict(X_train))
yhat_train.columns = ['Pred_Species']
eval_class = pd.concat([y_train, yhat_train], axis=1)
eval_class.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/KNN_Classification_Train.csv')
#Accuracy and F1-Score
f1_train = f1_score(eval_class['Species'], eval_class['Pred_Species'], average='micro')
#Confusion matrix
plot_confusion_matrix(confusion_matrix(eval_class['Species'], eval_class['Pred_Species']),
                      class_names=['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish'])


#Verify model on test data
yhat_test = pd.DataFrame(KNN_mod.predict(X_test))
yhat_test.columns = ['Pred_Species']
y_test = y_test.reset_index(drop=True)
eval_class_test = pd.concat([y_test, yhat_test], axis=1)
eval_class_test.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/KNN_Classification_Test.csv')
#Accuracy and F1-Score
f1_test = f1_score(eval_class_test['Species'], eval_class_test['Pred_Species'], average='macro')
#Confusion matrix
plot_confusion_matrix(confusion_matrix(eval_class_test['Species'], eval_class_test['Pred_Species']),
                      class_names=['Bream','Parkki','Perch','Pike','Roach','Smelt','Whitefish'])