# -*- coding: utf-8 -*-
"""
Fish weight and species predictions
 
Exploratory Data Analysis 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


fish = pd.read_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Fish.csv')


fish.describe()
fish.info()
fish.isnull().sum()


#Distribution of Weight
figure, axes = plt.subplots(2, 1, figsize=(10,8))
sns.boxplot(data=fish, x=fish['Weight'], width=0.45, palette="pastel", fliersize=8, ax=axes[0]).set(xlabel=None)
sns.histplot(fish['Weight'], ax=axes[1])
plt.xlabel('Weight (g)', fontsize=14, fontweight='bold')
figure.suptitle("Fish Weight Distribution", fontsize=20, fontweight='bold')
##Weight is skewed with 3 outliers


#Distribution of Lengths
plt.title("Fish Length Distribution")
sns.boxplot(data=fish[['Length1','Length2','Length3']], palette="pastel", fliersize=5).set_xticklabels(['Length 1 (cm)', 'Length 2 (cm)', 'Length 3 (cm)'])
##Lengths have a couple outliers


#Distribution of Height and Width
plt.title("Fish Height Distribution")
sns.boxplot(data=fish[['Height','Width']], palette='pastel', fliersize=5).set_xticklabels(['Height (cm)','Width (cm)'])
##No outliers


#Distribution of Species
plt.title("Species Frequency Count")
sns.histplot(fish['Species'], stat='probability')
#Unbalanced


#Compare categorical relationships
figure, axes = plt.subplots(2, 2, figsize=(12,8))
sns.boxplot(data=fish, x='Species', y='Weight', palette="pastel", ax=axes[0,0])
sns.boxplot(data=fish, x='Species', y='Length', palette="pastel", ax=axes[0,1])
sns.boxplot(data=fish, x='Species', y='Height', palette="pastel", ax=axes[1,0])
sns.boxplot(data=fish, x='Species', y='Width', palette="pastel", ax=axes[1,1])
figure.suptitle("Continous Distributions by Species", fontsize=20, fontweight='bold')


#One hot encode categorical variable
fish_species = fish['Species']
fish = pd.get_dummies(data=fish, drop_first=True)
fish = pd.concat([fish, fish_species], axis=1)


#Explore outliers and data anomalies
outliers = fish[(fish['Weight'] > 1500)|(fish['Weight'] == 0)]
#Remove outliers and incorrect data point
outliers['Remove'] = 'Y'
outliers = pd.DataFrame(outliers.loc[:, outliers.columns == 'Remove'])
fish = pd.merge(fish, outliers, left_index=True, right_index=True, how='outer')


#Analyze Relationships
pair = sns.pairplot(fish[['Weight','Height','Length','Width']], 
                    aspect=1.75, palette='hls', diag_kind='hist')
pair.fig.suptitle('Scatter Plot Matrix', fontsize=16, fontweight='bold', y=1.02)
#Detect Correlations
plt.title("Correlation Matrix", fontsize=16, fontweight='bold')
sns.heatmap(fish[['Weight','Length','Height','Width']].corr(), annot=True, square=True, fmt='.2f', cmap="Spectral_r")


#Export dataset
fish.to_csv(r'/Users/andrewmcdill/Documents/Data Science/Fish/Data/Outputs/Fish_Preprocessed.csv', index=False)
