# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:24:27 2023

@author: Aalok
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

#Path for working directory
path_f = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\Dockum_ET'
os.chdir(path_f) #Set working directory

##Reading the dataset
a = pd.read_csv('Well_Fil.csv')


#Fluoride
Y = a['Avg_Fl']

#Potential Independent Variables
X = a.iloc[:, 5:46]



##############################################################
########################CORRELATION###########################
# Compute correlations and plot correlation matrix
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

#sns.set(font_scale=0.4)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('corr.png', dpi=300, bbox_inches='tight') # Export the figure as a PNG image
##############################################################

#Correlation of each X Variable with Avg_Fluoride
X2 = X.copy() #Making copy of X
X2['Fl'] = a.Avg_Fl #Adding Fluoride
corr2 = X2.corr() #Getting correlation

last_row = len(corr2) # Accessing the last row
corr2 = corr2.drop(corr2.index[last_row-1]) # Deleting the last row

fig, ax = plt.subplots() # Create a figure and axis object
ax.bar(corr2.index, abs(corr2['Fl']), edgecolor='black') # Create a vertical bar plot with the data
ax.set_xticklabels(corr2.index, rotation=90, fontsize=7) # Set the x-axis labels to be the angles, with 90 degrees rotated
ax.set_xlabel('Parameters', fontsize=10)
ax.set_ylabel('Correlation with Avg_Fl')
ax.grid(color='gray', linestyle='-', linewidth=0.2) # Add a grid to the plot
plt.savefig('bar_graph.png', dpi=300, bbox_inches='tight') # Export the figure as a PNG image
plt.show()
##############################################################
##############################################################


##############################################################
#######################MULTICOLLINEARITY######################
# Compute Variance Inflation Factors
#VIF is measure of the amount of multicollinearity
#A factor with a VIF higher than 10 indicates a problem of multicollinearity existed (Dou et al.2019).
#https://www.tandfonline.com/doi/full/10.1080/17538947.2020.1718785
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
##############################################################



##############################################################
###################FEATURE IMPORTANCE#########################
#Performing random forest regressor for feature importance
sns.set(font_scale=0.4)
rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X, Y)
rf.feature_importances_
plt.barh(X.columns, rf.feature_importances_)
plt.grid()
plt.xlabel('Importance')
plt.savefig('RF_Imp.png', dpi=300, bbox_inches='tight') # Export the figure as a PNG image
plt.plot()

d={'X':X.columns, 'RF_Imp':rf.feature_importances_}
RF_imp = pd.DataFrame(d)
RF_imp
##############################################################
##############################################################



#Feature Importance by CART METHOD
# Fit a decision tree regressor using the CART algorithm
tree = DecisionTreeRegressor(random_state=0)
tree.fit(X, Y)#Actual Fitting
importances = tree.feature_importances_ # Calculate feature importances

plt.barh(X.columns, importances)
plt.grid()
plt.xlabel('Importance')
plt.savefig('CART_Imp.png', dpi=300, bbox_inches='tight') # Export the figure as a PNG image
plt.plot()



###############################################################



from sklearn.linear_model import Lasso, Ridge, ElasticNet

# Create Lasso, Ridge, and ElasticNet models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=0.1)
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Fit the models on the dataset
lasso.fit(X, Y)
ridge.fit(X, Y)
enet.fit(X, Y)

# Print the coefficients of each model
print("Lasso coefficients: ", lasso.coef_)
print("Ridge coefficients: ", ridge.coef_)
print("ElasticNet coefficients: ", enet.coef_)

zz = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(lasso.coef_), pd.DataFrame(ridge.coef_),pd.DataFrame(enet.coef_) ], axis=1)


##Lasso, Ridge, and Elastic Net are regularization techniques used in linear regression to address the problem of overfitting, 
#where the model fits the training data too closely and does not generalize well to new data.

#Here is a brief overview of each technique:

#    Lasso: Lasso stands for Least Absolute Shrinkage and Selection Operator. It is a regularization technique that adds a penalty term to the 
    #sum of squared errors (SSE) of the regression model, which shrinks the coefficients of less important features to zero. This results in a 
    #sparse model with only the most important features retained.

#    Ridge: Ridge regression adds a penalty term to the SSE of the regression model, but instead of shrinking the coefficients to zero like 
    #Lasso, it shrinks them towards zero. This helps to reduce the impact of multicollinearity in the data and can result in a more stable model.

#    Elastic Net: Elastic Net is a combination of Lasso and Ridge regression, where the regularization term is a weighted combination of 
     #both L1 and L2 penalties. This results in a model that is both sparse and can handle multicollinearity.

#In general, Lasso is a good choice when you have a large number of features and want to identify the most important features, Ridge is a 
#good choice when multicollinearity is a concern, and Elastic Net is a good choice when you want to balance between Lasso and Ridge regularization.

#These techniques are implemented in many popular machine learning libraries in Python, such as scikit-learn and statsmodels.