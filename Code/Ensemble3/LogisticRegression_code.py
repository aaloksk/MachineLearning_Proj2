# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:40:41 2023

@author: deseyi
"""

# load libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

 
#Start by setting working directory
path_f = 'C:\\Users\\desey\\OneDrive\\Desktop\\MACHINE_LEARN\\Project_2'
os.chdir(path_f)

 

#read well data from working directory
a = pd.read_csv('Wells_Finals.csv')
#Make a copy of data a
b = a.copy()

 

###arrange data set into dependent and independent Y & X variable to fit model

Y = a['Avg_Fl'] # for data in Y

#Extracting certain columns to be used as X variables
#In X extract variables from columns 5-46 to use as the independent variable
#X = a.iloc[:, 5:46]

#To extract columns not in a sequence
cols = [5,7,10,16,19,22,28,36,37,39,42,44,45]
X = a[a.columns[cols]]

#Normailze X
X = (X - X.min())/(X.max()-X.min())

 


####CORRELATION###
# Compute correlations and plot correlation matrix
corr = X.corr() #compute correlation

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

##############################################################

 
#Covert Avereage flouride from a continous variable to a decrete varable by creating a threshold
threshold = 2
b['Fl_Ind'] = np.where(a['Avg_Fl'] >= threshold, 1, 0)
Y = b['Fl_Ind']

 

#Splitting the data to training anad testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=99)


# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means no regularization

 
#####Training the data######

logreg.fit(X_train,y_train) # fit the model with data
y_pred=logreg.predict(X_train) # Make Prediction
yprob = logreg.predict_proba(X_train) #test output probabilities
zz = pd.DataFrame(yprob)  #Store prediction in zz

#zz.to_csv('nit_a.csv')

 

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

 

####Create a confusion Matrix and plot it####

cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

 

# Evaluate the model

acc = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)

 

print(f"Accuracy: {acc}")

print(f"Precision: {precision}")

print(f"Recall: {recall}")

print(f"F1 score: {f1}")

 

 

# ROC Curve
y_pred_proba = logreg.predict_proba(X_train)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
auc = metrics.roc_auc_score(y_train, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()


####TESTING DATA

y_pred=logreg.predict(X_test) # Make Predictions
yprob = logreg.predict_proba(X_test) #test output probabilities
zz = pd.DataFrame(yprob)
#zz.to_csv('nit_a.csv')

 

# Get the parameters

logreg.get_params()
logreg.coef_
logreg.intercept_

 

# Create a confusion Matrix and plot it

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)

disp.plot()

plt.show()

 

# Evaluate the model

acc = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

 

print(f"Accuracy: {acc}")

print(f"Precision: {precision}")

print(f"Recall: {recall}")

print(f"F1 score: {f1}")

 

 

# ROC Curve

y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))

plt.legend(loc=4)

plt.xlabel('1-Specificity')

plt.ylabel('Sensitivity')

plt.grid()

plt.show()



###

logreg.fit(X_train,y_train)

proba_LR = logreg.predict(X) # Make Predictions

proba_LR = logreg.predict_proba(X)[::,1]

 

d = {'StateWellNumber':a.StateWellNumber, 'LatDD':a.LatDD, 'LonDD':a.LonDD, 'predLR':proba_LR}

final = pd.DataFrame(d)

final.to_csv('Preds.csv', index=False)