# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:36:17 2023

@author: Aalok
"""


# load libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




#Setting working directory
path_f = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\Dockum_ET'
os.chdir(path_f)

#Reading the dataset
a = pd.read_csv('Well_Fil.csv')
b = a.copy()

# Arranging variables to perform regression
Y = a['Avg_Fl'] # strength is the Y variable


#Extracting selected columns to be used as X variables
X = a.iloc[:, 5:46]
#cols = [4,14,17,23,24,25,26,27,28,29,30,31]
#X = a[a.columns[cols]]
X

#Normalizing X
X = (X - X.min())/(X.max()-X.min())


#Converting Fluoride to indicator Variable
threshold = 2
b['Fl_Ind'] = np.where(a['Avg_Fl'] >= threshold, 1, 0)
Y = b['Fl_Ind']

#Splitting the data to training anad testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=99)

# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means no regularization

#Training the data
# fit the model with data
logreg.fit(X_train,y_train)

# Make Predictions on training data
y_pred=logreg.predict(X_train) # Make Predictions
yprob = logreg.predict_proba(X_train) #test output probabilities

# Get the parameters
params = logreg.get_params()
coef = logreg.coef_
intercpt = logreg.intercept_

# Create a confusion Matrix and plot it
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
# Make Predictions
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
y_pred_proba = logreg.predict_proba(X_train)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
auc = metrics.roc_auc_score(y_train, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()






###PREDICTING FOR FULL DATA
#logreg.fit(X_train,y_train)
pred_LR = logreg.predict(X) # Make Predictions
proba_LR = logreg.predict_proba(X)[::,1]

d = {'WellID':a.StateWellNumber, 'LatDD':a.LatDD, 'LonDD':a.LonDD, 'predLR':proba_LR}
final = pd.DataFrame(d)
final.to_csv('Preds.csv')
