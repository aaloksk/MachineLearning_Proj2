# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:33:40 2023

@author: Aalok
"""



# load libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB






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




# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=99)

# Naive Bayes Model
clf = GaussianNB() #create object
clf.fit(X_train,y_train)  # Fit the model
clf.predict(X_train) # Predict training data






# For training data
y_pred= clf.predict(X_train) # predict testing data
yprob = clf.predict_proba(X_train) #output probabilities

# Perform evaluation using contingency table
# Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_train, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(y_train, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(y_train, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(y_train, y_pred)) # predicting 1 (unsat)
print("F1:",metrics.f1_score(y_train, y_pred))

# ROC Curve
y_pred_proba = clf.predict_proba(X_train)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_train,  y_pred_proba)
auc = metrics.roc_auc_score(y_train, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.title('Receiver Operating Characteristics Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity') 
plt.grid() # Plot the grid
plt.show() #show the curve





# Predict testing data
y_pred= clf.predict(X_test) # predict testing data
yprob = clf.predict_proba(X_test) #output probabilities

# Perform evaluation using contingency table
# Create a confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(y_test, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(y_test, y_pred)) # predicting 1 (unsat)
print("F1:",metrics.f1_score(y_test, y_pred))

# ROC Curve
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.title('Receiver Operating Characteristics Curve')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity') 
plt.grid() # Plot the grid
plt.show() #show the curve










#Storing Probability for KNN Model
#Setting working directory
path_f = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\Dockum_ET'
os.chdir(path_f)

final = pd.read_csv('Preds.csv')

y_pred = clf.predict(X)
y_predNB = clf.predict_proba(X)

final['pred_NB'] = y_predNB[:,1]

final.to_csv('Preds.csv')
