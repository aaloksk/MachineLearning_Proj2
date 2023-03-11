# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:24:09 2023

@author: Aalok
"""
import os
import pandas as pd

#Path for working directory
path_f = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\Dockum_ET'
os.chdir(path_f) #Set working directory

#Read the well attribute table extracted from QGIS
wells = pd.read_csv('Wells.csv')

#Dropping rows with NA values
wells2 = wells.dropna()




os.chdir('C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Proj1\\WD\\GWDBDownload\\GWDBDownload')
df1 = pd.read_csv('WellMain.txt', sep= '|', encoding= 'unicode_escape')



Final_df = pd.merge(wells2, df1[['StateWellNumber','Aquifer']], on='StateWellNumber', how='left')
Final_df2 = Final_df[Final_df.Aquifer == 'Dockum']



path_f = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\Dockum_ET'
os.chdir(path_f) #Set working directory
#Exporting as a csv file
Final_df2.to_csv('Well_Fil.csv', index=False)
