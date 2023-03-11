# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:44:41 2023

@author: Aalok
"""

import os
import geopandas as gpd
import fiona
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\GIS\\SHP\\Doc_Pgn'
os.chdir(path)

fname = 'Doc_MUPOLYGON.shp'

mpgn = gpd.read_file(fname)
mpgn.columns



path2 = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\GIS\\SHP\\minor_aquifers'
os.chdir(path2)

min_aq = gpd.read_file('Minor_Aquifers.shp')

dockum = min_aq.loc[min_aq.AQU_NAME == 'DOCKUM']

espgaea = 5070
docaea = dockum.to_crs(espgaea)
docaea.plot()
docaea.to_file(' dockumea.shp')

