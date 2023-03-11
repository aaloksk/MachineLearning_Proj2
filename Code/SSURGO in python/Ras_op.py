# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:54:23 2023

@author: Aalok
"""

import os
import geopandas as gpd
import fiona
import pandas as pd

path = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\GIS\\SHP\\Doc_Pgn'
os.chdir(path)
fname = 'Doc_MUPOLYGON.shp'

mpgn = gpd.read_file(fname)


#Importing wells
path_well = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\GIS\\SHP\\Wells_MU'
os.chdir(path_well)
fwells = 'Wells.shp'
wells = gpd.read_file(fwells)



#Setting working directory
path_gdb ='C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project2\\GIS\\gSSURGO_TX\\gSSURGO_TX.gdb'
layers = fiona.listlayers(path_gdb)



#extracting the layers essential for soil texture
texture = gpd.read_file(path_gdb,layer = 'chtexture')
texturegrp = gpd.read_file(path_gdb,layer = 'chtexturegrp')
horizon = gpd.read_file(path_gdb,layer = 'chorizon')
component = gpd.read_file(path_gdb,layer = 'component')
mapunit = gpd.read_file(path_gdb,layer = 'mapunit')
legend = gpd.read_file(path_gdb,layer = 'legend')
#soilpoly = gpd.read_file(fname,layer = 'MUPOLYGON')
#soilpoly = wells
soilpoly = mpgn

#soilpoly.crs
#clipped_polygon = gpd.clip(soilpoly, oggaea)
#clipped_polygon.to_file('clipped_polygon.shp')
#clipped_polygon.plot()
#Plot the soil polygon
plotx = soilpoly.plot()
plotx.set_xlabel('Easting')
plotx.set_ylabel('Northing')



#perform joins to get soil texture with the map
texturegrp1 = pd.merge(texturegrp,texture[['texcl','chtgkey']], on='chtgkey',how='left')
horizon1 = pd.merge(horizon,texturegrp1[['texcl','chkey']], on='chkey',how='left')
component1 = pd.merge(component,horizon1[['texcl','cokey']], on='cokey',how='left')
mapunit1 = pd.merge(mapunit,component1[['texcl','mukey']], on='mukey',how='left')


mukey = soilpoly['MUKEY']
soilpoly = soilpoly.assign(mukey=mukey)


#extract the crs and geometry
crsx = soilpoly.crs
geomx = soilpoly.geometry


soilpoly3 = pd.merge(soilpoly,mapunit1,on='mukey',how='inner')
soilpoly3.set_geometry(geomx,inplace=True,crs = crsx)
soilpoly3.plot('texcl')



soilpoly4 = pd.merge(soilpoly,mapunit1,on='mukey',how='left')
soilpoly4.set_geometry(geomx,inplace=True,crs = crsx)
soilpoly4.plot('texcl')
