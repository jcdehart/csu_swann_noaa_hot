#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:46:00 2022

@author: jcdehart
"""

#%% import necessary packages

import numpy as np
#from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import argparser
from pyproj import Geod
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import os


#%% set up dirs

# local testing
inDir = '/bell-scratch/adesros/JHT_Michael_Test/'
ml_dir = inDir+'ML_models/'
sam_dir = inDir+'samurai_output/'
output_dir = inDir+'nn_output/'

# deployed on JHT/NHC
#ml_dir = './ML_models/'
#sam_dir = './samurai_output/'
#output_dir = './nn_output/'

# load best track data (tcvitals?)


#%% neural net

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+"/stable_model/NN_SCL20_model_v8.h5")
print("Loaded model from disk")

# read samurai file and reshape samurai output arrays
ncfile = Dataset(sam_dir+'samurai_RTZ_analysis.nc')
alt = ncfile['altitude']
theta = ncfile['theta'] # math coordinate system
radius = ncfile['radius']
u_storm = ncfile['U'] # wind in storm relative framework (need to add in storm motion????)
v_storm = ncfile['V']
u_earth = u_storm + u_motion # u and v motion from tcvitals file - combine scripts?
v_earth = v_storm + v_motion

alt_lev = (alt == 3)

# normalize and ravel data
r_norm = radius.ravel(order='C')/rmw

# convert storm motion from standard math degrees to met degrees
theta_met = 90 + (360 - theta)
theta_met[theta_met > 360] = theta_met[theta_met > 360] - 360

# rotate theta with respect to storm motion?
# theta_motionrel = 
# theta_nr = theta_motionrel.ravel(order='C')

FL_wind = np.sqrt(u_earth[0,alt_lev,:,:]**2 + v_earth[0,alt_lev,:,:]**2).ravel(order='C')

# normalized r (r/rmw), theta (with respect to motion?) x2, wind, and ??
X_ratio = np.asarray([r_norm.data, theta_nr.data, theta_nr.data, FL_wind.data, alt.data, BT_Vmax.data, SM_mag.data])  #double up on angle for sin and cosine
X_ratio[1,:] = np.sin(X_ratio[1,:])
X_ratio[2,:] = np.cos(X_ratio[2,:])

# make prediction with the neural net
predict = loaded_model.predict(x_data)


# reshape arrays
sfc_wind_pred = np.reshape(x_data,u.shape(),order='C')


# convert coords, first to cartesian
x_plot = radius*np.cos(np.radians(theta))
y_plot = radius*np.sin(np.radians(theta))

# then to lat lon (haversine formula?)
geod = Geod(ellps='WGS84')
lon_r, lat_r, _ = geod.fwd(lon_c, lat_c, theta_met, radius*1000) # check where zero is expected for azimuth

#%% save output data as NetCDF (adapted from MetPy documentation)

# open file
ncfile = Dataset('./nn_output/recon_analysis_'+date+'.nc',mode='w',format='NETCDF4') # which format?

# define dimensions
# are these two-dimensional?? (could do a simple, x/y)
y_dim = ncfile.createDimension('y0', len(y_plot))     # latitude axis
x_dim = ncfile.createDimension('x0', len(x_plot))    # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to)

# set up metadata
ncfile.title='OUTPUT FROM NEURAL NETWORK'
ncfile.subtitle="My model data subtitle"

# set up variables
lat = ncfile.createVariable('lat', np.float32, ('y0','x0'))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('y0','x0'))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'hours since 1800-01-01'
time.long_name = 'time'
# Define a 3D variable to hold the data
uv = ncfile.createVariable('wind_speed',np.float64,('time','y0','x0')) # note: unlimited dimension is leftmost
uv.units = 'm s-1' 
uv.standard_name = 'wind_speed' # this is a CF standard name
uv.long_name = 'neural_net_predicted_wind_speed'

# save data to arrays and reshape data into 2-D array
lat[:] = lon_r.T # (MAYBE?!) 
lon[:] = lat_r.T # (MAYBE?!)
uv[:,:,:] = sfc_wind_pred.T # check dimensions


#%% generate any images

# temp images
# change colormap to something more windy
# x,y instead of lat/lon? could add radial rings instead...

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(lon, lat, surface_wind, cmap='CMRmap')
c2 = axs[1].contourf(lon, lat, fl_wind, cmap='CMRmap')
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
