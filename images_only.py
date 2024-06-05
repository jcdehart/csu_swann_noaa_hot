#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:22:00 2023

@author: jcdehart
"""

#%% import necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import argparse
import os
import glob
#from pyproj import Geod
from tensorflow.keras.models import Sequential, model_from_json
import model_utils
#import hot_cen_file
from samurai_gen_file import make_cen_file, modify_param_file
import hot_grab_files
import hot_calc_centers
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({'mathtext.default':  'regular' })

#%% set up dirs
# local testing
inDir = '/bell-scratch/jcdehart/hot/'
data_dir = inDir+'ingest_dir/'
ml_dir_base = inDir+'ML_models/'
#ml_dir_base = inDir+'ML_models/stable_model/'
sam_dir_base = inDir+'samurai_parent/'
sam_ingest_dir = inDir+'samurai_parent/samurai_input/'
output_dir = inDir+'nn_testing/'
imDir = inDir+'images/'


ncfile = Dataset(sam_dir+sam_fn)
alt = ncfile['altitude'][:].data
alt_lev = (alt == alt_plane)

# will insert simplex center finding algorithm here... ********

if sam_fn == 'samurai_RTZ_analysis.nc':
    
    theta = ncfile['theta'][:].data # math coordinate system
    radius = ncfile['radius'][:].data

    # create theta/radius grids 
    th, rd = np.meshgrid(theta,radius,indexing='ij') # degrees
    th_nc = th*np.pi/180. # radians

    # grab velocity data at 3 km (math degrees) and convert from polar to cartesian and add storm motion back in
    u_polar = np.squeeze(ncfile['U'][:].data[0,alt_lev,:,:])
    v_polar = np.squeeze(ncfile['V'][:].data[0,alt_lev,:,:])
    u_polar[u_polar == -999] = np.nan
    v_polar[u_polar == -999] = np.nan
    u_storm = u_polar*np.cos(np.radians(th)) - v_polar*np.sin(np.radians(th))
    v_storm = u_polar*np.sin(np.radians(th)) + v_polar*np.cos(np.radians(th))

elif sam_fn == 'samurai_XYZ_analysis.nc': #******************
    
    x = ncfile['x'][:].data
    y = ncfile['y'][:].data
    X, Y = np.meshgrid(x - xc_avg,y - yc_avg,indexing='xy') #************
    #X, Y = np.meshgrid(x - x[i_off],y - y[j_off],indexing='xy') #************
    lon_nc = ncfile['longitude'][:].data
    lat_nc = ncfile['latitude'][:].data
    #**** SUBTRACT STORM CENTER
    u_storm = np.squeeze(ncfile['U'][:].data[0,alt_lev,:,:])
    v_storm = np.squeeze(ncfile['V'][:].data[0,alt_lev,:,:])
    #print(ncfile['latitude'][:])
    u_storm[u_storm == -999] = np.nan
    v_storm[u_storm == -999] = np.nan
    rd = np.sqrt(X**2 + Y**2)
    th_nc = np.arctan2(Y, X) # radians
    th = th_nc*180./np.pi
    th[th < 0] = th[th < 0] + 360 # degrees
    
#****************
#***************

u_earth = u_storm + u_motion # u and v motion from tcvitals file 
v_earth = v_storm + v_motion
wspd_earth = np.sqrt(u_earth**2 + v_earth**2)

## RMW
# normalize wrt RMW and ravel data
sam_rmw = rmw_avg
#sam_rmw = rd[np.unravel_index(np.nanargmax(wspd_earth),np.shape(wspd_earth))]
r_norm = rd.ravel(order='C')/sam_rmw
#r_norm = rd.ravel(order='C')/storm_rmw

# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('\n')
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('samurai FL RMW: '+str(sam_rmw))
print('samurai FL intens: '+str(np.nanmax(wspd_earth)))

## THETA
# convert samurai theta from math degrees (i.e., 0 = right) to met degrees (i.e., 0 = north)
theta_met = 90 + (360 - th)
theta_met[theta_met > 360] = theta_met[theta_met > 360] - 360

# rotate theta with respect to storm motion and convert to radians
theta_motionrel = theta_met - storm_dir
theta_motionrel[theta_motionrel < 0] = theta_motionrel[theta_motionrel < 0] + 360
theta_nr = np.radians(theta_motionrel.ravel(order='C'))

## FLIGHT LEVEL WIND
# grab flight level wind (i.e., 3 km) and reshape
FL_wind = wspd_earth.ravel(order='C')

## REMAINING
# set up 2-D arrays of repeating scalar values for flight level (make automatic), best track vmax (knots), storm motion magnitude (m/s)
alt = np.zeros_like(FL_wind)
if af == True:
    alt[:] = alt_plane*1000 # m
else:
    alt[:] = alt_plane*1000 # m

BT_Vmax = np.zeros_like(FL_wind)
BT_Vmax[:] = storm_intens*1.94 # convert to knots
SM_mag = np.zeros_like(FL_wind)
SM_mag[:] = storm_motion
RMW_arr = np.zeros_like(FL_wind)
RMW_arr[:] = sam_rmw

# normalized r (r/rmw), theta (wrt motion) x2, wind, altitude, vmax, storm motion magnitude
X_ratio = np.asarray([r_norm, theta_nr, theta_nr, FL_wind, alt, BT_Vmax, SM_mag, RMW_arr])  #double up on angle for sin and cosine
#X_ratio = np.asarray([r_norm.data, theta_nr.data, theta_nr.data, FL_wind.data, alt.data, BT_Vmax.data, SM_mag.data])  #double up on angle for sin and cosine
X_ratio[1,:] = np.sin(X_ratio[1,:]) # converted to radians above
X_ratio[2,:] = np.cos(X_ratio[2,:])

# standardize data
x_data = model_utils.Standardize_Vars(X_ratio.T)

# make prediction with the neural net
predict = nn_model.predict(x_data)
predict[r_norm < 0.5] = np.nan

# reshape arrays and mask orig missing data
# do diff math based on method
if ml_ver == 'WSPD':
    sfc_wind_pred = np.reshape(predict,u_storm.shape,order='C') # predicted surface wind
elif ml_ver == 'FRED':
    sfc_wind_pred = wspd_earth*np.reshape(predict,u_storm.shape,order='C') # multiply reduction factor and 3-km wind
elif ml_ver == 'DERR':
    sfc_wind_pred_oned = np.zeros_like(FL_wind)
    sfc_wind_pred_oned[:] = np.nan
    for i in range(len(predict)):
        sfc_wind_pred_oned[i] = model_utils.FWR(700,r_norm[i],FL_wind[i]) - predict[i]
    sfc_wind_pred = np.reshape(sfc_wind_pred_oned,u_storm.shape,order='C') # predicted wind errors


mag_3km_srel = np.sqrt(u_storm**2 + v_storm**2)
mag_3km = wspd_earth
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
sfc_wind_pred[mag_3km*1.94 < 20] = np.nan
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
mag_3km[(rd/sam_rmw < 0.5)] = np.nan
print('/n')
print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

#%% main code: step 5 - generate any images

# convert wind speed to knots, might actually change variables later, maybe more sig digits? *******
fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r',extend='max')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,105,5), cmap='Spectral_r',extend='max')
#c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred, levels=np.arange(0,60,2.5), cmap='Spectral_r')
#c2 = axs[1].contourf(x_plot, y_plot, mag_3km, levels=np.arange(0,60,2.5), cmap='Spectral_r')
axs[0].set_title('predicted surface wind speed')
axs[1].set_title('FL wind speed')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('kt')
fig.savefig(imDir+'NN_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'_vnhc_premade_jonrmw.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r', extend='max')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.75,1.10,0.05), cmap='coolwarm', extend='both')
#c2 = axs[1].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.8,1.01,0.01), cmap='PuOr_r')
axs[1].contour(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(30,155,5))
axs[0].set_title('predicted surface wind speed')
axs[1].set_title('ratio: sfc/FL')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('')
fig.savefig(imDir+'NN_ratio_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'_vnhc_premade_jonrmw.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(8,2))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r', extend='max')
c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,105,5), cmap='Spectral_r', extend='max')
c3 = axs[2].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.75,1.10,0.05), cmap='coolwarm', extend='both')
axs[0].set_xticks([-100, 0, 100])
axs[0].set_yticks([-100, 0, 100])
axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].yaxis.set_minor_locator(AutoMinorLocator())
axs[0].set_xlabel('distance from center (km)')
axs[0].set_ylabel('distance from center (km)')
axs[1].set_xlabel('distance from center (km)')
axs[2].set_xlabel('distance from center (km)')
axs[0].set_title('sfc wind speed')
axs[1].set_title('FL wind speed')
axs[2].set_title('ratio: sfc/FL')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('kt')
cb3 = plt.colorbar(c3,ax=axs[2])
cb3.ax.set_title('')
fig.savefig(imDir+'NN_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_prelim_3pan_'+ml_ver+'_vnhc_premade_jonrmw.png', dpi=200, bbox_inches='tight')


