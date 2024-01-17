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

plt.rcParams.update({'mathtext.default':  'regular' })

def latlon(cenlon, cenlat, dom_x, dom_y):
    latrad = np.radians(cenlat)

    # do math
    fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
    fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
    dom_lon = cenlon + (dom_x)/fac_lon # distance in km
    dom_lat = cenlat + (dom_y)/fac_lat

    return(dom_lon, dom_lat)


#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("CENPATH", help="TC Vitals directory", type=str)
parser.add_argument("CENFN", help="TC Vitals filename", type=str)
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("CENTIME", help="cen datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

samurai_time = pd.to_datetime(args.ANALYSISTIME,format='%Y%m%d%H%M',utc=True)
leg_start = samurai_time - pd.Timedelta(45, 'min')
leg_end = samurai_time + pd.Timedelta(45, 'min')
print('leg start time: '+leg_start.strftime('%Y%m%d%H%M'))
print('leg end time: '+leg_end.strftime('%Y%m%d%H%M'))

# grab center from tcvitals
storm_lat, storm_lon, storm_intens, storm_rmw, storm_dir, storm_motion, center_time, u_motion, v_motion, storm_dir_rot = hot_calc_centers.center_tcvitals(args)

# testing adeck
storm_lat_2, storm_lon_2, storm_intens_2, storm_dir_2, storm_motion_2, df2, u_motion_2, v_motion_2, storm_dir_rot_2 = hot_calc_centers.center_adeck(args, samurai_time)

print([storm_lat, storm_lon, storm_intens, storm_rmw, storm_dir, storm_motion, u_motion, v_motion, storm_dir_rot])
print([storm_lat_2, storm_lon_2, storm_intens_2, np.nan, storm_dir_2, storm_motion_2, u_motion_2, v_motion_2, storm_dir_rot_2])

# more center finding from files will go here... adeck, flight plus *********


# create center file
# ***** update time to something more flexible, or include in inputs
# ref_latlon = make_cen_file(center_time, samurai_time, 45, storm_lat, storm_lon, u_motion, v_motion, './testing/')



#%% main code: step 2 - run samurai

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

ml_ver = 'WSPD'
#ml_ver = 'FRED'
#ml_ver = 'DERR'

if ml_ver == 'WSPD':
    ml_dir = ml_dir_base + 'AGU_23/'
    ml_file = 'NN_MSE_SCL_pred_v2.h5'
    #ml_dir = ml_dir_base + 'old_stable_model_v1/'
    #ml_file = 'NN_SCL20_model_v8.h5'
elif ml_ver == 'FRED':
    ml_dir = ml_dir_base + 'current_stable_model_v2/'
    ml_file = 'NN_WRpred_v4.h5'
elif ml_ver == 'DERR':
    ml_dir = ml_dir_base + 'stable_model_2d/'
    ml_file = 'NN_MSE_SCL_ERRpred_v1.h5'

# deployed on JHT/NHC
# inDir = os.getcwd()+'/'
#ml_dir = inDir+'./ML_models/'
#sam_dir = inDir+'./samurai_output/'
#output_dir = inDir+'./nn_output/'

# use tcvitals file to create background file
# hot_gen_background_file.py

# move all necessary files to ./samurai_input
# hrd = hot_grab_files.create_dataframe(data_dir+'hrd_radials',start,end)
# hdobs = hot_grab_files.create_dataframe(data_dir+'hdobs',start,end)
# hrd_sm = hot_grab_files.shrink_df(hrd, start, end)
# hdobs_sm = hot_grab_files.shrink_df(hdobs, start, end)
#hot_grab_files.copy_files(hrd_sm,sam_ingest_dir)
#hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)
# os.system('gunzip sam_ingest_dir/*.gz')

# generate samurai params file from master
# analysis_dir = modify_param_file(samurai_time, './samurai_parent/master_params/samurai_HOT_cart.params', './samurai_parent/samurai_params_cart')

# run samurai in XYZ mode
#os.system('samurai -params ./samurai_parent/samurai_params_cart')

# move files to analysis_dir
#os.system('mv ./samurai_parent/samurai_params '+analysis_dir)
#os.system('mv ./samurai_parent/samurai_input/*.cen '+analysis_dir)

# clean samurai_input
#os.system('rm ./samurai_parent/samurai_input/*')


#%% grab center from SAMURAI analysis
# fix link to be more adaptable
obj_master = './samurai_parent/master_params/objective_simplex.jl'
cart_file = './samurai_parent/samurai_testing/idalia_test/samurai_XYZ_analysis.nc'
hot_calc_centers.modify_obj_jl_file(obj_master, './objective_simplex.jl', storm_rmw, cart_file)

# run julia simplex code
os.system('sh run_julia.sh')

# open julia results
sam_cen = Dataset('samurai_center.nc', 'r')
xc_all = sam_cen.variables['final_xc'][:]
yc_all = sam_cen.variables['final_yc'][:]
xc_avg = np.nanmean(xc_all)
yc_avg = np.nanmean(yc_all)
print('avg: '+str(xc_avg))
print(xc_all)
print('avg: '+str(yc_avg))
print(yc_all)

# *** interpolate center to lat/lon *** julia indices are 1-based??????
ncfile_cart = Dataset(cart_file)
sam_lon = np.interp(xc_avg-1, ncfile_cart['x'][:].data, ncfile_cart['longitude'][:].data)
sam_lat = np.interp(yc_avg-1, ncfile_cart['y'][:].data, ncfile_cart['latitude'][:].data)


#%% re run SAMURAI in RTZ mode using updated center?
# move all necessary files to ./samurai_input
hrd = hot_grab_files.create_dataframe(data_dir+'hrd_radials',leg_start,leg_end)
#hdobs = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
hrd_sm = hot_grab_files.shrink_df(hrd, leg_start, leg_end)
#hdobs_sm = hot_grab_files.shrink_df(hdobs, leg_start, leg_end)
hot_grab_files.copy_files(hrd_sm,sam_ingest_dir)
#hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)
os.system('gunzip sam_ingest_dir/*.gz')

# ***** TIME ISSUE *******
ref_latlon = make_cen_file(center_time, samurai_time, 45, sam_lat, sam_lon, u_motion, v_motion, './samurai_parent/samurai_input/')

# generate samurai params file from master
analysis_dir = modify_param_file(samurai_time, './samurai_parent/master_params/samurai_HOT_wave.params', './samurai_parent/samurai_params_cyl')
sam_dir = sam_dir_base + analysis_dir

# run samurai in RTZ mode
os.system('samurai -params ./samurai_parent/samurai_params_cyl')

# move files to analysis_dir
os.system('mv ./samurai_parent/samurai_params_cyl '+analysis_dir)
os.system('mv ./samurai_parent/samurai_input/*.cen '+analysis_dir)

# clean samurai_input
#os.system('rm ./samurai_parent/samurai_input/*')

sam_fn = 'samurai_RTZ_analysis.nc'


#%% main code: step 3 - neural net

# load json and create model
json_file = open(ml_dir+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+ml_file)
print("Loaded model from disk")

# read samurai file and reshape samurai output arrays

#if args.STORM == 'AL10': #'IDALIA': may need to add year to storm... *******
#    test_dir = 'idalia_test/'
#    i_off = 61
#    j_off = 62
#    sam_fn = 'samurai_XYZ_analysis.nc'
#elif args.STORM == 'AL13': #'LEE':
#    test_dir = 'lee_pass1/'
#    i_off = 51
#    j_off = 45
#    sam_fn = 'samurai_XYZ_analysis.nc'
#elif args.STORM == 'AL14': #'MICHAEL':
#    test_dir = 'michael_superobs/'
#    i_off = 50
#    j_off = 50
#    sam_fn = 'samurai_RTZ_analysis.nc'

#ncfile = Dataset(sam_dir+test_dir+sam_fn)
ncfile = Dataset(sam_dir+sam_fn)
#ncfile = Dataset(sam_dir+'samurai_RTZ_analysis')
alt = ncfile['altitude'][:].data
alt_lev = (alt == 3)

# will insert simplex center finding algorithm here... ********

if sam_fn == 'samurai_RTZ_analysis.nc':
    
    theta = ncfile['theta'][:].data # math coordinate system
    radius = ncfile['radius'][:].data

    # create theta/radius grids
    th, rd = np.meshgrid(theta,radius,indexing='ij')

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
    X, Y = np.meshgrid(x - x[i_off],y - y[j_off],indexing='xy') #************
    lon_nc = ncfile['longitude'][:].data
    lat_nc = ncfile['latitude'][:].data
    #**** SUBTRACT STORM CENTER
    u_storm = np.squeeze(ncfile['U'][:].data[0,alt_lev,:,:])
    v_storm = np.squeeze(ncfile['V'][:].data[0,alt_lev,:,:])
    #print(ncfile['latitude'][:])
    u_storm[u_storm == -999] = np.nan
    v_storm[u_storm == -999] = np.nan
    rd = np.sqrt(X**2 + Y**2)
    th_nc = np.arctan2(Y, X)
    th = th_nc*180/np.pi
    th[th < 0] = th[th < 0] + 360
    
#****************
#***************

#print(X)


u_earth = u_storm + u_motion # u and v motion from tcvitals file 
v_earth = v_storm + v_motion
wspd_earth = np.sqrt(u_earth**2 + v_earth**2)

## RMW
# normalize wrt RMW and ravel data
sam_rmw = rd[np.unravel_index(np.nanargmax(wspd_earth),np.shape(wspd_earth))]
r_norm = rd.ravel(order='C')/sam_rmw
#r_norm = rd.ravel(order='C')/storm_rmw

# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('samurai 3-km RMW: '+str(sam_rmw))
#print('samurai 3-km RMW: '+str(radius[np.nanargmax(np.nanmean(np.sqrt(u_earth**2 + v_earth**2),axis=0))]))
print('samurai 3-km intens: '+str(np.nanmax(wspd_earth)))
#print('samurai 3-km intens: '+str(np.nanmax(np.nanmean(np.sqrt(u_earth**2 + v_earth**2),axis=0))))

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
alt[:] = 3000 # m
BT_Vmax = np.zeros_like(FL_wind)
BT_Vmax[:] = storm_intens*1.94 # convert to knots
SM_mag = np.zeros_like(FL_wind)
SM_mag[:] = storm_motion
RMW_arr = np.zeros_like(FL_wind)
RMW_arr[:] = sam_rmw

#print([np.nanmax(r_norm), np.nanmin(r_norm)])
#print([np.nanmax(theta_motionrel), np.nanmin(theta_motionrel)])
#print([np.nanmax(wspd_earth), np.nanmin(wspd_earth)])
#print([storm_intens, storm_motion])

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
mag_3km[(rd/sam_rmw < 0.5)] = np.nan
print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))
#print(theta_met[np.nanargmax(sfc_wind_pred)])
#print(rd[np.nanargmax(sfc_wind_pred)])

# convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012
u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th_nc) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th_nc)
v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th_nc) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th_nc)

# convert coords, first to cartesian
if sam_fn == 'samurai_RTZ_analysis.nc':
    x_plot = rd*np.cos(np.radians(th))
    y_plot = rd*np.sin(np.radians(th))
elif sam_fn == 'samurai_XYZ_analysis.nc':
    x_plot = X
    y_plot = Y

## regrid to cartesian grid?????? ******* (update: NO! want lat/lon in 1-D variables)
## may want to regrid polar to lat/lon grid...
# horiz_dim = np.arange(-175,176,1)
# x, y = np.meshgrid(horiz_dim, horiz_dim, indexing='xy')
# sfc_wind_pred_griddata = griddata((x_plot.ravel(), y_plot.ravel()), fl_wind.ravel(), (x, y), method='linear')

## then to lat lon (haversine formula? or Michael's simple function from before?)
# geod = Geod(ellps='WGS84')
# lon_r, lat_r, _ = geod.fwd(storm_lon, storm_lat, theta_met, rd*1000) # check where zero is expected for azimuth
# lon_r, lat_r = latlon(ref_latlon[1], ref_latlon[0], x, y) # Michael's code

#%% main code: step 4 - save output data as NetCDF (adapted from MetPy documentation)

# open file
ncfile_sfc = Dataset('./nn_output/HOT_SAMURAI_sfc_analysis_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_v2.nc',mode='w',format='NETCDF4') 

# define dimensions
# are these two-dimensional?? (could do a simple, x/y)
y_dim = ncfile_sfc.createDimension('latitude', len(lat_nc))     # latitude axis
x_dim = ncfile_sfc.createDimension('longitude', len(lon_nc))    # longitude axis
time_dim = ncfile_sfc.createDimension('time', 1) # unlimited axis (can be appended to)

# set up metadata
ncfile_sfc.title='CSU Predicted Surface Wind'
ncfile_sfc.subtitle="Generated using CSU Neural Net"

# set up variables
nclat = ncfile_sfc.createVariable('latitude', np.float32, ('latitude'))
nclat.units = 'degrees_north'
nclat.long_name = 'latitude'
nclon = ncfile_sfc.createVariable('longitude', np.float32, ('longitude'))
nclon.units = 'degrees_east'
nclon.long_name = 'longitude'
nctime = ncfile_sfc.createVariable('time', np.float64, ('time',))
nctime.units = 'seconds since 1970-01-01'
nctime.long_name = 'time'
# Define a 3D variable to hold the data
ncu = ncfile_sfc.createVariable('u_wind',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
ncu.units = 'm s-1' 
ncu.standard_name = 'eastward_wind' # this is a CF standard name
ncu.long_name = 'U component of the predicted surface wind'
ncv = ncfile_sfc.createVariable('v_wind',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
ncv.units = 'm s-1' 
ncv.standard_name = 'northward_wind' # this is a CF standard name
ncv.long_name = 'V component of the predicted surface wind'

# save data to arrays and reshape data into 2-D array
nclat[:] = lat_nc # (MAYBE?!) 
nclon[:] = lon_nc # (MAYBE?!)
nctime[:] = (pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True) - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
ncu[:,:,:] = u_nc[np.newaxis,:,:] # check dimensions
ncv[:,:,:] = v_nc[np.newaxis,:,:] # check dimensions

#print(ncu[:])
#print(u_nc)
#print(ncfile_sfc)
ncfile_sfc.close()

#%% main code: step 5 - generate any images

# temp images
# change colormap to something more windy
# x,y instead of lat/lon? could add radial rings instead...

# convert wind speed to knots, might actually change variables later, maybe more sig digits? *******
fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,105,5), cmap='Spectral_r')
#c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred, levels=np.arange(0,60,2.5), cmap='Spectral_r')
#c2 = axs[1].contourf(x_plot, y_plot, mag_3km, levels=np.arange(0,60,2.5), cmap='Spectral_r')
#c1 = axs[0].contourf(lon_r, lat_r, surface_wind, cmap='CMRmap')
#c2 = axs[1].contourf(lon_r, lat_r, fl_wind, cmap='CMRmap')i
axs[0].set_title('predicted surface wind speed')
#axs[0].set_title('predicted surface WN0+1 wind speed')
axs[1].set_title('3-km wind speed')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('kt')
fig.savefig(imDir+'NN_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'_v2.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r')
#c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,160,10), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.5,1.35,0.05), cmap='PuOr_r')
#c2 = axs[1].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.8,1.01,0.01), cmap='PuOr_r')
axs[1].contour(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(50,155,5))
axs[0].set_title('predicted surface wind speed')
axs[1].set_title('ratio: sfc/3km')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('')
fig.savefig(imDir+'NN_ratio_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'_v2.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(8,3))
c1 = axs[0].contourf(x_plot, y_plot, sfc_wind_pred*1.94, levels=np.arange(0,105,5), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, mag_3km*1.94, levels=np.arange(0,105,5), cmap='Spectral_r')
c3 = axs[2].contourf(x_plot, y_plot, sfc_wind_pred/mag_3km, levels=np.arange(0.5,1.35,0.05), cmap='magma')
axs[0].set_xticks([-100, -50, 0, 50])
axs[0].set_yticks([-100, -50, 0, 50])
axs[0].set_xlabel('distance from center (km)')
axs[0].set_ylabel('distance from center (km)')
axs[1].set_xlabel('distance from center (km)')
axs[2].set_xlabel('distance from center (km)')
axs[0].set_title('sfc wind speed')
axs[1].set_title('3-km wind speed')
axs[2].set_title('ratio: sfc/3-km')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title('kt')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title('kt')
cb3 = plt.colorbar(c3,ax=axs[2])
cb3.ax.set_title('')
fig.savefig(imDir+'NN_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_prelim_3pan_'+ml_ver+'_v2.png', dpi=200, bbox_inches='tight')



fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, mag_3km_srel, levels=np.arange(0,80,5), cmap='Spectral_r')
c2 = axs[1].contourf(x_plot, y_plot, mag_3km,levels=np.arange(0,80,5), cmap='Spectral_r')
#c1 = axs[0].contourf(x_plot, y_plot, mag_3km_srel, levels=np.arange(0,60,2.5), cmap='Spectral_r')
#c2 = axs[1].contourf(x_plot, y_plot, mag_3km,levels=np.arange(0,60,2.5), cmap='Spectral_r')
#c1 = axs[0].contourf(lon_r, lat_r, surface_wind, cmap='CMRmap')
#c2 = axs[1].contourf(lon_r, lat_r, fl_wind, cmap='CMRmap')i
axs[0].set_title('3-km storm relative wind')
axs[1].set_title('3-km earth relative wind')
cb1 = plt.colorbar(c1,ax=axs[0])
cb1.ax.set_title(r'm s$^{-1}$')
cb2 = plt.colorbar(c2,ax=axs[1])
cb2.ax.set_title(r'm s$^{-1}$')
#fig.savefig(imDir+'storm_motion_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, u_storm, levels=np.arange(-80,80,10), cmap='RdBu_r')
c2 = axs[1].contourf(x_plot, y_plot, u_earth,levels=np.arange(-80,80,10), cmap='RdBu_r')
axs[0].set_title('3-km storm relative u wind')
axs[1].set_title('3-km earth relative u wind')
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
#fig.savefig(imDir+'u_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, v_storm, levels=np.arange(-80,80,10), cmap='RdBu_r')
c2 = axs[1].contourf(x_plot, y_plot, v_earth,levels=np.arange(-80,80,10), cmap='RdBu_r')
axs[0].set_title('3-km storm relative v wind')
axs[1].set_title('3-km earth relative v wind')
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
#fig.savefig(imDir+'v_comparison_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'.png', dpi=200, bbox_inches='tight')

#fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
#c1 = axs[0].contourf(sfc_wind_pred, levels=np.arange(0,80,5), cmap='Spectral_r')
#c2 = axs[1].contourf(mag_3km, levels=np.arange(0,80,5),cmap='Spectral_r')
#c1 = axs[0].contourf(lon_r, lat_r, surface_wind, cmap='CMRmap')
#c2 = axs[1].contourf(lon_r, lat_r, fl_wind, cmap='CMRmap')
#cb1 = plt.colorbar(c1,ax=axs[0])
#cb2 = plt.colorbar(c2,ax=axs[1])
#fig.savefig(imDir+'NN_comparison_noxy.png', dpi=200, bbox_inches='tight')


fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, rd)
c2 = axs[1].contourf(x_plot, y_plot, np.radians(theta_motionrel))
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
#fig.savefig(imDir+'testing_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'.png', dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4))
c1 = axs[0].contourf(x_plot, y_plot, np.reshape(r_norm,u_storm.shape,order='C'))
c2 = axs[1].contourf(x_plot, y_plot, np.reshape(theta_nr,u_storm.shape,order='C'))
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
#fig.savefig(imDir+'testing2_'+args.STORM+'_'+args.ANALYSISTIME+'_preliminary_'+ml_ver+'.png', dpi=200, bbox_inches='tight')
