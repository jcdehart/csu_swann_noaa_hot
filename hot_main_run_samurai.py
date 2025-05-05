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
import center_funcs
import hot_prep_data
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


plt.rcParams.update({'mathtext.default':  'regular' })

def latlon(cenlon, cenlat, dom_x, dom_y):
    latrad = np.radians(cenlat)

    # do math
    fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
    fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
    dom_lon = cenlon + (dom_x)/fac_lon # distance in km
    dom_lat = cenlat + (dom_y)/fac_lat

    return(dom_lon, dom_lat)


def xy(lat, lon, lat0, lon0):
    # Approximate radius of earth in km
    R = 6373.0
    lat_plane = np.radians(lat)
    lon_plane = np.radians(lon)
    lat_cen = np.radians(lat0)
    lon_cen = np.radians(lon0)
    #dlon = lon_cen - lon_plane
    #dlat = lat_cen - lat_plane
    dlon = lon_plane - lon_cen
    dlat = lat_plane - lat_cen
    #a = np.sin(dlat / 2)**2 + np.cos(lat_plane) * np.cos(lat_cen) * np.sin(dlon / 2)**2
    a = np.sin(dlat / 2)**2 + np.cos(lat_cen) * np.cos(lat_plane) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    #bearing = np.arctan2(np.sin(lon_cen-lon_plane)*np.cos(lat_cen), np.cos(lat_plane)*np.sin(lat_cen)-np.sin(lat_plane)*np.cos(lat_cen)*np.cos(lon_cen-lon_plane))
    bearing = np.arctan2(np.sin(lon_plane-lon_cen)*np.cos(lat_plane), np.cos(lat_cen)*np.sin(lat_plane)-np.sin(lat_cen)*np.cos(lat_plane)*np.cos(lon_plane-lon_cen))
    #bearing is in north-facing coords...
    x = distance*np.cos(-1*(bearing-(np.pi/2)))
    y = distance*np.sin(-1*(bearing-(np.pi/2)))
    return(x,y)

#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--CENFN", default="gfs.tXXz.syndata.tcvitals.tm00", help="TC Vitals filename", type=str)
parser.add_argument("--CENPATH", default="./ingest_dir/center_data", help="TC Vitals directory", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

# ***** change into an argument, or change to start time + duration, model will be settled in future, deal with af only later
cyl = False
af = False
#alt_plane = 1.5
#alt_plane = 3.0
ml_ver = 'FRED'

if cyl == True:
    mode = '_cyl'
else:
    mode = ''

#%% set up dirs
# local testing
inDir = '/bell-scratch/jcdehart/hot/'
data_dir = inDir+'ingest_dir/'
ml_dir_base = inDir+'ML_models/'
sam_dir_base = inDir+'samurai_parent/'
sam_ingest_dir = inDir+'samurai_parent/samurai_input/'
output_dir = inDir+'nn_testing/'
imDir = inDir+'images/'

ml_dir = ml_dir_base + 'Current_HOT_Model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'

leg_start = pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True)
leg_end = pd.to_datetime(args.ENDTIME,format='%Y%m%d%H%M',utc=True)
samurai_time = leg_start + ((leg_end-leg_start)/2).round('min')
analysis_time = samurai_time.strftime('%Y%m%d%H%M')
print('\n')
print('leg start time: '+leg_start.strftime('%Y%m%d%H%M'))
print('leg end time: '+leg_end.strftime('%Y%m%d%H%M'))

# grab center from tcvitals (renaming storm_name to storm_name_2)....****
storm_lat_1, storm_lon_1, storm_intens, storm_rmw, storm_dir, storm_motion, center_time, u_motion_1, v_motion_1, storm_dir_rot, storm_name_2 = hot_calc_centers.center_tcvitals(args)

# grab center from adeck
storm_lat_2, storm_lon_2, storm_intens_2, storm_dir_2, storm_motion_2, df_2, u_motion_2, v_motion_2, storm_dir_rot_2 = hot_calc_centers.center_adeck(args, samurai_time)

print('\n')
print('center stats from tcvitals and adeck:')
print([storm_lat_1, storm_lon_1, storm_intens, storm_rmw, storm_dir, storm_motion, u_motion_1, v_motion_1, storm_dir_rot])
print([storm_lat_2, storm_lon_2, storm_intens_2, np.nan, storm_dir_2, storm_motion_2, u_motion_2, v_motion_2, storm_dir_rot_2])

# more center finding from files will go here... flight plus *********


# calc W-C center (hdobs)
# ******* I think code adds buffer on to beginning and end - potentially change *******

# move all necessary files to ./samurai_input
# **** for now specifying whether AF only or not ***** UPDATE IN FUTURE

# first remove any existing files
os.system('rm -rf '+sam_ingest_dir+'/*.list')
os.system('rm -rf '+sam_ingest_dir+'/*.hdob')
os.system('rm -rf '+sam_ingest_dir+'/*.gz')

if af == False:
    hrd_init = hot_grab_files.create_dataframe(data_dir+'hrd_radials',leg_start,leg_end)
    hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
    hrd_sm = hot_grab_files.shrink_df(hrd_init, leg_start, leg_end, storm_name_2, af)
    hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name_2, af)
    hot_grab_files.copy_files(hrd_sm,sam_ingest_dir)
    hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)
    os.system('for i in '+sam_ingest_dir+'/*.gz; do gunzip $i; done')
else:
    hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
    hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name_2, af)
    hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)
    #os.system('rm '+sam_ingest_dir+'*KWBC*') # *** check??????

print(hdobs_sm.mission.unique())
if len(hdobs_sm.mission.unique()) > 1:
    print('warning: more than one flight ID in hdobs files!!')

# rename hdob .list files to .hdob for SAMURAI ingestion
os.system('for i in '+sam_ingest_dir+'/*.txt; do mv "$i" "${i%.txt}.hdob"; done')

if af == True:
    hdobs = hot_calc_centers.read_hdobs('KNHC', storm_name_2)
else:
    hdobs = hot_calc_centers.read_hdobs('KWBC', storm_name_2)

print('avg altitude: '+str(hdobs.hgt.mean()))
print('min altitude: '+str(hdobs.hgt.min()))
print('max altitude: '+str(hdobs.hgt.max()))

print('avg p: '+str(hdobs.p.mean()))
print('min p: '+str(hdobs.p.min()))
print('max p: '+str(hdobs.p.max()))

peaks, properties, willfunc, wdir_rel = center_funcs.find_peaks(hdobs.wsp.values, hdobs.wdir.values, hdobs.dval.values, 0, 0) # change u_tc/v_tc?????
peaks_refined = peaks.astype(int)
window = 50
approaches = len(peaks)
peaks_refined = center_funcs.refine_peaks_minima(peaks_refined, willfunc)
dt_wc, lon_wc_old, lat_wc_old = center_funcs.peaks_wc(peaks_refined, approaches, hdobs.lat.values, hdobs.lon.values, wdir_rel, hdobs.dt)

# may have error in center finding code... ******* ask Chris
lon_wc = hdobs.lon.iloc[dt_wc[0]]
lat_wc = hdobs.lat.iloc[dt_wc[0]]

print('W-C center lat: '+str(lat_wc)+', center lon: '+str(lon_wc))
print(dt_wc)

print('averaging all 3 centers')

# perhaps have different weights based on tcvitals intensity??? ********
wgt = np.array([1, 1, 3])
storm_lon = lon_wc
storm_lat = lat_wc
#storm_lon = np.average(np.array([lon_wc,storm_lon_1,storm_lon_2]),weights=wgt)
#storm_lat = np.average(np.array([lat_wc,storm_lat_1,storm_lat_2]),weights=wgt)
u_motion = np.nanmean(np.array([u_motion_1,u_motion_2]))
v_motion = np.nanmean(np.array([v_motion_1,v_motion_2]))
print([storm_lat, storm_lon])
print([u_motion, v_motion])

# grab plane altitude manually
med_hgt = (500*(hdobs.hgt/500).round()).median()

if med_hgt == 1500.:
    alt_plane = 1.5
    print('plane hgt = '+str(alt_plane))
elif med_hgt == 3000.:
    alt_plane = 3.
    print('plane hgt = '+str(alt_plane))
else:
    hgt_options = np.array([1.5, 3.0])
    alt_plane = hgt_options[np.argmin(np.abs(med_hgt - hgt_options))]
    print('plane hgt not within 500 m of options')
    print('using plane hgt = '+str(alt_plane))
    print('actual med plane hgt = '+str(med_hgt))


#%% main code: step 2 - run samurai


# deployed on JHT/NHC
# inDir = os.getcwd()+'/'
#ml_dir = inDir+'./ML_models/'
#sam_dir = inDir+'./samurai_output/'
#output_dir = inDir+'./nn_output/'

# use tcvitals file to create background file
# hot_gen_background_file.py


# create center file
ref_latlon_cart = make_cen_file(samurai_time, leg_start, leg_end, storm_lat, storm_lon, u_motion, v_motion, './samurai_parent/samurai_input/')
#ref_latlon_cart = make_cen_file(samurai_time, samurai_time, dur, storm_lat, storm_lon, u_motion, v_motion, './samurai_parent/samurai_input/')

# generate samurai params file from master
if af == True:
    analysis_dir_cart = modify_param_file(samurai_time, './samurai_parent/master_params/samurai_HOT_cart_mask.params', './samurai_parent/samurai_params_cart')
else:
    analysis_dir_cart = modify_param_file(samurai_time, './samurai_parent/master_params/samurai_HOT_cart.params', './samurai_parent/samurai_params_cart')
sam_dir_cart = sam_dir_base + analysis_dir_cart +'_cart/'
#sam_dir_cart = sam_dir_base + 'samurai_testing/idalia_test/'
os.system('mkdir -p '+sam_dir_cart)

# run samurai in XYZ mode
#os.system('samurai -params ./samurai_parent/samurai_params_cart')
os.system('/bell-scratch/mmbell/hot/samurai-hot/release/bin/samurai -params ./samurai_parent/samurai_params_cart')

# move files to analysis_dir
os.system('mv ./samurai_parent/samurai_params_cart '+sam_dir_cart)
os.system('mv ./samurai_parent/samurai_input/*.cen '+sam_dir_cart)
os.system('mv ./samurai_parent/samurai_input/*.in '+sam_dir_cart)


#%% grab center from SAMURAI analysis
# fix link to be more adaptable
obj_master = './samurai_parent/master_params/objective_simplex.jl'
#cart_file = './samurai_parent/samurai_testing/idalia_test/samurai_XYZ_analysis.nc'
cart_file = sam_dir_cart+'samurai_XYZ_analysis.nc'
hot_calc_centers.modify_obj_jl_file(obj_master, './objective_simplex.jl', storm_rmw, cart_file)

# run julia simplex code
os.system('sh run_julia.sh')

# open julia results
sam_cen = Dataset('samurai_center.nc', 'r')
xc_all = sam_cen.variables['final_xc'][:]
yc_all = sam_cen.variables['final_yc'][:]
rmw_all = sam_cen.variables['final_rmw'][:]

# avg values based on aircraft flight level
if alt_plane == 3.0:
    xc_avg = np.nanmean(xc_all[3:])
    yc_avg = np.nanmean(yc_all[3:])
    rmw_avg = np.nanmean(rmw_all[3:])
elif alt_plane == 1.5:
    xc_avg = np.nanmean(xc_all[0:2])
    yc_avg = np.nanmean(yc_all[0:2])
    rmw_avg = np.nanmean(rmw_all[0:2])
else:
    xc_avg = np.nanmean(xc_all)
    yc_avg = np.nanmean(yc_all)
    rmw_avg = np.nanmean(rmw_all)
print('\n')
print('avg xc: '+str(xc_avg)+', yc: '+str(yc_avg)+', rmw: '+str(rmw_avg))

# interpolate center to lat/lon
ncfile_cart = Dataset(cart_file)
sam_lon_tmp = np.interp(xc_avg, ncfile_cart['x'][:].data, ncfile_cart['longitude'][:].data)
sam_lat_tmp = np.interp(yc_avg, ncfile_cart['y'][:].data, ncfile_cart['latitude'][:].data)

# check for distance from W-C center *** fix later???
if (np.abs(sam_lon_tmp - lon_wc) > 0.4) | (np.abs(sam_lat_tmp - lat_wc) > 0.4):
    print('objective center too far from W-C, defaulting to W-C center')
    sam_lon = lon_wc
    sam_lat = lat_wc
else:
    print('objective center seems reasonable')
    sam_lon = sam_lon_tmp
    sam_lat = sam_lat_tmp

print('samurai center lat: '+str(sam_lat)+', center lon: '+str(sam_lon))

# convert hdobs to xy
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,sam_lat,sam_lon)


#%% re run SAMURAI in RTZ mode using updated center, using same input files

if cyl == True:

    # changing ref time to time associated with storm center
    ref_latlon = make_cen_file(dt_wc[0], leg_start, leg_end, lat_wc, lon_wc, u_motion, v_motion, './samurai_parent/samurai_input/')

    # generate samurai params file from master
    analysis_dir = modify_param_file(dt_wc[0], './samurai_parent/master_params/samurai_HOT_wave.params', './samurai_parent/samurai_params_cyl')
    sam_dir = sam_dir_base + analysis_dir +'_cyl/'
    os.system('mkdir -p '+sam_dir)

    # run samurai in RTZ mode
    os.system('/bell-scratch/mmbell/hot/samurai-hot/release/bin/samurai -params ./samurai_parent/samurai_params_cyl')

    # move files to analysis_dir
    os.system('mv ./samurai_parent/samurai_params_cyl '+sam_dir)
    os.system('mv ./samurai_parent/samurai_input/*.cen '+sam_dir)
    os.system('mv ./samurai_parent/samurai_input/*.in '+sam_dir)

    # clean samurai_input
    os.system('rm ./samurai_parent/samurai_input/*')

    sam_fn = 'samurai_RTZ_analysis.nc'

else:

    sam_fn = 'samurai_XYZ_analysis.nc'
    sam_dir = sam_dir_cart


#%% main code: step 3 - neural net

# load json and create model
json_file = open(ml_dir+json_fn, 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+ml_file)
print("Loaded model from disk")

# read samurai file and reshape samurai output arrays

ncfile = Dataset(sam_dir+sam_fn)
alt = ncfile['altitude'][:].data
alt_lev = (alt == alt_plane)

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

    # calculate earth-relative components and magnitude
    # use rmw from analysis if cylindrical
    u_earth = u_storm + u_motion # u and v motion from tcvitals file 
    v_earth = v_storm + v_motion
    wspd_earth = np.sqrt(u_earth**2 + v_earth**2)
    sam_rmw = rd[np.unravel_index(np.nanargmax(wspd_earth),np.shape(wspd_earth))]

elif sam_fn == 'samurai_XYZ_analysis.nc': 
    
    x = ncfile['x'][:].data
    y = ncfile['y'][:].data
    X, Y = np.meshgrid(x - xc_avg,y - yc_avg,indexing='xy') 
    lon_nc = ncfile['longitude'][:].data
    lat_nc = ncfile['latitude'][:].data
    u_storm = np.squeeze(ncfile['U'][:].data[0,alt_lev,:,:])
    v_storm = np.squeeze(ncfile['V'][:].data[0,alt_lev,:,:])
    u_storm[u_storm == -999] = np.nan
    v_storm[u_storm == -999] = np.nan
    rd = np.sqrt(X**2 + Y**2)
    th_nc = np.arctan2(Y, X) # radians
    th = th_nc*180./np.pi
    th[th < 0] = th[th < 0] + 360 # degrees
    
    # calculate earth-relative components and magnitude
    # rmw from Jon's code if cartesian
    u_earth = u_storm + u_motion # u and v motion from tcvitals file 
    v_earth = v_storm + v_motion
    wspd_earth = np.sqrt(u_earth**2 + v_earth**2)
    sam_rmw = rmw_avg


# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('\n')
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('samurai FL RMW: '+str(sam_rmw))
print('samurai FL intens: '+str(np.nanmax(wspd_earth)))

# prepare variables for NN model - SAMURAI wind field
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km
X_ratio, r_norm = hot_prep_data.process_nn_vars(rd, sam_rmw, th, storm_dir, storm_intens, storm_motion, wspd_earth, alt_plane, af)

# standardize data
x_data = model_utils.Standardize_Vars(X_ratio.T)

# make prediction with the neural net
predict = nn_model.predict(x_data)
predict[r_norm < 0.3] = np.nan
#predict[r_norm < 0.5] = np.nan

# reshape arrays and mask orig missing data
sfc_wind_pred = wspd_earth*np.reshape(predict,u_storm.shape,order='C') # multiply reduction factor and flight-level wind

# grab flight-level storm-relative data and remove bad data
mag_3km_srel = np.sqrt(u_storm**2 + v_storm**2)
mag_3km = wspd_earth
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
sfc_wind_pred[mag_3km*1.94 < 20] = np.nan
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
mag_3km[(rd/sam_rmw < 0.3)] = np.nan
#mag_3km[(rd/sam_rmw < 0.5)] = np.nan
print('/n')
print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

# get RMW of max point
swann_rmw = np.nanmin(rd[sfc_wind_pred == np.nanmax(sfc_wind_pred)]) # think this should just be a point location... *******

# censor out boundaries with spectral ringing
mag_3km[:4,:] = np.nan
mag_3km[-4:,:] = np.nan
mag_3km[:,:4] = np.nan
mag_3km[:,-4:] = np.nan
sfc_wind_pred[:4,:] = np.nan
sfc_wind_pred[-4:,:] = np.nan
sfc_wind_pred[:,:4] = np.nan
sfc_wind_pred[:,-4:] = np.nan

# convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012
u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th_nc) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th_nc)
v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th_nc) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th_nc)

### run aircraft data through model ###
# create theta/radius grids
rd_ac = np.sqrt(x_plane**2 + y_plane**2)
th_r_ac = np.arctan2(y_plane, x_plane)
th_ac = th_r_ac*180./np.pi

wspd_earth_ac = hdobs.wsp.values/1.94 # CONVERTING HDOBS KTS TO M/S NEEDED FOR ALEX'S MODEL ******

# prepare variables for NN model - HDOBs wind field
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km
X_ratio_ac, r_norm_ac = hot_prep_data.process_nn_vars(rd_ac, sam_rmw, th_ac, storm_dir, storm_intens, storm_motion, wspd_earth_ac, alt_plane, af)

# standardize data
x_data_ac = model_utils.Standardize_Vars(X_ratio_ac.T)

# make prediction with the neural net
predict_ac = nn_model.predict(x_data_ac)
predict_ac[r_norm_ac < 0.3] = np.nan

# reshape arrays and mask orig missing data
sfc_wind_pred_ac = wspd_earth_ac*predict_ac.T[0] # multiply reduction factor and flight-level wind

# grab flight-level storm-relative data and remove bad data
mag_3km_ac = wspd_earth_ac
sfc_wind_pred_ac[np.isnan(mag_3km_ac)] = np.nan
sfc_wind_pred_ac[mag_3km_ac*1.94 < 20] = np.nan ##### UNITS ALREADY IN KTS
sfc_wind_pred_ac[np.isnan(mag_3km_ac)] = np.nan
mag_3km_ac[(rd_ac/sam_rmw < 0.3)] = np.nan

#%% main code: step 4 - save output data as NetCDF (adapted from MetPy documentation)

f = open(sam_dir+args.STORM+'_'+analysis_time+'_data.txt','w')
lines = ['SWANN RMW: '+str(swann_rmw)+'\n', 'samurai FL intens: '+str(np.nanmax(wspd_earth))+'\n', 'SWANN Vmax: '+str(np.nanmax(sfc_wind_pred))]
#lines = ['samurai FL RMW: '+str(sam_rmw)+'\n', 'samurai FL intens: '+str(np.nanmax(wspd_earth))+'\n', 'predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred))]
f.writelines(lines)
f.close()

if alt_plane == 1.5:
    sf_frac = 0.8
elif alt_plane == 3.0:
    sf_frac = 0.9

sam_fl_vmax = np.nanmax(wspd_earth*1.94)
hdobs_fl_vmax = np.nanmax(hdobs.wsp)
swann_sam_vmax = np.nanmax(sfc_wind_pred*1.94)
swann_hdobs_vmax = np.nanmax(sfc_wind_pred_ac*1.94)
simp_frank = sf_frac*sam_fl_vmax

figtitle = storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + hdobs_sm.mission.unique()[0] + ' | ' + leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M') + ' UTC'

textstr = '\n'.join((
    'Inputs: HRD TDR, HDOBS',
    'SAM Center: %.2f N, %.2f W' % (sam_lat,np.abs(sam_lon),), # currently assuming negative longitudes **********
    'RMW: %.1f (nm)' % (sam_rmw,),
    'Simp. Franklin: %.1f (kt)' % (simp_frank,), ))

# figure out how to add simplified franklin number back in
#    'Simplified Franklin: %.1f (kt)' % (sf_frac*np.nanmax(wspd_earth*1.94), ) ))

# convert coords, first to cartesian
if sam_fn == 'samurai_RTZ_analysis.nc':
    x_plot = rd*np.cos(np.radians(th))
    y_plot = rd*np.sin(np.radians(th))
    # lon_nc, lat_nc = latlon(sam_lon, sam_lat, x_plot, x_plot) # Michael's code ####### CONFIRM LAT LON with best center ########
    # geod = Geod(ellps='WGS84')
    # lon_r, lat_r, _ = geod.fwd(storm_lon, storm_lat, theta_met, rd*1000) # check where zero is expected for azimuth
elif sam_fn == 'samurai_XYZ_analysis.nc':
    x_plot = X
    y_plot = Y

    # open file
    ncfile_sfc = Dataset('./nn_output/HOT_SAMURAI_sfc_analysis_'+args.STORM+'_'+analysis_time+'.nc',mode='w',format='NETCDF4') 

    # define dimensions
    # are these two-dimensional?? (could do a simple, x/y)
    y_dim = ncfile_sfc.createDimension('latitude', len(lat_nc))     # latitude axis
    x_dim = ncfile_sfc.createDimension('longitude', len(lon_nc))    # longitude axis
    time_dim = ncfile_sfc.createDimension('time', 1) # unlimited axis (can be appended to)
    
    # set up metadata
    ncfile_sfc.title='CSU Predicted Surface Wind'
    ncfile_sfc.subtitle="Generated using CSU SWANN"

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
    nctime[:] = (samurai_time - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
    ncu[:,:,:] = u_nc[np.newaxis,:,:] # check dimensions
    ncv[:,:,:] = v_nc[np.newaxis,:,:] # check dimensions
    
    ncfile_sfc.close()

#%% main code: step 5 - generate any images
# x,y instead of lat/lon? 

# wind radii calculations

wind_radii = [34,50,64]
radii_vals = np.zeros((3,4)) # NE, SE, SW, NW

for i in range(len(wind_radii)):
    radii_vals[i,0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot < 0) | (y_plot < 0), np.nan, rd))
    radii_vals[i,1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot < 0) | (y_plot > 0), np.nan, rd))
    radii_vals[i,2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot > 0) | (y_plot > 0), np.nan, rd))
    radii_vals[i,3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot > 0) | (y_plot < 0), np.nan, rd))

echo_edges = np.zeros(4)
echo_edges[0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot < 0) | (y_plot < 0), np.nan, rd))
echo_edges[1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot < 0) | (y_plot > 0), np.nan, rd))
echo_edges[2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot > 0) | (y_plot > 0), np.nan, rd))
echo_edges[3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot > 0) | (y_plot < 0), np.nan, rd))

vmax_col_labels = ['HDOBS\nVmax (kt)','SAMURAI\nVmax (kt)']
vmax_row_labels = ['FL','SWANN']
vmax_table = [[hdobs_fl_vmax,sam_fl_vmax],[swann_hdobs_vmax, swann_sam_vmax]]

radii_col_labels = ['NE','SE','SW','NW']
radii_row_labels = ['R34','R50','R64']

colors1 = plt.cm.Blues(np.linspace(0.2, 0.8, 7,endpoint=False)+0.5/7.)
colors2 = plt.cm.Greens(np.linspace(0.2, 0.8, 8,endpoint=False)+0.5/8.)
colors3 = plt.cm.YlOrRd(np.linspace(0.0, 0.25, 7,endpoint=False)+0.5/7.)
colors4 = plt.cm.Reds(np.linspace(0.5, 0.8, 8,endpoint=False)+0.5/8.)
colors5 = plt.cm.RdPu(np.linspace(0.3, 0.9, 14,endpoint=False)+0.5/14.)

# combine them and build a new colormap
cs = np.vstack((colors1, colors2, colors3, colors4, colors5))

bounds = np.hstack((np.arange(20,34,2),np.arange(34,50,2),np.arange(50,64,2),np.arange(64,96,4),np.arange(96,200,8)))
norm = colors.BoundaryNorm(boundaries=bounds,ncolors=len(bounds))
spd_ticks = [20,34,50,64,83,96,113,137]

mymap = colors.ListedColormap(cs)

line = Line2D([0], [0], label='RMW', color='k', linestyle='--')

fig = plt.figure(figsize=(8.5,7))
#fig = plt.figure(constrained_layout=True,figsize=(8,6))
gs = fig.add_gridspec(3,3,height_ratios=[1.0,0.05,1.0])
f_ax1 = fig.add_subplot(gs[0, 0])
f_ax2 = fig.add_subplot(gs[0, 1])
f_ax3 = fig.add_subplot(gs[0, 2])
f_ax4 = fig.add_subplot(gs[2, :-1])
f_ax5 = fig.add_subplot(gs[2, -1])
c1 = f_ax1.contourf(x_plot/1.852, y_plot/1.852, sfc_wind_pred*1.94, levels=bounds, norm=norm, cmap=mymap, extend='max');
c2 = f_ax2.contourf(x_plot/1.852, y_plot/1.852, mag_3km*1.94, levels=bounds, norm=norm, cmap=mymap, extend='max');
t1 = f_ax1.contour(x_plot/1.852, y_plot/1.852, sfc_wind_pred*1.94,colors=['k','k','k'],
                 linewidths=[0.35,0.7,1.15], levels=[83,113,137])
t2 = f_ax2.contour(x_plot/1.852, y_plot/1.852, mag_3km*1.94,colors=['k','k','k'],
                 linewidths=[0.35,0.7,1.15], levels=[83,113,137])
ln2, = f_ax2.plot(x_plane/1.852,y_plane/1.852,'k')
f_ax2.legend([ln2],['flight path'])
f_ax2.plot(x_plane[0]/1.852,y_plane[0]/1.852,'kx') # flight start
f_ax2.plot(x_plane[-1]/1.852,y_plane[-1]/1.852,'ko') # flight end
c3 = f_ax3.contourf(x_plot/1.852, y_plot/1.852, sfc_wind_pred/mag_3km, levels=np.arange(0.75,1.05,0.05), cmap='coolwarm', extend='both')
#f_ax1.contour(x_plot/1.852, y_plot/1.852, rd/1.852, levels=np.array([swann_rmw/1.852]), colors='k', linestyles='dotted');
#f_ax1.legend([line],['RMW'])
f_ax3.contour(x_plot/1.852, y_plot/1.852, rd/1.852, levels=np.array([swann_rmw/1.852]), colors='k', linestyles='dotted');
f_ax3.legend([line],['RMW'])
f_ax1.set_aspect('equal')
f_ax2.set_aspect('equal')
f_ax3.set_aspect('equal')
f_ax4.plot(hdobs.dt, hdobs.wsp, 'r')
f_ax4.plot(hdobs.dt, sfc_wind_pred_ac*1.94, color='#1E4D2B')
f_ax4.plot(hdobs.dt.values[0], hdobs.wsp.values[0], 'kx') # flight start
f_ax4.plot(hdobs.dt.values[-1], hdobs.wsp.values[-1], 'ko') # flight end
#f_ax4.plot(hdobs.dt, hdobs.sfmr, 'k',hdobs.dt, hdobs.wsp, 'r')
f_ax5.text(-0.075, 0.99, textstr, transform=f_ax5.transAxes, fontsize=10,verticalalignment='top')
my_table = f_ax5.table(cellText=np.round(vmax_table,decimals=1), 
                     rowLabels=vmax_row_labels,
                     colLabels=vmax_col_labels,
                     bbox=[0.15,0.3,0.8,0.375])
for (row, col), cell in my_table.get_celld().items():
    if (row == 2):
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        cell.get_text().set_color('#1E4D2B')

my_table2 = f_ax5.table(cellText=np.rint(radii_vals/1.852).astype(int), # convert radii from km to nm
                     rowLabels=radii_row_labels,
                     colLabels=radii_col_labels,
                     bbox=[0.15,-0.025,0.8,0.3])

for (row, col), cell in my_table2.get_celld().items():
    if (row == 0) | (col == -1):
        continue
    if ((radii_vals[row-1,col]/echo_edges[col]) > 0.95):
        cell.set_text_props(fontproperties=FontProperties(style='italic',weight='ultralight'))
        cell.get_text().set_color('red')

f_ax5.set_axis_off()
f_ax4.legend(['HDOBS FL','HDOBS SWANN'])
#f_ax4.legend(['SFMR (kt)','FL (kt)'])
f_ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
f_ax1.set_xlim([-85,85])
f_ax1.set_ylim([-85,85])
f_ax2.set_xlim([-85,85])
f_ax2.set_ylim([-85,85])
f_ax3.set_xlim([-85,85])
f_ax3.set_ylim([-85,85])
f_ax1.set_xticks([-80, -40, 0, 40, 80]);
f_ax1.set_yticks([-80, -40, 0, 40, 80]);
f_ax2.set_xticks([-80, -40, 0, 40, 80]);
f_ax2.set_yticks([-80, -40, 0, 40, 80]);
f_ax2.set_yticklabels([])
f_ax3.set_xticks([-80, -40, 0, 40, 80]);
f_ax3.set_yticks([-80, -40, 0, 40, 80]);
f_ax3.set_yticklabels([])
f_ax4.grid(True)
f_ax1.xaxis.set_minor_locator(AutoMinorLocator())
f_ax1.yaxis.set_minor_locator(AutoMinorLocator())
f_ax2.xaxis.set_minor_locator(AutoMinorLocator())
f_ax2.yaxis.set_minor_locator(AutoMinorLocator())
f_ax3.xaxis.set_minor_locator(AutoMinorLocator())
f_ax3.yaxis.set_minor_locator(AutoMinorLocator())
f_ax1.set_xlabel('distance from center (nm)');
f_ax1.set_ylabel('distance from center (nm)');
f_ax2.set_xlabel('distance from center (nm)');
f_ax3.set_xlabel('distance from center (nm)');
f_ax1.set_title('SWANN SFC wind (kt)');
f_ax2.set_title('SAMURAI FL wind (kt)');
f_ax3.set_title('ratio: SFC/FL');
f_ax4.set_ylabel('wind speed (kt)');
plt.suptitle(figtitle,y=0.915)
cb1 = plt.colorbar(mappable=c1,cax=fig.add_subplot(gs[1,:2]), orientation='horizontal',ticks=spd_ticks)
cb1.ax.set_title('');
cb1.add_lines(t1)
cb3 = plt.colorbar(mappable=c3,cax=fig.add_subplot(gs[1,2]), orientation='horizontal', ticks=[0.75, 0.85, 0.95, 1.05])
cb3.ax.set_title('');
fig.savefig(imDir+args.STORM+'_'+analysis_time+'_4pan.png', dpi=200, bbox_inches='tight')
#fig.savefig(imDir+args.STORM+'_'+args.ANALYSISTIME+'_4pan_'+ml_ver+mode+'.png', dpi=200, bbox_inches='tight')


