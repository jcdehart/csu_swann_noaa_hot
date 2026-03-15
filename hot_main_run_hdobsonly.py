#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:22:00 2023

@author: jcdehart
"""

#%% import necessary packages

import numpy as np
import pandas as pd
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import model_from_json
import model_utils
from geo_conversion import xy
import hot_grab_files
import hot_calc_centers
import center_funcs
import hot_prep_data
import save_files

#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("PLANE", help="plane: NOAA (N) or AF (A)", type=str)
parser.add_argument("--VDMLAT", default="0.0", help="VDM center lat", type=float)
parser.add_argument("--VDMLON", default="0.0", help="VDM center lon", type=float)
parser.add_argument("--CENFN", default="gfs.tXXz.syndata.tcvitals.tm00", help="TC Vitals filename", type=str)
parser.add_argument("--CENPATH", default="./ingest_dir/center_data", help="TC Vitals directory", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

if args.PLANE == 'A':
    af = True
elif args.PLANE == 'N':
    af = False
else:
    print('SPECIFY PLANE!!')

print(af)

ml_ver = 'FRED'

#%% set up dirs
# local testing
inDir = '/bell-scratch/jcdehart/hot_operational/csu_swann_noaa_hot/'
data_dir = inDir+'ingest_dir/'
ml_dir_base = inDir+'ML_models/'
hdobs_ingest_dir = inDir+'hdobs_parent/hdobs_input/'
output_dir = inDir+'nn_testing/'
imDir = inDir+'images/'

ml_dir = ml_dir_base + 'Current_HOT_Model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'


#samurai_time = pd.to_datetime(args.ANALYSISTIME,format='%Y%m%d%H%M',utc=True)
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
print('########')
print('center stats from tcvitals and adeck:')
print([storm_lat_1, storm_lon_1, storm_intens, storm_rmw, storm_dir, storm_motion, u_motion_1, v_motion_1, storm_dir_rot])
print([storm_lat_2, storm_lon_2, storm_intens_2, np.nan, storm_dir_2, storm_motion_2, u_motion_2, v_motion_2, storm_dir_rot_2])
#print('using tcvitals center')

# more center finding from files will go here... flight plus *********

# first remove any existing files
os.system('rm -rf '+hdobs_ingest_dir+'/*.list')
os.system('rm -rf '+hdobs_ingest_dir+'/*.hdob')
os.system('rm -rf '+hdobs_ingest_dir+'/*.gz')

print('\n')
print('########')
print('moving data over and reading HDOBS files')

hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
#print(hdobs_init)
hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name_2, af)
hot_grab_files.copy_files(hdobs_sm,hdobs_ingest_dir)

print(hdobs_sm.mission.unique())
if len(hdobs_sm.mission.unique()) > 1:
    print('warning: more than one flight ID in hdobs files!!')

# rename hdob .list files to .hdob for SAMURAI ingestion
os.system('for i in '+hdobs_ingest_dir+'/*.txt; do mv "$i" "${i%.txt}.hdob"; done')

# read in hdobs data
if af == True:
    hdobs, mission = hot_calc_centers.read_hdobs('KNHC', storm_name_2,'HDOBS', leg_start, leg_end)
elif af == False:
    hdobs, mission = hot_calc_centers.read_hdobs('KWBC', storm_name_2,'HDOBS', leg_start, leg_end)

print('avg, min, max altitude (km): '+str(hdobs.hgt.mean().round()/1000.)+', '+str(np.round(hdobs.hgt.min())/1000.)+', '+str(np.round(hdobs.hgt.max())/1000.))

print('avg, min, max p: '+str(hdobs.p.mean().round())+', '+str(np.round(hdobs.p.min()))+', '+str(np.round(hdobs.p.max())))

# run Chris's Willoughby-Chelmow algorithm
lat_wc, lon_wc, dt_wc, prominent = hot_calc_centers.run_wc(hdobs)

print('W-C center lat: '+str(lat_wc)+', center lon: '+str(lon_wc)+', time: '+dt_wc[0].strftime('%Y%m%d%H%M'))

# use VDM lat/lon if exists, or W-C (**** might avg later*****)
if (args.VDMLON != 0.0) & (args.VDMLAT != 0.0):
    storm_lon = args.VDMLON
    storm_lat = args.VDMLAT
    print('Using VDM center')
else:
    if prominent == True:
        print('using W-C center')
        storm_lon = lon_wc
        storm_lat = lat_wc
    elif (prominent == False) & (hdobs.dt.diff().max() < pd.Timedelta(10,'min')):
        print('no prominent peaks, no HDOBs gap (>10 min), using W-C center')
        storm_lon = lon_wc
        storm_lat = lat_wc
    elif (prominent == False) & (hdobs.dt.diff().max() >= pd.Timedelta(10,'min')):
        print('no prominent peaks, HDOBs gap (>10 min), using a-deck center')
        storm_lon = storm_lon_2
        storm_lat = storm_lat_2

# keeping averaging in case we want it in the future
#print('averaging all 3 centers')
# wgt = np.array([1, 1, 3])
#storm_lon = np.average(np.array([lon_wc,storm_lon_1,storm_lon_2]),weights=wgt)
#storm_lat = np.average(np.array([lat_wc,storm_lat_1,storm_lat_2]),weights=wgt)

u_motion = np.nanmean(np.array([u_motion_1,u_motion_2]))
v_motion = np.nanmean(np.array([v_motion_1,v_motion_2]))
#print([storm_lat, storm_lon])
print([u_motion, v_motion])

# grab plane altitude manually
alt_plane = hdobs.hgt # in meters
print('using height time series for HDOBs data')

# convert hdobs to xy
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,storm_lat,storm_lon)

#%% main code: step 3 - neural net

print('\n')
print('########')
print('run SWANN on HDOBS')

# load json and create model
json_file = open(ml_dir+json_fn, 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+ml_file)
print("Loaded model from disk")

# create theta/radius grids 
rd = np.sqrt(x_plane**2 + y_plane**2)
th_r = np.arctan2(y_plane, x_plane)
th = th_r*180./np.pi

wspd_earth = hdobs.wsp.values/1.94 # CONVERTING TO M/S NEEDED FOR ALEX'S MODEL ******
hdobs_rmw = rd[np.unravel_index(np.nanargmax(wspd_earth),np.shape(wspd_earth))]

# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('\n')
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('hdobs FL RMW: '+str(hdobs_rmw))
print('hdobs FL intens: '+str(np.nanmax(wspd_earth)))

# prepare variables for NN model - SAMURAI wind field
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km (m for HDOBs), is HDOBS or SAM?
X_ratio, r_norm = hot_prep_data.process_nn_vars(rd, hdobs_rmw, th, storm_dir, storm_intens, storm_motion, wspd_earth, alt_plane, True)

# standardize data
x_data = model_utils.Standardize_Vars(X_ratio.T)

# make prediction with the neural net
predict = nn_model.predict(x_data)
predict[r_norm < 0.3] = np.nan

# reshape arrays and mask orig missing data
sfc_wind_pred = wspd_earth*predict.T[0] # multiply reduction factor and flight-level wind

# grab flight-level storm-relative data and remove bad data
mag_3km = wspd_earth
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
sfc_wind_pred[mag_3km*1.94 < 20] = np.nan ##### UNITS ALREADY IN KTS
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
mag_3km[(rd/hdobs_rmw < 0.3)] = np.nan
print('/n')
print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

# grab RMW
swann_rmw = rd[np.unravel_index(np.nanargmax(sfc_wind_pred),np.shape(sfc_wind_pred))]

# ****** CHANGE TO JUST SPEED! ********
# convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012, use th_r in radians
u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th_r) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th_r)
v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th_r) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th_r)

#%% main code: step 4 - prep for saving files

hdobs_fl_vmax = np.nanmax(hdobs.wsp)
swann_hdobs_vmax = np.nanmax(sfc_wind_pred*1.94)

# determine simplified franklin reduction based on altitude of peak HDOBs wind
med_hgt = 500*(alt_plane[np.nanargmax(hdobs.wsp)]/500).round()

if med_hgt == 1500.:
    sf_frac = 0.8
elif med_hgt == 3000.:
    sf_frac = 0.9
else:
    hgt_options = np.array([1500., 3000.])
    alt_tmp = hgt_options[np.argmin(np.abs(med_hgt - hgt_options))]
    
    if alt_tmp == 1500.:
        sf_frac = 0.8
    elif alt_tmp == 3000.:
        sf_frac = 0.9

simp_frank = sf_frac*hdobs_fl_vmax

# set up info for figure
figtitle = storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + mission + ' | ' + leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M') + ' UTC'

textstr = '\n'.join((
    'Inputs: HDOBS',
    'W-C Center: %.2f N, %.2f W' % (storm_lat,np.abs(storm_lon),), # assuming western hemisphere
    'RMW: %.1f (nm)' % (swann_rmw/1.852,),
    'Simp. Franklin: %.1f (kt)' % (simp_frank,), ))

#%% main code: step 5 - save all files

print('\n')
print('########')
print('save txt file, netcdf, image')

# save netcdf file
save_files.save_1d_netcdf(hdobs, u_nc, v_nc, samurai_time, args)

x_plot, y_plot = np.meshgrid(np.arange(np.nanmin(x_plane),np.nanmax(x_plane)), 
    np.arange(np.nanmin(y_plane),np.nanmax(y_plane)))
radii = np.sqrt(x_plot**2 + y_plot**2)

# wind radii calculations

wind_radii = [34,50,64]
radii_vals = np.zeros((3,4)) # NE, SE, SW, NW

for i in range(len(wind_radii)):
    radii_vals[i,0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plane < 0) | (y_plane < 0), np.nan, rd))
    radii_vals[i,1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plane < 0) | (y_plane > 0), np.nan, rd))
    radii_vals[i,2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plane > 0) | (y_plane > 0), np.nan, rd))
    radii_vals[i,3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plane > 0) | (y_plane < 0), np.nan, rd))

# deal with NaN issue
radii_vals_nm = np.rint(radii_vals/1.852) # convert radii from km to nm
radii_vals_nm[np.isnan(radii_vals_nm)] = -999
radii_vals_str = radii_vals_nm.astype(int).astype(str)
radii_vals_str[np.isin(radii_vals_str,'-999')] = 'N/A'

echo_edges = np.zeros(4)
echo_edges[0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plane < 0) | (y_plane < 0), np.nan, rd))
echo_edges[1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plane < 0) | (y_plane > 0), np.nan, rd))
echo_edges[2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plane > 0) | (y_plane > 0), np.nan, rd))
echo_edges[3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plane > 0) | (y_plane < 0), np.nan, rd))

vmax_table = [[hdobs_fl_vmax],[swann_hdobs_vmax]]

# save text file
save_files.save_txt(storm_lat, storm_lon, hdobs_fl_vmax, swann_hdobs_vmax, swann_rmw, simp_frank, radii_vals_nm, echo_edges,
                    inDir, args, analysis_time, 'HDOBS')


# save image
save_files.plot_image_2pan(x_plane, y_plane, sfc_wind_pred, hdobs, radii_vals_str, radii_vals, echo_edges, 
                           textstr, vmax_table, figtitle, args, imDir, samurai_time)