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
import hot_prep_data
import save_files

#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("PLANE", help="plane: NOAA (N) or AF (A)", type=str)
parser.add_argument("--MODE", default="normal", help="run mode (test or normal)", type=str)
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

mode = args.MODE

if mode == 'test':
    ext = 'testing/output/'
else:
    ext = ''

print(af)

#%% set up dirs
# local testing
inDir = '/bell-scratch/jcdehart/hot_operational/csu_swann_noaa_hot/'
ml_dir = inDir+'ml_model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'
hdobs_ingest_dir = inDir+ext+'hdobs_parent/hdobs_input/'
output_dir = inDir+ext+'nn_testing/'
imDir = inDir+ext+'images/'

# make sure dirs exist
os.system('mkdir -p '+hdobs_ingest_dir)
os.system('mkdir -p '+output_dir)
os.system('mkdir -p '+imDir)

# set up mode specific paths/vars
if mode == 'normal':
    data_dir = inDir+'ingest_dir/'
    leg_start = pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True)
    leg_end = pd.to_datetime(args.ENDTIME,format='%Y%m%d%H%M',utc=True)
elif mode == 'test':
    data_dir = inDir+'testing/data/'
    args.CENPATH = './testing/data/center_data' # overwrite default, but consider removing entirely
    leg_start = pd.to_datetime('202510281328',format='%Y%m%d%H%M',utc=True)
    leg_end = pd.to_datetime('202510281403',format='%Y%m%d%H%M',utc=True)
    args.STARTTIME = leg_start.strftime('%Y%m%d%H%M')
    args.STORM = 'AL13'

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

print('avg, min, max altitude (km): '+str(np.round(hdobs.hgt.mean())/1000.)+', '+str(np.round(hdobs.hgt.min())/1000.)+', '+str(np.round(hdobs.hgt.max())/1000.))

print('avg, min, max p: '+str(np.round(hdobs.p.mean()))+', '+str(np.round(hdobs.p.min()))+', '+str(np.round(hdobs.p.max())))

# run Chris's Willoughby-Chelmow algorithm
lat_wc, lon_wc, dt_wc, prominent = hot_calc_centers.run_wc(hdobs)

print('W-C center lat: '+str(lat_wc)+', center lon: '+str(lon_wc)+', time: '+dt_wc.strftime('%Y%m%d%H%M'))

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

# calculate radii, angle, windspeed (in m/s needed for SWANN), and RMW
rd, th, wspd_earth, hdobs_rmw = hot_prep_data.prep_hdobs_data(hdobs, x_plane, y_plane)

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

# load json and create model
json_file = open(ml_dir+json_fn, 'r')
loaded_model_json = json_file.read()
json_file.close()
nn_model = model_from_json(loaded_model_json)

# load weights into new model
nn_model.load_weights(ml_dir+ml_file)
print("Loaded model from disk")

# make prediction with the neural net
predict = nn_model.predict(x_data)
predict[r_norm < 0.3] = np.nan # remove data within radius of 0.3*RMW where SWANN shouldn't be applied

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

# unit conversion
sfc_wind_pred_ms = sfc_wind_pred*1.94 # convert to m/s

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
save_files.save_1d_netcdf(hdobs, sfc_wind_pred_ms, samurai_time, args)

# calculate wind radii and echo edges
### EDGES RIGHT NOW IN KM, FIX OR CONVERT TO NM
# affect save_txt and plot_image_4pan (and SAM code)
fl_vmax = [hdobs_fl_vmax]
swann_vmax = [swann_hdobs_vmax]
radii_vals, radii_vals_nm, radii_vals_str, echo_edges, vmax_table = save_files.calc_radii_edges(sfc_wind_pred, x_plane, y_plane, rd, fl_vmax, swann_vmax)

# save text file
save_files.save_txt(storm_lat, storm_lon, hdobs_fl_vmax, swann_hdobs_vmax, swann_rmw, simp_frank, radii_vals_nm, echo_edges,
                    inDir, args, analysis_time, 'HDOBS')

# save image
save_files.plot_image_2pan(x_plane, y_plane, sfc_wind_pred, hdobs, radii_vals_str, radii_vals, echo_edges, 
                           textstr, vmax_table, figtitle, args, imDir, samurai_time)