#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:22:00 2023

@author: jcdehart
"""

#%% import necessary packages

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import model_from_json
import model_utils
from samurai_gen_file import make_cen_file, modify_param_file
from geo_conversion import xy
import hot_grab_files
import hot_calc_centers
import hot_prep_data
import save_files

#%% #### main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers) ####

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--MODE", default="normal", help="run mode (test or normal)", type=str)
parser.add_argument("--VDMLAT", default="0.0", help="VDM center lat", type=float)
parser.add_argument("--VDMLON", default="0.0", help="VDM center lon", type=float)
parser.add_argument("--CENFN", default="gfs.tXXz.syndata.tcvitals.tm00", help="TC Vitals filename", type=str)
parser.add_argument("--CENPATH", default="./ingest_dir/center_data", help="TC Vitals directory", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

af = False
mode = args.MODE

if mode == 'test':
    ext = 'testing/output/'
else:
    ext = ''

#%% set up dirs
inDir = './'
ml_dir = inDir+'ml_model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'
sam_dir_base = inDir+'samurai_parent/'
sam_ingest_dir = inDir+'samurai_parent/samurai_input/'
sam_bin = '/bell-scratch/mmbell/hot/samurai-hot/release/bin/samurai'
output_dir = inDir+ext+'nn_testing/'
imDir = inDir+ext+'images/'

# make sure dirs exist
os.system('mkdir -p '+sam_ingest_dir)
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

# move all necessary files to ./samurai_input
# first remove any existing files
os.system('rm -rf '+sam_ingest_dir+'/*.list')
os.system('rm -rf '+sam_ingest_dir+'/*.hdob')
os.system('rm -rf '+sam_ingest_dir+'/*.gz')

print('\n')
print('########')
print('moving data over and reading HDOBS files')

# move radials and HDOBS
hrd_init = hot_grab_files.create_dataframe(data_dir+'hrd_radials',leg_start,leg_end)
hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
hrd_sm = hot_grab_files.shrink_df(hrd_init, leg_start, leg_end, storm_name_2, af)
hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name_2, af)
hot_grab_files.copy_files(hrd_sm,sam_ingest_dir)
hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)
os.system('for i in '+sam_ingest_dir+'/*.gz; do gunzip $i; done')

print(hdobs_sm.mission.unique())
if len(hdobs_sm.mission.unique()) > 1:
    print('warning: more than one flight ID in hdobs files!!')

# rename hdob .list files to .hdob for SAMURAI ingestion
os.system('for i in '+sam_ingest_dir+'/*.txt; do mv "$i" "${i%.txt}.hdob"; done')

# read HDOBS
hdobs, mission = hot_calc_centers.read_hdobs('KWBC', storm_name_2,'SAMURAI', leg_start, leg_end)

print('avg, min, max altitude (km): '+str(np.round(hdobs.hgt.mean())/1000.)+', '+str(np.round(hdobs.hgt.min())/1000.)+', '+str(np.round(hdobs.hgt.max())/1000.))

print('avg, min, max p: '+str(np.round(hdobs.p.mean()))+', '+str(np.round(hdobs.p.min()))+', '+str(np.round(hdobs.p.max())))

# run Chris's Willoughby-Chelmow algorithm
lat_wc, lon_wc, dt_wc, prominent = hot_calc_centers.run_wc(hdobs)

print('W-C center lat: '+str(lat_wc)+', center lon: '+str(lon_wc)+', time: '+dt_wc.strftime('%Y%m%d%H%M'))

# choose flight level center from VDM, W-C, and assess goodness of W-C center
storm_lat, storm_lon, wc_good, vdm_good = hot_calc_centers.choose_fl_cen(args, prominent, hdobs, [lat_wc, lon_wc], 
                                                                         [storm_lat_2, storm_lon_2])

# average motion vectors from tcvitals and a-deck (REVISIT*******)
u_motion = np.nanmean(np.array([u_motion_1,u_motion_2]))
v_motion = np.nanmean(np.array([v_motion_1,v_motion_2]))
print([storm_lat, storm_lon])
print([u_motion, v_motion])

# set approximate height of P3 during flight leg (closest to 1.5 or 3 km)
# for selecting SAMURAI level, HDOBS SWANN takes realtime altitude
alt_plane = hot_prep_data.grab_p3_alt(hdobs)


#%% #### main code: step 2 - run samurai ####

print('\n')
print('########')
print('running SAMURAI')

# create center file
ref_latlon = make_cen_file(samurai_time, leg_start, leg_end, storm_lat, storm_lon, u_motion, v_motion, './samurai_parent/samurai_input/')

# generate samurai params file from master
analysis_dir = modify_param_file(samurai_time, ext, './samurai_parent/master_params/samurai_HOT_cart.params', './samurai_parent/samurai_params_cart')
sam_dir = './' + analysis_dir +'_cart/'
os.system('mkdir -p '+sam_dir)

# run samurai in XYZ mode
os.system(sam_bin+' -params ./samurai_parent/samurai_params_cart')

# move files to analysis_dir
os.system('mv ./samurai_parent/samurai_params_cart '+sam_dir)
os.system('mv ./samurai_parent/samurai_input/*.cen '+sam_dir)
os.system('mv ./samurai_parent/samurai_input/*.in '+sam_dir)


#%% grab center from SAMURAI analysis
# fix link to be more adaptable
obj_master = './samurai_parent/master_params/objective_simplex.jl'
cart_file = sam_dir+'samurai_XYZ_analysis.nc'
hot_calc_centers.modify_obj_jl_file(obj_master, './objective_simplex.jl', storm_rmw, cart_file)

print('\n')
print('########')
print('calculate simplex center')

# run julia simplex code
os.system('sh run_julia.sh')

# read simplex output, avg layer around flight altitude, interpolate to lat/lon,
# check for distance from W-C center, assuming it's good, to see if simplex center good
sam_lon, sam_lat, xc, yc, wccen, rmw_avg = hot_calc_centers.process_simplex_cen('samurai_center.nc', alt_plane, cart_file, 
                                                                                [lat_wc, lon_wc], wc_good)

# convert hdobs to xy
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,sam_lat,sam_lon)


#%% #### main code: step 3 - run SWANN ####

print('\n')
print('########')
print('run SWANN on SAMURAI output and HDOBS')

# read and process samurai file
ncvars = { 
  'alt': 'altitude', 'x': 'x', 'y': 'y',
  'lon': 'longitude', 'lat': 'latitude',
  'u': 'U', 'v': 'V'
}

# read SAMURAI file, recenter, calculate radius, angle
u_storm, v_storm, lon_nc, lat_nc, th, th_nc, rd, X, Y = hot_prep_data.read_netcdf(sam_dir, 'samurai_XYZ_analysis.nc', 
                                                                                  ncvars, alt_plane, [xc, yc])
    
# add storm motion, calculate wind speed
wspd_earth = hot_prep_data.calc_wspd_earth(u_storm, v_storm, u_motion, v_motion, True)

# calculate rmw from max SAMURAI point if reverted to W-C center above (testing lol)
if wccen == True:
    sam_rmw = np.nanmin(rd[wspd_earth == np.nanmax(wspd_earth)])
elif wccen == False:
    sam_rmw = rmw_avg

# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('\n')
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('samurai FL RMW: '+str(sam_rmw))
print('samurai FL intens: '+str(np.nanmax(wspd_earth)))

# prepare variables for NN model - SAMURAI wind field
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km (m for HDOBs)
X_ratio, r_norm = hot_prep_data.process_nn_vars(rd, sam_rmw, th, storm_dir, storm_intens, storm_motion, wspd_earth, alt_plane, af)

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

# postprocess model prediction (remove data < 0.3 R* or 3-km wind < 20 kts, 
# 4 grid spaces from boundary edges due to spectral ringing, 
# and converting to u/v assuming 22.6º inflow angle from zhang and uhlhorn)
mag_3km, sfc_wind_pred, swann_rmw, u_nc, v_nc = hot_prep_data.postprocess_swann_sam(r_norm, wspd_earth, predict, u_storm, v_storm, rd, sam_rmw, th_nc)


#%% ### run aircraft data through model ###
# calculate radii, angle, windspeed (**in m/s needed for SWANN**), and RMW for in situ obs
rd_ac, th_ac, wspd_earth_ac, hdobs_rmw_ac = hot_prep_data.prep_hdobs_data(hdobs, x_plane, y_plane)

# prepare variables for NN model - HDOBs wind field
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km (m for HDOBs)
X_ratio_ac, r_norm_ac = hot_prep_data.process_nn_vars(rd_ac, sam_rmw, th_ac, storm_dir, storm_intens, storm_motion, wspd_earth_ac, hdobs.hgt.values, True)

# standardize data
x_data_ac = model_utils.Standardize_Vars(X_ratio_ac.T)

# make prediction with the neural net
predict_ac = nn_model.predict(x_data_ac)

# postprocess model prediction (remove data < 0.3 R* or 3-km wind < 20 kts, grab rmw, convert wind to m/s)
sfc_wind_pred_ac, swann_rmw_ac, sfc_wind_pred_ms_ac = hot_prep_data.postprocess_swann_af(r_norm_ac, wspd_earth_ac, predict_ac, rd_ac)


#%% #### main code: step 4 - prep for file saving ####

# calculate vmax values needed and convert remaining vars to kts
sam_fl_vmax, hdobs_fl_vmax, swann_sam_vmax, swann_hdobs_vmax, simp_frank = hot_prep_data.vmax_calcs_sam(alt_plane, wspd_earth, hdobs, sfc_wind_pred, sfc_wind_pred_ac)

# create text strings for image
figtitle, textstr = hot_prep_data.create_fig_str(storm_name_2, mission, leg_start, leg_end, sam_lat, sam_lon, swann_rmw, simp_frank, 'N')

#%% #### main code: step 5 - generate any images ####

print('\n')
print('########')
print('save txt file, netcdf, image')

# save netcdf file
save_files.save_2d_netcdf(lat_nc, lon_nc, u_nc, v_nc, samurai_time, analysis_time, args)

# calculate wind radii and echo edges
### EDGES RIGHT NOW IN KM, FIX OR CONVERT TO NM
# affect save_txt and plot_image_4pan (and AF code)
fl_vmax = [hdobs_fl_vmax,sam_fl_vmax]
swann_vmax = [swann_hdobs_vmax, swann_sam_vmax]
radii_vals, radii_vals_nm, radii_vals_str, echo_edges, vmax_table = save_files.calc_radii_edges(sfc_wind_pred, X, Y, rd, fl_vmax, swann_vmax)

# save output text file
save_files.save_txt(sam_lat, sam_lon, sam_fl_vmax, swann_sam_vmax, sam_rmw, simp_frank, radii_vals_nm, echo_edges,
                    inDir, args, analysis_time, 'SAM')

# save figure
save_files.plot_image_4pan(X, Y, rd, x_plane, y_plane, sfc_wind_pred, mag_3km, sfc_wind_pred_ac, hdobs, swann_rmw,
                           radii_vals_str, radii_vals, echo_edges, textstr, vmax_table, figtitle, args, imDir, analysis_time)
