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

#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

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
inDir = '/bell-scratch/jcdehart/hot_operational/csu_swann_noaa_hot/'
ml_dir = inDir+'ml_model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'
sam_dir_base = inDir+'samurai_parent/'
sam_ingest_dir = inDir+ext+'samurai_parent/samurai_input/'
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
    leg_start = pd.to_datetime('202510281328',format='%Y%m%d%H%M',utc=True)
    leg_end = pd.to_datetime('202510281403',format='%Y%m%d%H%M',utc=True)

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
# **** for now specifying whether AF only or not ***** UPDATE IN FUTURE

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

# use this in final comparison with objective center
wc_good = True
vdm_good = False

# use VDM lat/lon if exists, or W-C (**** might avg later*****)
if (args.VDMLON != 0.0) & (args.VDMLAT != 0.0):
    storm_lon = args.VDMLON
    storm_lat = args.VDMLAT
    print('Using VDM center')
    wc_good = False # maybe add similar param for VDM? would assume vdm is good though....
    vdm_good = True # might include a check for these cases...
else:
    if (prominent == True) & (hdobs.dt.diff().max() < pd.Timedelta(10,'min')):
        print('using W-C center')
        storm_lon = lon_wc
        storm_lat = lat_wc
    # elif (prominent == True) & (hdobs.dt.diff().max() >= pd.Timedelta(10,'min')):
    #     print('HDOBs gap (>10 min), using a-deck center')
    #     storm_lon = storm_lon_2
    #     storm_lat = storm_lat_2
    #     wc_good = False
    elif (prominent == False) & (hdobs.dt.diff().max() < pd.Timedelta(10,'min')):
        print('no prominent peaks, no HDOBs gap (>10 min), using W-C center')
        storm_lon = lon_wc
        storm_lat = lat_wc
    elif (prominent == False) & (hdobs.dt.diff().max() >= pd.Timedelta(10,'min')):
        print('no prominent peaks, HDOBs gap (>10 min), using a-deck center')
        storm_lon = storm_lon_2
        storm_lat = storm_lat_2
        wc_good = False

# keeping averaging in case we want it in the future
#print('averaging all 3 centers')
# wgt = np.array([1, 1, 3])
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
    hgt_options = np.array([1500, 3000])
    alt_plane = hgt_options[np.argmin(np.abs(med_hgt - hgt_options))]/1000
    print('plane hgt not within 500 m of options')
    print('using plane hgt = '+str(alt_plane))
    print('actual med plane hgt = '+str(med_hgt/1000))


#%% main code: step 2 - run samurai

print('\n')
print('########')
print('running SAMURAI')

# create center file
ref_latlon_cart = make_cen_file(samurai_time, leg_start, leg_end, storm_lat, storm_lon, u_motion, v_motion, './samurai_parent/samurai_input/')

# generate samurai params file from master
analysis_dir_cart = modify_param_file(samurai_time, './samurai_parent/master_params/samurai_HOT_cart.params', './samurai_parent/samurai_params_cart')
sam_dir_cart = sam_dir_base + analysis_dir_cart +'_cart/'
os.system('mkdir -p '+sam_dir_cart)

# run samurai in XYZ mode
os.system('/bell-scratch/mmbell/hot/samurai-hot/release/bin/samurai -params ./samurai_parent/samurai_params_cart')

# move files to analysis_dir
os.system('mv ./samurai_parent/samurai_params_cart '+sam_dir_cart)
os.system('mv ./samurai_parent/samurai_input/*.cen '+sam_dir_cart)
os.system('mv ./samurai_parent/samurai_input/*.in '+sam_dir_cart)


#%% grab center from SAMURAI analysis
# fix link to be more adaptable
obj_master = './samurai_parent/master_params/objective_simplex.jl'
cart_file = sam_dir_cart+'samurai_XYZ_analysis.nc'
hot_calc_centers.modify_obj_jl_file(obj_master, './objective_simplex.jl', storm_rmw, cart_file)

print('\n')
print('########')
print('calculate simplex center')

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

# check for distance from W-C center *** fix later??? *******
if ((np.abs(sam_lon_tmp - lon_wc) > 0.4) | (np.abs(sam_lat_tmp - lat_wc) > 0.4)) & wc_good:
    print('objective center too far from W-C, W-C good, defaulting to W-C center')
    sam_lon = lon_wc
    sam_lat = lat_wc
    xc = np.interp(lon_wc, ncfile_cart['longitude'][:].data, ncfile_cart['x'][:].data)
    yc = np.interp(lat_wc, ncfile_cart['latitude'][:].data, ncfile_cart['y'][:].data)
    wccen = True # set this to calc rmw later
else:
    print('objective center seems reasonable or W-C bad')
    sam_lon = sam_lon_tmp
    sam_lat = sam_lat_tmp
    xc = xc_avg
    yc = yc_avg
    wccen = False


print('samurai center lat: '+str(sam_lat)+', center lon: '+str(sam_lon))

# convert hdobs to xy
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,sam_lat,sam_lon)

# name files
sam_fn = 'samurai_XYZ_analysis.nc'
sam_dir = sam_dir_cart

#%% main code: step 3 - neural net

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
u_storm, v_storm, lon_nc, lat_nc, th, th_nc, rd, X, Y = hot_prep_data.read_netcdf(sam_dir, sam_fn, ncvars, alt_plane, [xc, yc])
    
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
predict[r_norm < 0.3] = np.nan # remove data within radius of 0.3*RMW where SWANN shouldn't be applied
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
predict_ac[r_norm_ac < 0.3] = np.nan

# reshape arrays and mask orig missing data
sfc_wind_pred_ac = wspd_earth_ac*predict_ac.T[0] # multiply reduction factor and flight-level wind

# grab flight-level storm-relative data and remove bad data
mag_3km_ac = wspd_earth_ac
sfc_wind_pred_ac[np.isnan(mag_3km_ac)] = np.nan
sfc_wind_pred_ac[mag_3km_ac*1.94 < 20] = np.nan ##### UNITS ALREADY IN KTS
sfc_wind_pred_ac[np.isnan(mag_3km_ac)] = np.nan
mag_3km_ac[(rd_ac/sam_rmw < 0.3)] = np.nan # remove data within radius of 0.3*RMW where SWANN shouldn't be applied

#%% main code: step 4 - prep for file saving

if alt_plane == 1.5:
    sf_frac = 0.8
elif alt_plane == 3.0:
    sf_frac = 0.9

sam_fl_vmax = np.nanmax(wspd_earth*1.94)
hdobs_fl_vmax = np.nanmax(hdobs.wsp)
swann_sam_vmax = np.nanmax(sfc_wind_pred*1.94)
swann_hdobs_vmax = np.nanmax(sfc_wind_pred_ac*1.94)
simp_frank = sf_frac*sam_fl_vmax

figtitle = storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + mission + ' | ' + leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M') + ' UTC'

textstr = '\n'.join((
    'Inputs: HRD TDR, HDOBS',
    'SAM Center: %.2f N, %.2f W' % (sam_lat,np.abs(sam_lon),), # currently assuming negative longitudes **********
    'RMW: %.1f (nm)' % (swann_rmw/1.852,),
    'Simp. Franklin: %.1f (kt)' % (simp_frank,), ))

# convert coords, first to cartesian
x_plot = X
y_plot = Y

#%% main code: step 5 - generate any images

print('\n')
print('########')
print('save txt file, netcdf, image')

# save netcdf file
save_files.save_2d_netcdf(lat_nc, lon_nc, u_nc, v_nc, samurai_time, analysis_time, args)

# wind radii calculations

wind_radii = [34,50,64]
radii_vals = np.zeros((3,4)) # NE, SE, SW, NW

for i in range(len(wind_radii)):
    radii_vals[i,0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot < 0) | (y_plot < 0), np.nan, rd))
    radii_vals[i,1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot < 0) | (y_plot > 0), np.nan, rd))
    radii_vals[i,2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot > 0) | (y_plot > 0), np.nan, rd))
    radii_vals[i,3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (1.94*sfc_wind_pred < wind_radii[i]) | (x_plot > 0) | (y_plot < 0), np.nan, rd))

# deal with NaN issue
radii_vals_nm = np.rint(radii_vals/1.852) # convert radii from km to nm
radii_vals_nm[np.isnan(radii_vals_nm)] = -999
radii_vals_str = radii_vals_nm.astype(int).astype(str)
radii_vals_str[np.isin(radii_vals_str,'-999')] = 'N/A'

echo_edges = np.zeros(4)
echo_edges[0] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot < 0) | (y_plot < 0), np.nan, rd))
echo_edges[1] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot < 0) | (y_plot > 0), np.nan, rd))
echo_edges[2] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot > 0) | (y_plot > 0), np.nan, rd))
echo_edges[3] = np.nanmax(np.where(np.isnan(sfc_wind_pred) | (x_plot > 0) | (y_plot < 0), np.nan, rd))

### EDGES RIGHT NOW IN KM, FIX OR CONVERT TO NM
# affect save_txt and plot_image_4pan (and AF code)

vmax_table = [[hdobs_fl_vmax,sam_fl_vmax],[swann_hdobs_vmax, swann_sam_vmax]]

# save output text file
save_files.save_txt(sam_lat, sam_lon, sam_fl_vmax, swann_sam_vmax, sam_rmw, simp_frank, radii_vals_nm, echo_edges,
                    inDir, args, analysis_time, 'SAM')


# save figure
save_files.plot_image_4pan(x_plot, y_plot, rd, x_plane, y_plane, sfc_wind_pred, mag_3km, sfc_wind_pred_ac, hdobs, swann_rmw,
                           radii_vals_str, radii_vals, echo_edges, textstr, vmax_table, figtitle, args, imDir, analysis_time)
