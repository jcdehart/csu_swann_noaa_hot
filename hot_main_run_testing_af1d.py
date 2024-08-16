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
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as colors
import matplotlib.dates as mdates

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
    dlon = lon_plane - lon_cen
    dlat = lat_plane - lat_cen
    a = np.sin(dlat / 2)**2 + np.cos(lat_cen) * np.cos(lat_plane) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    bearing = np.arctan2(np.sin(lon_plane-lon_cen)*np.cos(lat_plane), np.cos(lat_cen)*np.sin(lat_plane)-np.sin(lat_cen)*np.cos(lat_plane)*np.cos(lon_plane-lon_cen))
    x = distance*np.cos(bearing)
    y = distance*np.sin(bearing)
    return(x,y)

#%% main code: step 1 - make center file from tcvitals of flight+ file (hot_calc_centers)

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("STORM", help="storm name (all caps)", type=str)
#parser.add_argument("CENTIME", help="cen datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--CENFN", default="gfs.tXXz.syndata.tcvitals.tm00", help="TC Vitals filename", type=str)
parser.add_argument("--CENPATH", default="./ingest_dir/center_data", help="TC Vitals directory", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

# ***** change into an argument, or change to start time + duration, model will be settled in future, deal with af only later
#dur = 45
cyl = False
af = True
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
#ml_dir_base = inDir+'ML_models/stable_model/'
sam_dir_base = inDir+'samurai_parent/'
sam_ingest_dir = inDir+'samurai_parent/samurai_input/'
output_dir = inDir+'nn_testing/'
imDir = inDir+'images/'

ml_dir = ml_dir_base + 'Current_HOT_Model/'
ml_file = 'HS24_SCL_2DNN_model_v2.h5'
json_fn = 'HS24_SCL_2DNN_model_v2.json'


samurai_time = pd.to_datetime(args.ANALYSISTIME,format='%Y%m%d%H%M',utc=True)
leg_start = pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True)
leg_end = pd.to_datetime(args.ENDTIME,format='%Y%m%d%H%M',utc=True)
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
#print('using tcvitals center')

# more center finding from files will go here... flight plus *********


# calc W-C center (hdobs)
# ******* I think code adds buffer on to beginning and end - potentially change *******

# move all necessary files to ./samurai_input
# **** for now specifying whether AF only or not ***** UPDATE IN FUTURE

# first remove any existing files
os.system('rm -rf '+sam_ingest_dir+'/*.list')
os.system('rm -rf '+sam_ingest_dir+'/*.hdob')
os.system('rm -rf '+sam_ingest_dir+'/*.gz')

hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name_2, af)
hot_grab_files.copy_files(hdobs_sm,sam_ingest_dir)

print(hdobs_sm.mission.unique())
if len(hdobs_sm.mission.unique()) > 1:
    print('warning: more than one flight ID in hdobs files!!')

# rename hdob .list files to .hdob for SAMURAI ingestion
os.system('for i in '+sam_ingest_dir+'/*.txt; do mv "$i" "${i%.txt}.hdob"; done')

# read in hdobs data
hdobs = hot_calc_centers.read_hdobs('KNHC', storm_name_2)

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
#print('W-C center lat: '+str(lat_wc[0])+', center lon: '+str(lon_wc[0]))
print(dt_wc)

print('averaging all 3 centers')

# perhaps have different weights based on tcvitals intensity??? ********
wgt = np.array([1, 1, 3])
storm_lon = np.average(np.array([lon_wc,storm_lon_1,storm_lon_2]),weights=wgt)
storm_lat = np.average(np.array([lat_wc,storm_lat_1,storm_lat_2]),weights=wgt)
#storm_lon = np.nanmean(np.array([lon_wc,storm_lon_1,storm_lon_2]))
#storm_lat = np.nanmean(np.array([lat_wc,storm_lat_1,storm_lat_2]))
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

# convert hdobs to xy
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,lat_wc,lon_wc)

#%% main code: step 3 - neural net

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

## RMW
# normalize wrt RMW and ravel data
r_norm = rd.ravel(order='C')/hdobs_rmw

# compare RMW values ( ***edit for coverage in samurai analysis*** )
print('\n')
print('tcvitals RMW: '+str(storm_rmw))
print('tcvitals intens: '+str(storm_intens))
print('hdobs FL RMW: '+str(hdobs_rmw))
print('hdobs FL intens: '+str(np.nanmax(wspd_earth)))

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
alt[:] = alt_plane*1000 # m

BT_Vmax = np.zeros_like(FL_wind)
BT_Vmax[:] = storm_intens*1.94 # convert to knots
SM_mag = np.zeros_like(FL_wind)
SM_mag[:] = storm_motion
RMW_arr = np.zeros_like(FL_wind)
RMW_arr[:] = hdobs_rmw

# normalized r (r/rmw), theta (wrt motion) x2, wind, altitude, vmax, storm motion magnitude
X_ratio = np.asarray([r_norm, theta_nr, theta_nr, FL_wind, alt, BT_Vmax, SM_mag, RMW_arr])  #double up on angle for sin and cosine
X_ratio[1,:] = np.sin(X_ratio[1,:]) # converted to radians above
X_ratio[2,:] = np.cos(X_ratio[2,:])

# standardize data
x_data = model_utils.Standardize_Vars(X_ratio.T)

# make prediction with the neural net
predict = nn_model.predict(x_data)
predict[r_norm < 0.3] = np.nan
#predict[r_norm < 0.5] = np.nan

# reshape arrays and mask orig missing data
sfc_wind_pred = wspd_earth*predict.T[0] # multiply reduction factor and flight-level wind

# grab flight-level storm-relative data and remove bad data
mag_3km = wspd_earth
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
sfc_wind_pred[mag_3km*1.94 < 20] = np.nan ##### UNITS ALREADY IN KTS
sfc_wind_pred[np.isnan(mag_3km)] = np.nan
mag_3km[(rd/hdobs_rmw < 0.3)] = np.nan
#mag_3km[(rd/hdobs_rmw < 0.5)] = np.nan
print('/n')
print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

# grab RMW
swann_rmw = rd[np.unravel_index(np.nanargmax(sfc_wind_pred),np.shape(sfc_wind_pred))]

# convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012
u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th)
v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th)

#%% main code: step 4 - save output data as NetCDF (adapted from MetPy documentation)


# FIGURE OUT WHERE TO PUT TEXT FILE ********************
#f = open(sam_dir+args.STORM+'_'+args.ANALYSISTIME+'_data.txt','w')
#lines = ['samurai FL RMW: '+str(hdobs_rmw)+'\n', 'samurai FL intens: '+str(np.nanmax(wspd_earth))+'\n', 'predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred))]
#f.writelines(lines)
#f.close()

if alt_plane == 1.5:
    sf_frac = 0.8
elif alt_plane == 3.0:
    sf_frac = 0.9

textstr = '\n'.join((
    storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + hdobs_sm.mission.unique()[0],
    leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M'),
    'Inputs: HDOBS',
    '\n',
    r'HDOB FL V$_{max}$: %.1f (kt)' % (np.nanmax(hdobs.wsp), ),
    '\n',
    'SWANN RMW: %.1f (nm)' % (swann_rmw, ),
    r'$\bf{SWANN\ V_{max}:\ %.1f\ (kt)}$' % (np.nanmax(sfc_wind_pred*1.94), ),
    r'SFMR V$_{max}$: %.1f (kt)' % (np.nanmax(hdobs.sfmr), ),
    'Simplfied Franklin: %.1f (kt)' % (sf_frac*np.nanmax(wspd_earth*1.94), ) ))

# convert coords, first to cartesian
#if sam_fn == 'samurai_RTZ_analysis.nc':
#    x_plot = rd*np.cos(np.radians(th))
#    y_plot = rd*np.sin(np.radians(th))
    # lon_nc, lat_nc = latlon(sam_lon, sam_lat, x_plot, x_plot) # Michael's code ####### CONFIRM LAT LON with best center ########
    # geod = Geod(ellps='WGS84')
    # lon_r, lat_r, _ = geod.fwd(storm_lon, storm_lat, theta_met, rd*1000) # check where zero is expected for azimuth

#%% main code: step 5 - generate any images
# x,y instead of lat/lon? 

fig = plt.figure(figsize=(8.5,3.5))
#fig = plt.figure(constrained_layout=True,figsize=(8,6))
gs = fig.add_gridspec(1,3)
f_ax4 = fig.add_subplot(gs[0, :-1])
f_ax5 = fig.add_subplot(gs[0, -1])
f_ax4.plot(hdobs.dt, hdobs.sfmr, 'k',hdobs.dt, hdobs.wsp, 'r',hdobs.dt, sfc_wind_pred*1.94, 'green')
f_ax5.text(0.0, 0.95, textstr, transform=f_ax5.transAxes, fontsize=11,verticalalignment='top')
f_ax5.set_axis_off()
f_ax4.legend(['SFMR (kt)','FL (kt)','SWANN (kt)'])
f_ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.savefig(imDir+args.STORM+'_'+args.ANALYSISTIME+'_af_2pan_'+ml_ver+mode+'.png', dpi=200, bbox_inches='tight')

