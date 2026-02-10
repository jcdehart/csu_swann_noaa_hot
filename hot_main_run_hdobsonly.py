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
#parser.add_argument("CENTIME", help="cen datetime (YYYYMMDDHHMM)", type=str)
#parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("STARTTIME", help="samurai start datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ENDTIME", help="samurai end datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("PLANE", help="plane: NOAA (N) or AF (A)", type=str)
parser.add_argument("--CENFN", default="gfs.tXXz.syndata.tcvitals.tm00", help="TC Vitals filename", type=str)
parser.add_argument("--CENPATH", default="./ingest_dir/center_data", help="TC Vitals directory", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

# ***** change into an argument, or change to start time + duration, model will be settled in future, deal with af only later
#dur = 45
if args.PLANE == 'A':
    af = True
elif args.PLANE == 'N':
    af = False
else:
    print('SPECIFY PLANE!!')

#alt_plane = 1.5
#alt_plane = 3.0
ml_ver = 'FRED'

#%% set up dirs
# local testing
inDir = '/bell-scratch/jcdehart/hot_operational/retrospective_testing/'
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
    hdobs = hot_calc_centers.read_hdobs('KNHC', storm_name_2,'HDOBS', leg_start, leg_end)
elif af == False:
    hdobs = hot_calc_centers.read_hdobs('KWBC', storm_name_2,'HDOBS', leg_start, leg_end)

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

#print('averaging all 3 centers')

# perhaps have different weights based on tcvitals intensity??? ********
wgt = np.array([1, 1, 5])
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
x_plane,y_plane = xy(hdobs.lat.values,hdobs.lon.values,lat_wc,lon_wc)
print('Using W-C center')

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
# expected units of input vars (for this function, not model): km, km, deg (math), deg (math), kts, m/s, m/s, km (m for HDOBs)
X_ratio, r_norm = hot_prep_data.process_nn_vars(rd, hdobs_rmw, th, storm_dir, storm_intens, storm_motion, wspd_earth, alt_plane, af)

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

# convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012, use th_r in radians
u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th_r) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th_r)
v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th_r) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th_r)

#%% main code: step 4 - save output data as NetCDF (adapted from MetPy documentation)

print('\n')
print('########')
print('save txt file, netcdf, image')

# open file
ncfile_sfc = Dataset('./nn_output/HOT_HDOBS_sfc_analysis_'+args.STORM+'_'+samurai_time.strftime('%Y%m%d%H%M')+'.nc',mode='w',format='NETCDF4') 

# define dimensions
time_dim = ncfile_sfc.createDimension('time', len(hdobs.dt)) # unlimited axis (can be appended to)
    
# set up metadata
ncfile_sfc.title='CSU Predicted Surface Wind'
ncfile_sfc.subtitle="Generated using CSU SWANN"

# set up variables
nclat = ncfile_sfc.createVariable('latitude', np.float32, ('time'))
nclat.units = 'degrees_north'
nclat.long_name = 'latitude'
nclon = ncfile_sfc.createVariable('longitude', np.float32, ('time'))
nclon.units = 'degrees_east'
nclon.long_name = 'longitude'
nctime = ncfile_sfc.createVariable('time', np.float64, ('time'))
nctime.units = 'seconds since 1970-01-01'
nctime.long_name = 'time'
# Define a 3D variable to hold the data
ncu = ncfile_sfc.createVariable('u_wind',np.float64,('time')) # note: unlimited dimension is leftmost
ncu.units = 'm s-1' 
ncu.standard_name = 'eastward_wind' # this is a CF standard name
ncu.long_name = 'U component of the predicted surface wind'
ncv = ncfile_sfc.createVariable('v_wind',np.float64,('time')) # note: unlimited dimension is leftmost
ncv.units = 'm s-1' 
ncv.standard_name = 'northward_wind' # this is a CF standard name
ncv.long_name = 'V component of the predicted surface wind'

# save data to arrays 
nclat[:] = hdobs.lat.values
nclon[:] = hdobs.lon.values
nctime[:] = (hdobs.dt  - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
ncu[:] = u_nc
ncv[:] = v_nc
    
ncfile_sfc.close()

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

figtitle = storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + hdobs_sm.mission.unique()[0] + ' | ' + leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M') + ' UTC'

textstr = '\n'.join((
    'Inputs: HDOBS',
    'W-C Center: %.2f N, %.2f W' % (lat_wc,np.abs(lon_wc),), # assuming western hemisphere
    'RMW: %.1f (nm)' % (swann_rmw/1.852,),
    'Simp. Franklin: %.1f (kt)' % (simp_frank,), ))

# save text file
f = open(inDir+'txt_output/'+args.STORM+'_'+analysis_time+'_data_hdobsonly.txt','w')
lines = ['Inputs: HDOBS\n', 'W-C Center: '+str(lat_wc)+', '+str(lon_wc)+'\n', 'HDOBS Vmax (kts): '+str(hdobs_fl_vmax)+'\n', 'SWANN Vmax (kts): '+str(swann_hdobs_vmax), 'SWANN RMW (nm): '+str(swann_rmw/1.852), 'Simplified Franklin (kts): '+str(simp_frank)]
f.writelines(lines)
f.close()

#textstr = '\n'.join((
#    storm_name_2 + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + hdobs_sm.mission.unique()[0],
#    leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M'),
#    'Inputs: HDOBS',
#    '\n',
#    r'HDOB FL V$_{max}$: %.1f (kt)' % (np.nanmax(hdobs.wsp), ),
#    '\n',
#    'SWANN RMW: %.1f (nm)' % (swann_rmw, ),
#    r'$\bf{SWANN\ V_{max}:\ %.1f\ (kt)}$' % (np.nanmax(sfc_wind_pred*1.94), ),
#    r'SFMR V$_{max}$: %.1f (kt)' % (np.nanmax(hdobs.sfmr), ),
#    'Simplified Franklin: %.1f (kt)' % (sf_frac*np.nanmax(wspd_earth*1.94), ) ))

#%% main code: step 5 - generate any images
# x,y instead of lat/lon? 

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

vmax_col_labels = ['HDOBS\nVmax (kt)']
vmax_row_labels = ['FL','SWANN']
vmax_table = [[hdobs_fl_vmax],[swann_hdobs_vmax]]

radii_col_labels = ['NE','SE','SW','NW']
radii_row_labels = ['R34','R50','R64']

fig = plt.figure(figsize=(8.5,3.5))
#fig = plt.figure(constrained_layout=True,figsize=(8,6))
gs = fig.add_gridspec(1,3)
f_ax4 = fig.add_subplot(gs[0, :-1])
f_ax5 = fig.add_subplot(gs[0, -1])
f_ax4.plot(hdobs.dt, hdobs.sfmr, 'k',hdobs.dt, hdobs.wsp, 'r')
f_ax4.plot(hdobs.dt, sfc_wind_pred*1.94, color='#1E4D2B')
f_ax4.plot(hdobs.dt.values[0], hdobs.wsp.values[0], 'kx') # flight start
f_ax4.plot(hdobs.dt.values[-1], hdobs.wsp.values[-1], 'ko') # flight end
axins = f_ax4.inset_axes(
    [0.02, 0.78, 0.15, 0.2], xticklabels=[], yticklabels=[])
axins.plot(x_plane, y_plane,'r')
axins.plot(x_plane[0], y_plane[0],'kx')
axins.plot(x_plane[-1], y_plane[-1],'ko')
axins.plot(0, 0,'k*')
#axins.contour(x_plot, y_plot, radii, levels=np.array([swann_rmw]), colors='r', linestyles='dotted');
#axins.contour(x_plot/1.852, y_plot/1.852, rd/1.852, levels=np.array([swann_rmw/1.852]), colors='r', linestyles='dotted');

f_ax5.text(-0.075, 0.99, textstr, transform=f_ax5.transAxes, fontsize=10,verticalalignment='top')
my_table = f_ax5.table(cellText=np.round(vmax_table,decimals=1),
                     rowLabels=vmax_row_labels,
                     colLabels=vmax_col_labels,
                     bbox=[0.15,0.3,0.4,0.375])
for (row, col), cell in my_table.get_celld().items():
    if (row == 2):
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        cell.get_text().set_color('#1E4D2B')

my_table2 = f_ax5.table(cellText=radii_vals_str, # convert radii from km to nm
#my_table2 = f_ax5.table(cellText=np.rint(radii_vals/1.852).astype(int), # convert radii from km to nm
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
f_ax4.legend(['SFMR','FL','SWANN'], loc='lower right')
f_ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
f_ax4.grid(True)
f_ax4.set_ylabel('wind speed (kt)')
plt.suptitle(figtitle,y=0.94)
fig.savefig(imDir+args.STORM+'_'+samurai_time.strftime(format='%Y%m%d%H%M')+'_af_2pan.png', dpi=200, bbox_inches='tight')

