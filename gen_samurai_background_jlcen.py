#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:30:05 2023

@author: jcdehart
"""

import numpy as np
import pandas as pd
#import os
import argparse
import matplotlib.pyplot as plt
from hot_calc_centers import motion_calcs

plt.rcParams.update({'mathtext.default':  'regular' })

imDir = '/bell-scratch/jcdehart/hot/testing/background_images/'

#%% functions and dunion sounding

# angMOM function
def angMOM(r, v, f):
    
    M = r*v + f*(r**2)/2
    
    return(M)

def latlon(cenlon, cenlat, dom_x, dom_y):
    latrad = np.radians(cenlat)

    # do math
    fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
    fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
    dom_lon = cenlon + (dom_x)/fac_lon # distance in km
    dom_lat = cenlat + (dom_y)/fac_lat
    
    return(dom_lon, dom_lat)

# dunion sounding
# https://raw.githubusercontent.com/mmbell/samurai/71794e28a68efa7c2c84201a65bd0a2329a76138/util/dunion_mt_sounding.txt
z = np.array([0, 124.00, 810.00, 1541.0, 3178.0, 4437.0, 5887.0, 7596.0])
T = np.array([26.800, 26.500, 21.900, 17.600, 8.9000, 1.6000, -6.6000, -17.100])
qv = np.array([18.650, 18.500, 15.270, 11.960, 6.7400, 4.1100, 2.4100, 1.1000])
rhoa = np.array([1.1435, 1.1282, 1.0655, 0.99905, 0.85538, 0.75588, 0.65106, 0.54336])

#%% 

# grab info from tc vitals file
parser = argparse.ArgumentParser()
parser.add_argument("VITALSPATH", help="TC Vitals directory", type=str)
parser.add_argument("STORM", help="storm name", type=str)
parser.add_argument("DATETIME", help="tcvitals datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("LAT", help="storm center lat", type=str)
parser.add_argument("LON", help="storm center lon", type=str)
args = parser.parse_args()

cyl = True

tc_vital = []
tcvitals_path = args.VITALSPATH + args.DATETIME[:8] + '/'
tcvitals_fn = 'gfs.tXXz.syndata.tcvitals.tm00'.replace('XX',args.DATETIME[8:10])

# grab all lines that contain storm
file = open(tcvitals_path+tcvitals_fn)
searchwds = [args.STORM, args.DATETIME[:8], args.DATETIME[8:]]
for line in file: 
    if all(word in line for word in searchwds):
        tc_vital.append(line)
        
#%% extract info 

storm_lat = float(args.LAT)
storm_lon = float(args.LON)
storm_intens_full = float(tc_vital[0][67:69]) # m/s
storm_rmw = 1000*float(tc_vital[0][70:73]) # convert to m
storm_r34 = 1000*np.array([float(tc_vital[0][74:78]), float(tc_vital[0][79:83]), float(tc_vital[0][84:88]), float(tc_vital[0][89:93])]) # convert to m
storm_dir = float(tc_vital[0][44:47])
storm_motion = float(tc_vital[0][48:51])/10.
storm_intens = storm_intens_full - storm_motion # remove motion wave-1 component from intensity

u_motion, v_motion, storm_dir_rot = motion_calcs(storm_dir, storm_motion)

# turn -999 to np.nan
storm_r34[storm_r34 == -999000] = np.nan
storm_r34_avg = np.nanmean(storm_r34)

#%% create array

horiz_dim = np.arange(-175,176,1)
horiz_len = len(horiz_dim)

# box that will actually be created
x, y = np.meshgrid(horiz_dim, horiz_dim, indexing='xy')
surface_r = np.sqrt(x**2 + y**2)
surface_wind = np.zeros(np.shape(surface_r))
surface_wind[:] = np.nan

# linearly increasing surface wind
surface_wind[surface_r*1000. <= storm_rmw] = (storm_intens/storm_rmw)*surface_r[surface_r*1000. <= storm_rmw]*1000.

# calc angMOM at rmw, r34, and the slope between those points (dandan, jon's papers)
f = 2*(7.2921e-5)*np.sin(np.radians(storm_lat))
m_max = angMOM(storm_rmw, storm_intens, f)
m_34 = angMOM(storm_r34_avg, 17.4911, f) # converted 34 knots to m/s
m_sl = (-1 + m_34/m_max)/(-1 + storm_r34_avg/storm_rmw)

# calc angMOM for full domain and surface wind outside of rmw
m = m_max*(m_sl*(-1 + surface_r*1000./storm_rmw) + 1)
inner_wind = (surface_r*1000. <= storm_rmw)
outer_wind = (surface_r*1000. > storm_rmw)
surface_wind[outer_wind] = m[outer_wind]/(surface_r[outer_wind]*1000.) + f*surface_r[outer_wind]*1000./2
m[inner_wind] = angMOM(surface_r[inner_wind]*1000., surface_wind[inner_wind], f)

#%% create 3-D wind array

# initialize 3-D array and include surface wind
wind_3d_storm = np.zeros((11, horiz_len, horiz_len))
wind_3d_storm[:] = np.nan
wind_3d_storm[0,:,:] = surface_wind

# back out 3-km wind (1/0.9 < 2RMW, 1/0.8 > 2RMW)
wind_3d_storm[6,:,:] = surface_wind/0.9
# wind_3d[6,:,:][inside_2rmw] = surface_wind[inside_2rmw]/0.9
# wind_3d[6,:,:][outside_2rmw] = surface_wind[outside_2rmw]/0.8

# back out 0.5-km wind (1.1 * 3-km wind)
wind_3d_storm[1,:,:] = 1.1*wind_3d_storm[6,:,:]

# calculate slope for remaining levels
slope_aloft = (wind_3d_storm[6,:,:] - wind_3d_storm[1,:,:])/2.5

# fill in remaining levels
levs = np.array([2,3,4,5,7,8,9,10])
wind_3d_storm[levs,:,:] = (0.5*np.tile(levs[:,None,None],(1,horiz_len,horiz_len)) - 0.5)*slope_aloft + np.tile(wind_3d_storm[1,:,:],(len(levs),1,1))

# create u and v arrays
u_3d_storm = -1*wind_3d_storm*np.sin(np.arctan2(y, x))
v_3d_storm = wind_3d_storm*np.cos(np.arctan2(y, x))
w_3d = np.zeros(np.shape(u_3d_storm))

# add storm motion back in to arrays *******
u_3d_tmp = u_3d_storm + u_motion
v_3d_tmp = v_3d_storm + v_motion

if cyl == True:
    print('converting cartesian u/v to cylindrical')
    u_3d = (u_3d_tmp*x + v_3d_tmp*y)/surface_r
    v_3d = (-u_3d_tmp*y + v_3d_tmp*x) /surface_r
else:
    print('keeping cartesian u/v')
    u_3d = u_3d_tmp
    v_3d = v_3d_tmp

wind_3d = np.sqrt(u_3d**2 + v_3d**2)

#%% create 3-D thermo arrays from dunion sounding

T_K_levs = np.interp(np.arange(0,5.5,0.5),z/1000.,T) + 273.15
qv_levs = np.interp(np.arange(0,5.5,0.5),z/1000.,qv)
rhoa_levs = np.interp(np.arange(0,5.5,0.5),z/1000.,rhoa)

T_K_3d = np.tile(T_K_levs[:, None, None], (1, horiz_len, horiz_len))
qv_3d = np.tile(qv_levs[:, None, None], (1, horiz_len, horiz_len))
rhoa_3d = np.tile(rhoa_levs[:, None, None], (1, horiz_len, horiz_len))

#%% time and location arrays

# calculate time in unix seconds
vitals_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True).strftime('%Y-%m-%d_%H:%M:%S')

# 3-d arrays of lat, lon
surface_lon, surface_lat = latlon(storm_lon, storm_lat, x, y)
lon_3d = np.tile(surface_lon, (11,1,1))
lat_3d = np.tile(surface_lat, (11,1,1))

# altitude
alt_3d = np.tile(np.arange(0,5500,500)[:,None,None], (1, horiz_len, horiz_len))

#%% create master array for writing to file

# need to move axes for lat, lon, alt, u, v, w, T, qv, rhoa to conform to samurai standards (columns)
# time, qr, terr_hgt don't matter because they're single numbers (FOR NOW)
# removing data where u or v are nan (origin of cylindrical coordinates)

len1d = len(lon_3d.ravel())
good = np.moveaxis(~np.isnan(u_3d), 0, 2).ravel()
d = {'time': np.tile(vitals_time, (len1d))[good], 'lat': np.moveaxis(lat_3d, 0, 2).ravel()[good], 'lon': np.moveaxis(lon_3d, 0, 2).ravel()[good], 
     'alt': np.moveaxis(alt_3d, 0, 2).ravel()[good], 'u': np.moveaxis(u_3d, 0, 2).ravel()[good], 'v': np.moveaxis(v_3d, 0, 2).ravel()[good], 
     'w': np.moveaxis(w_3d, 0, 2).ravel()[good], 'T': np.moveaxis(T_K_3d, 0, 2).ravel()[good], 'qv': np.moveaxis(qv_3d, 0, 2).ravel()[good], 
     'rhoa': np.moveaxis(rhoa_3d, 0, 2).ravel()[good], 'qr': np.tile(0.0, (len1d))[good], 'terr_hgt': np.tile(0.0, (len1d))[good]}

background_file = pd.DataFrame(d)

background_file.to_csv('samurai_Background.in', header=None, index=None, sep=' ', float_format='%.4f')
# , fmt=['%s','%f','%f','%.1f','%f','%f','%.2f','%.2f','%.2f','%.3f',]

#%% plots

cmax = np.ceil(np.nanmax(wind_3d)/5)*5 + 2.5
umax = np.ceil(np.nanmax(u_3d)/5)*5 + 2.5
vmax = np.ceil(np.nanmax(v_3d)/5)*5 + 2.5

fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(8,4), dpi=200)
c1 = axs[0].contourf(x, y, wind_3d[0,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c2 = axs[1].contourf(x, y, m/m_max, cmap='YlOrBr')
cb1 = plt.colorbar(c1,ax=axs[0])
cb2 = plt.colorbar(c2,ax=axs[1])
cb1.ax.set_title(r'm s$^{-1}$')
cb2.ax.set_title(r'kg m$^2$ s$^{-1}$')
axs[0].set_xlabel('distance (km)')
axs[0].set_ylabel('distance (km)')
axs[1].set_xlabel('distance (km)')
axs[0].set_title('surface wind speed')
axs[1].set_title('surface ang momentum')
plt.savefig(imDir+'sfc_wind_angmom.png', dpi=200)

fig, axs = plt.subplots(1,2, sharex=True, figsize=(8,4), dpi=200)
axs[0].plot(horiz_dim,wind_3d[0,176,:])
axs[1].plot(horiz_dim,m[176,:]/m_max)
axs[0].set_xlabel('distance (km)')
axs[0].set_ylabel('wind speed (m/s)')
axs[1].set_xlabel('distance (km)')
axs[1].set_ylabel('angular momentum')
plt.savefig(imDir+'cen_wind_angmom.png', dpi=200)

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(6,6), dpi=200)
c1 = axs[0][0].contourf(x, y, wind_3d[0,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c2 = axs[0][1].contourf(x, y, wind_3d[1,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c3 = axs[1][0].contourf(x, y, wind_3d[6,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c4 = axs[1][1].contourf(x, y, wind_3d[10,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
axs[0][0].plot(x[204,204],y[204,204],'ko')
axs[0][0].plot(x[224,224],y[224,224],'ko')
cb1 = plt.colorbar(c1,ax=axs[0][0])
cb2 = plt.colorbar(c2,ax=axs[0][1])
cb3 = plt.colorbar(c1,ax=axs[1][0])
cb4 = plt.colorbar(c2,ax=axs[1][1])
cb1.ax.set_title(r'm s$^{-1}$')
cb2.ax.set_title(r'm s$^{-1}$')
cb3.ax.set_title(r'm s$^{-1}$')
cb4.ax.set_title(r'm s$^{-1}$')
axs[1][0].set_xlabel('distance (km)')
axs[0][0].set_ylabel('distance (km)')
axs[1][0].set_ylabel('distance (km)')
axs[1][1].set_xlabel('distance (km)')
axs[0][0].set_title('surface')
axs[0][1].set_title('0.5 km')
axs[1][0].set_title('3 km')
axs[1][1].set_title('5 km')
plt.savefig(imDir+'wind_levs.png', dpi=200)

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(6,6), dpi=200)
c1 = axs[0][0].contourf(surface_lon, surface_lat, wind_3d[0,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c2 = axs[0][1].contourf(surface_lon, surface_lat, wind_3d[1,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c3 = axs[1][0].contourf(surface_lon, surface_lat, wind_3d[6,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
c4 = axs[1][1].contourf(surface_lon, surface_lat, wind_3d[10,:,:], levels=np.arange(0,cmax,2.5), cmap='Spectral_r')
cb1 = plt.colorbar(c1,ax=axs[0][0])
cb2 = plt.colorbar(c2,ax=axs[0][1])
cb3 = plt.colorbar(c1,ax=axs[1][0])
cb4 = plt.colorbar(c2,ax=axs[1][1])
cb1.ax.set_title(r'm s$^{-1}$')
cb2.ax.set_title(r'm s$^{-1}$')
cb3.ax.set_title(r'm s$^{-1}$')
cb4.ax.set_title(r'm s$^{-1}$')
axs[0][0].set_title('surface')
axs[0][1].set_title('0.5 km')
axs[1][0].set_title('3 km')
axs[1][1].set_title('5 km')
plt.savefig(imDir+'wind_levs_latlon.png', dpi=200)

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(6,6), dpi=200)
c1 = axs[0][0].contourf(x, y, u_3d[0,:,:], levels=np.arange(-1*umax + 2.5,umax,2.5), cmap='Spectral_r')
c2 = axs[0][1].contourf(x, y, v_3d[0,:,:], levels=np.arange(-1*vmax + 2.5,vmax,2.5), cmap='Spectral_r')
c3 = axs[1][0].contourf(x, y, u_3d[6,:,:], levels=np.arange(-1*umax + 2.5,umax,2.5), cmap='Spectral_r')
c4 = axs[1][1].contourf(x, y, v_3d[6,:,:], levels=np.arange(-1*vmax + 2.5,vmax,2.5), cmap='Spectral_r')
cb1 = plt.colorbar(c1,ax=axs[0][0])
cb2 = plt.colorbar(c2,ax=axs[0][1])
cb3 = plt.colorbar(c1,ax=axs[1][0])
cb4 = plt.colorbar(c2,ax=axs[1][1])
cb1.ax.set_title(r'm s$^{-1}$')
cb2.ax.set_title(r'm s$^{-1}$')
cb3.ax.set_title(r'm s$^{-1}$')
cb4.ax.set_title(r'm s$^{-1}$')
axs[1][0].set_xlabel('distance (km)')
axs[0][0].set_ylabel('distance (km)')
axs[1][0].set_ylabel('distance (km)')
plt.savefig(imDir+'uv_levs.png', dpi=200)

fig, axs = plt.subplots(1,2, sharex=True, figsize=(8,4), dpi=200)
axs[0].plot(wind_3d[:, 204, 204], alt_3d[:, 204, 204]/1000.)
axs[1].plot(wind_3d[:, 224, 224], alt_3d[:, 224, 224]/1000.)
axs[0].set_xlabel('wind speed (m/s)')
axs[0].set_ylabel('altitude (km)')
axs[1].set_xlabel('wind speed (m/s)')
axs[1].set_ylabel('altitude (km)')
plt.savefig(imDir+'wind_profile.png', dpi=200)
