import glob
import pandas as pd
import numpy as np
import os
import argparse
from netCDF4 import Dataset
from samurai_gen_file import make_cen_file, modify_param_file


#%% main code

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("CENPATH", help="TC Vitals directory", type=str)
parser.add_argument("CENFN", help="TC Vitals filename", type=str)
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("CENTIME", help="cen datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals, fplus, samurai)", type=str)
parser.add_argument("--SAM_LAT", default=[], help="center lat from samurai", type=float)
parser.add_argument("--SAM_LON", default=[], help="center lon from samurai", type=float)
parser.add_argument("--U", default=[], help="storm u", type=float)
parser.add_argument("--V", default=[], help="storm v", type=float)
args = parser.parse_args()

if args.CENTYPE == "fplus":

    print('using flight+ data for center info')

    fplus_path = args.CENPATH+'/fplus/'
    fplus_fn = args.CENFN 
    
    fplus = Dataset(fplus_path+fplus_fn)
    fplus_centime = fplus['FL_WC_wind_center_time_offset']
    fplus_cenlat = fplus['FL_WC_wind_center_time_latitude']
    fplus_cenlon = fplus['FL_WC_wind_center_time_longitude']

    # FIX CENTER TIME ***********
    # fplus_centime normally is an array with the center by flight...
    # unsure how that will work in realtime, but will need to select the correct center... *********
    center_time = pd.to_datetime(fplus_centime, unit='s', utc=True)
    samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)
    tcvitals_time = samurai_time.floor('6h')
    
    # grab time from flight+ file?
    tcvitals_path = args.CENPATH+'/tcvitals/'+args.CENTIME[:8]+'/'
    tcvitals_fn = args.CENFN # ****** FIX FILENAME OR GET IT FROM IT????? *******

    # grab all lines that contain storm
    file = open(tcvitals_path+tcvitals_fn)
    searchwds = [args.STORM, tcvitals_time.strftime('%Y%m%d'), tcvitals_time.strftime('%H%M')]
    for line in file: 
        if all(word in line for word in searchwds):
            tc_vital.append(line)
    
    # grab metrics from flight+ file
    storm_dir = float(tc_vital[0][44:47])
    storm_motion = float(tc_vital[0][48:51])/10.

    make_cen_file(center_time, samurai_time, 1, fplus_cenlat, fplus_cenlon, u, v, './testing/')

elif args.CENTYPE == "tcvitals":

    print('using tcvitals data for center info')

    tc_vital = []
    tcvitals_path = args.CENPATH+'/tcvitals/'+args.CENTIME[:8]+'/'
    tcvitals_fn = args.CENFN 
    #tcvitals_path = '/bell-scratch/jcdehart/hot/JHT_Michael_Test/TC_Vitals/'
    #tcvitals_fn = 'syndat_tcvitals.2018'i

    # grab all lines that contain storm
    file = open(tcvitals_path+tcvitals_fn)
    searchwds = [args.STORM, args.CENTIME[0:8], args.CENTIME[8:12]]
    #searchwds = ['MICHAEL','20181010','1200']
    for line in file: 
        if all(word in line for word in searchwds):
            tc_vital.append(line)

    # grab variables from TC Vitals file, character numbers taken from EMC website
    # https://www.emc.ncep.noaa.gov/mmb/data_processing/tcvitals_description.htm
    storm_lat_r = float(tc_vital[0][33:36])/10.
    storm_lat_hemi = tc_vital[0][36]
    storm_lon_r = float(tc_vital[0][38:42])/10.
    storm_lon_hemi = tc_vital[0][42]
    storm_dir = float(tc_vital[0][44:47])
    storm_motion = float(tc_vital[0][48:51])/10.

    # convert W, S to -
    if storm_lat_hemi == 'S':
        storm_lat = -1*storm_lat_r
    else:
        storm_lat = storm_lat_r

    if storm_lon_hemi == 'W':
        storm_lon = -1*storm_lon_r
    else:
        storm_lon = storm_lon_r

    # convert storm motion from met degrees to standard math degrees
    storm_dir_rot = 90 + (360 - storm_dir)
    if storm_dir_rot > 360:
        storm_dir_rot = storm_dir_rot - 360

    # calculate u and v components
    u = storm_motion*np.cos(np.radians(storm_dir_rot))
    v = storm_motion*np.sin(np.radians(storm_dir_rot))

    # check for precision errors
    if (np.abs(u) < 1e-3):
        u = np.round(u)
    if (np.abs(v) < 1e-3):
        v = np.round(v)

    #print([storm_lat, storm_lon, storm_dir, storm_motion, storm_dir_rot, u, v])

    # get datetime object from TC Vitals time
    center_time = pd.to_datetime(searchwds[1]+searchwds[2], format='%Y%m%d%H%M', utc=True)
    samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)

    # create center file
    #make_cen_file(center_time, samurai_time, 110, storm_lat, storm_lon, u, v, './testing/')
    make_cen_file(center_time, samurai_time, 45, storm_lat, storm_lon, u, v, './testing/')

elif args.CENTYPE == "samurai":

    print('using samurai/prior data for center info')

    # get datetime object from TC Vitals time
    center_time = pd.to_datetime(searchwds[1]+searchwds[2], format='%Y%m%d%H%M', utc=True)
    samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)

    # create center file
    #make_cen_file(center_time, samurai_time, 110, storm_lat, storm_lon, u, v, './testing/')
    make_cen_file(center_time, samurai_time, 45, args.SAM_LAT, args.SAM_LON, args.U, args.V, './testing/')

else:
    print('error: check center type name')

