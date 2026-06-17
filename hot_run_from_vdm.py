#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 2026

@author: jcdehart
"""

from hot_calc_centers import read_vdm
import hot_grab_files
import argparse
import save_files
import pandas as pd
import numpy as np
# grab info from tcvitals or flight+ file

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="VDM file path", type=str)
parser.add_argument("--MODE", default="normal", help="run mode (test or normal)", type=str)
args = parser.parse_args()

inDir = './'

mode = args.MODE

if mode == 'test':
    ext = 'testing/output/'
    data_dir = inDir+'testing/data/'
else:
    ext = ''
    data_dir = inDir+'ingest_dir/'

outDir = inDir+ext
imDir = outDir+'images/'

if args.path is not None:
    file = args.path
else:
    file = './ingest_dir/center_data/vdm/2025/REPNT2-KWBC.202508181148.txt'

# storm code (AL132025), mission code (2313A), storm name (MELISSA)
vdm_center_time, storm_id, mission_id, storm_name, lat, lon = read_vdm(file)

# assuming 45 minute buffer for now
leg_start = vdm_center_time - pd.Timedelta(45,unit='m')
leg_end = vdm_center_time + pd.Timedelta(45,unit='m')

analysis_time = (leg_start + ((leg_end-leg_start)/2).round('min')).strftime('%Y%m%d%H%M')

# check if enough data files have arrived
hdobs_init = hot_grab_files.create_dataframe(data_dir+'hdobs',leg_start,leg_end)
hdobs_sm = hot_grab_files.shrink_df(hdobs_init, leg_start, leg_end, storm_name, True)
data_start_diff, data_end_diff = hot_grab_files.check_dates(hdobs_sm, leg_start, leg_end)

# check if files were created already
out = save_files.check_files(outDir, imDir, storm_id[:4], analysis_time, 'HDOBS')

# check if flight is actually a hurricane flight
# no invests, PTCs, training, or winter storm flights
flight_ignore = np.isin(storm_name, ['INVEST', 'TRAIN', 'CYCLONE']) # True if storm name matches these
ptc = storm_name.startswith('PTC') # True if storm name starts with PTC 
stormnum = storm_id[2:4].isalpha() # True if 3rd/4th characters in storm id are letters

if flight_ignore | ptc | stormnum:
    tc_check = 'other'
    print(storm_name)
    print('PTC: '+str(ptc))
else:
    tc_check = 'TC'

if out == True:
    print('file summary:')
    print(data_end_diff)
    print('Y')
    print(tc_check)
    print(storm_id[:4])
    print(storm_name)
    print(leg_start.strftime('%Y%m%d%H%M'))
    print(leg_end.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)
elif out == False:
    print('file summary:')
    print(data_end_diff)
    print('N')
    print(tc_check)
    print(storm_id[:4])
    print(storm_name)
    print(leg_start.strftime('%Y%m%d%H%M'))
    print(leg_end.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)