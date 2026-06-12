#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 2026

@author: jcdehart
"""

from hot_calc_centers import read_hrdsumm
import hot_grab_files
import argparse
import pandas as pd
import os
import save_files
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="HRD summary file path", type=str)
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
    file = './ingest_dir/hrd_tar/2026/260421I1_1514_1520_analysis.tar'
    # file = './ingest_dir/hrd_tar/2025/251028H1_1328_1403_analysis.tar'

# grab latest file (will need to add modes most likely ******)
filename = file.strip().split('/')[-1]
pieces = filename.split('_')

# get times (should probably make 20 a var...)
yymmdd = '20' + pieces[0][:6]

# convert to pandas timedelta (to deal with 24 hour clock issue)
td1 = pd.Timedelta(hours=int(pieces[1][:2]), minutes=int(pieces[1][2:]))
td2 = pd.Timedelta(hours=int(pieces[2][:2]), minutes=int(pieces[2][2:]))

# create start and end time variables
leg_start = pd.to_datetime(yymmdd, format='%Y%m%d') + td1
leg_end = pd.to_datetime(yymmdd, format='%Y%m%d') + td2

analysis_time = (leg_start + ((leg_end-leg_start)/2).round('min')).strftime('%Y%m%d%H%M')

# create directories and untar hrd files
os.system('mkdir -p ./samurai_parent/hrd_output')
os.system('tar -xf '+file+' -C ./samurai_parent/hrd_output/')
os.system('rm -f ./samurai_parent/hrd_output/*.gz ./samurai_parent/hrd_output/*files* ./samurai_parent/hrd_output/parameters* ./samurai_parent/hrd_output/run')

# HRD flight code (20251028H1), storm code (AL132025)
# mission code (2313A), storm name (MELISSA)
flight_id, storm_id, mission_id, storm_name, lat, lon = read_hrdsumm('./samurai_parent/hrd_output/summary')

# check if enough data files have arrived
hrd_init = hot_grab_files.create_dataframe(data_dir+'hrd_radials',leg_start,leg_end)
hrd_sm = hot_grab_files.shrink_df(hrd_init, leg_start, leg_end, storm_name, False)
data_start_diff, data_end_diff = hot_grab_files.check_dates(hrd_sm, leg_start, leg_end)

# check if files were created already
out = save_files.check_files(outDir, imDir, storm_id[:4], analysis_time, 'SAM')

# check if flight is actually a hurricane flight
# no invests, PTCs, training, or winter storm flights
flight_ignore = np.isin(storm_name, ['INVEST', 'TRAIN', 'CYCLONE']) # True if storm name matches these
ptc = storm_name.startswith('PTC') # True if storm name starts with PTC 
stormnum = storm_id[2:4].isalpha() # True if 3rd/4th characters in storm id are letters

if flight_ignore | ptc | stormnum:
    tc_check = 'other'
else:
    tc_check = 'TC'

if out == True:
    print('file summary:')
    print(data_end_diff)
    print('Y')
    print(tc_check)
    print(storm_id[:4])
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
    print(leg_start.strftime('%Y%m%d%H%M'))
    print(leg_end.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)