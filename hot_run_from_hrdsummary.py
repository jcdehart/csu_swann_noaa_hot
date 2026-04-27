#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 2026

@author: jcdehart
"""

from hot_calc_centers import read_hrdsumm
import argparse
import pandas as pd
import os
import save_files

parser = argparse.ArgumentParser()
parser.add_argument("path", help="HRD summary file path", type=str)
parser.add_argument("--MODE", default="normal", help="run mode (test or normal)", type=str)
args = parser.parse_args()

mode = args.MODE

if mode == 'test':
    ext = 'testing/output/'
else:
    ext = ''

inDir = './'
outDir = inDir+ext
imDir = outDir+'images/'

if (len(args.path) > 0):
    file = args.path
else:
    file = './ingest_dir/center_data/hrd_tar/2025/251028H1_1328_1403_analysis.tar'

# grab latest file (will need to add modes most likely ******)
filename = file.strip().split('/')[-1]
pieces = filename.split('_')

# get times (should probably make 20 a var...)
yymmdd = '20' + pieces[0][:6]

# convert to pandas timedelta (to deal with 24 hour clock issue)
td1 = pd.Timedelta(hours=int(pieces[1][:2]), minutes=int(pieces[1][2:]))
td2 = pd.Timedelta(hours=int(pieces[2][:2]), minutes=int(pieces[2][2:]))

# create start and end time variables
starttime = pd.to_datetime(yymmdd, format='%Y%m%d') + td1
endtime = pd.to_datetime(yymmdd, format='%Y%m%d') + td2

analysis_time = (starttime + ((endtime-starttime)/2).round('min')).strftime('%Y%m%d%H%M')

# check if files were created already
out = save_files.check_files(outDir, imDir, args, analysis_time, 'SAM')

if out == True:
    print('file already processed.')
elif out == False:
    print('files do not exist, proceeding.')
    os.system('tar -xf '+file)
    flight_id, storm_id, mission_id, storm_name, lat, lon = read_hrdsumm(file[:-4]+'/summary')
    print(storm_id[:4])
    print(starttime.strftime('%Y%m%d%H%M'))
    print(endtime.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)