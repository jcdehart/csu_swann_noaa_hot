#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 2026

@author: jcdehart
"""

from hot_calc_centers import read_vdm
import argparse
import save_files
import pandas as pd
# grab info from tcvitals or flight+ file

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="VDM file path", type=str)
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

# check if files were created already
out = save_files.check_files(outDir, imDir, storm_id[:4], analysis_time, 'HDOBS')

if out == True:
    print('files: Y')
    print(storm_id[:4])
    print(leg_start.strftime('%Y%m%d%H%M'))
    print(leg_end.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)
elif out == False:
    print('files: N')
    print(storm_id[:4])
    print(leg_start.strftime('%Y%m%d%H%M'))
    print(leg_end.strftime('%Y%m%d%H%M'))
    print(lat)
    print(lon)