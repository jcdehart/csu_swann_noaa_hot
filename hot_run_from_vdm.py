#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 2026

@author: jcdehart
"""

from hot_calc_centers import read_vdm
import argparse
# grab info from tcvitals or flight+ file

parser = argparse.ArgumentParser()
parser.add_argument("path", help="VDM file path", type=str)
args = parser.parse_args()

if (len(args.path) > 0):
    file = args.path
else:
    file = './ingest_dir/center_data/vdm/2025/REPNT2-KWBC.202508181148.txt'

read_vdm(file,'trigger')