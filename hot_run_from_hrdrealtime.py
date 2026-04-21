#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 2026

@author: jcdehart
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# parse realtime analyses websites
response = requests.get('https://www.aoml.noaa.gov/ftp/pub/hrd/reasor/realtime_analyses/')
soup = BeautifulSoup(response.text, 'html.parser')

# keep only file names that include "tar"
filelist = [x for x in soup.get_text().split('\n') if 'tar' in x]

# grab latest file (will need to add modes most likely ******)
lastfile = filelist[-1].strip()
pieces = lastfile.split('_')

# get times (should probably make 20 a var...)
yymmdd = '20' + pieces[0][:6]

# convert to pandas timedelta (to deal with 24 hour clock issue)
td1 = pd.Timedelta(hours=int(pieces[1][:2]), minutes=int(pieces[1][2:]))
td2 = pd.Timedelta(hours=int(pieces[2][:2]), minutes=int(pieces[2][2:]))

# create start and end time variables
starttime = pd.to_datetime(yymmdd, format='%Y%m%d') + td1
endtime = pd.to_datetime(yymmdd, format='%Y%m%d') + td2