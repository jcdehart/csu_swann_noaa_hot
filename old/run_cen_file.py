import pandas as pd
import numpy as np
import argparse
from samurai_gen_file import make_cen_file, modify_param_file


#%% main code

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("REFTIME", help="reference UTC datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis UTC datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("OFFSET", help="1/2 length of analysis (min, symmetric)", type=float)
parser.add_argument("LAT", help="reference latitude", type=float)
parser.add_argument("LON", help="reference longitude", type=float)
parser.add_argument("u", help="domain zonal motion (m/s)", type=float)
parser.add_argument("v", help="domain meridional motion (m/s)", type=float)
parser.add_argument("--outdir", default="./", help="output directory", type=str)
args = parser.parse_args()

u = args.u
v = args.v

# check for precision errors
if (np.abs(u) < 1e-3):
    u = np.round(u)
if (np.abs(v) < 1e-3):
    v = np.round(v)

# get datetime objects
center_time = pd.to_datetime(args.REFTIME, format='%Y%m%d%H%M', utc=True)
samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)

# create center file
make_cen_file(center_time, samurai_time, args.OFFSET, args.LAT, args.LON, u, v, args.outdir)
