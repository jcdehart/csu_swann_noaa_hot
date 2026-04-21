import pandas as pd
import numpy as np
import os
import argparse
import requests

# grab date from arguments
parser = argparse.ArgumentParser()
parser.add_argument("INITDATE", help="initial date (YYYYMMDDHH)", type=str)
args = parser.parse_args()

# URL to TC Vitals file
base_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.XXXXXXXX/HH/atmos/'
base_fn = 'gfs.tHHz.syndata.tcvitals.tm00'

# create datetime object and specific date/time strings
tcvitals_time = pd.to_datetime(args.INITDATE, format='%Y%m%d%H', utc=True)
datetime = tcvitals_time.strftime('%Y%m%d')
hour = tcvitals_time.strftime('%H')

# create new url and filename for user-specified period
current_url = base_url.replace('XXXXXXXX',datetime).replace('HH',hour)
current_fn = base_fn.replace('HH',hour)

# check if file exists already or download file
final_path = './center_data/tcvitals/'+datetime+'/'
os.system('mkdir -p '+final_path)

if os.path.isfile(final_path + current_fn):
    print('file already exists: '+final_path+current_fn)
else:
    resp = requests.get(current_url + current_fn, stream=True)
    print(resp.status_code)

    if (resp.status_code == 200):

        print('downloading file: '+current_url+current_fn)

        with open(final_path + current_fn, "wb") as f: # opening a file handler to create new file 
            f.write(resp.content)

    # if a 404 code is returned, grab the last time stamp
    elif (resp.status_code == 404):
    #if (os.stat('./' + current_fn).st_size == 0):
        
        print('404 code returned, trying 6 hours before')

        td = pd.Timedelta(hours=6)
        tcvitals_time_prev = tcvitals_time - td
        datetime_prev = tcvitals_time_prev.strftime('%Y%m%d')
        hour_prev = tcvitals_time_prev.strftime('%H')
        prev_url =  base_url.replace('XXXXXXXX',datetime_prev).replace('HH',hour_prev)
        prev_fn = base_fn.replace('HH',hour_prev)
        
        # set up prev directory
        prev_path = './center_data/tcvitals/'+datetime_prev+'/'
        os.system('mkdir -p '+prev_path)
    
        if os.path.isfile(prev_path + prev_fn):
            print('file already exists: '+prev_path+prev_fn)
        else:
            resp_prev = requests.get(prev_url + prev_fn)
            print(resp_prev.status_code)

            # download file if it exists
            if (resp_prev.status_code == 200):
                print('downloading file: '+prev_url+prev_fn)

                with open(prev_path + prev_fn, "wb") as f: # opening a file handler to create new file 
                    f.write(resp_prev.content)

            else:
                print(resp_prev.status_code)
                print('some other error occurring - investigate')

    else:
        print(resp.status_code)
        print('some other error occurring - investigate')
