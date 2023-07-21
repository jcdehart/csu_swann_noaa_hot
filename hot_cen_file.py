import glob
import pandas as pd
import numpy as np
import os
import argparse
from netCDF4 import Dataset


def make_cen_file_fplus(analysis_time, tol, total_s, times, lat, lon, u, v, outDir):

    """
    Creates a center file for samurai from flight+ data, rewriting Michael's perl script into python

    Parameters
    ----------
    analysis_time : pandas datetime value
        Datetime object defining the samurai analysis time
    tol : int
        Time in minutes defining the start before the samurai analysis time
    total_s : int
        Duration of the samurai analysis, in seconds
    lat : float
        Reference latitude
    lon : float
        Reference longitude
    outDir: str
        Path where center file will be written
    """
    
    # create dataframe from array of times, then reindex to every second
    cen_time_full = pd.DataFrame(pd.to_datetime(times, unit='s', utc=True), columns=['cen_time'], index=times-times[0])
    cen_time_full['lat'] = pd.Series(lat)
    cen_time_full['lon'] = pd.Series(lon)
    cen_time_full = cen_time_full.reindex(np.arange(times[0],times[-1]+1))

    # extract part of dataframe that is desired
    start_time = analysis_time - pd.to_timedelta(tol, unit='m')
    end_time = analysis_time + pd.to_timedelta(tol, unit='m')

    cen_time = cen_time_full[(cen_time_full.cen_time >= start_time) & (cen_time_full.cen_time <= end_time)]
    cen_time = cen_time.interpolate(method = 'linear', inplace = True)
    cen_time['u'] = u
    cen_time['v'] = v
    
    # Samurai struggles with 24 hour clock, for times after midnight, add integers so that they're greater than 23
    # right now doesn't account for full range of possibilities, UPDATE ********
    if (cen_time_sm.cen_time.iloc[0][0:2] == '23') & (cen_time_sm.cen_time.iloc[-1][0:2] == '00'):
        cen_time_sm['cen_time'] = cen_time_sm.cen_time.replace(r"^00",r"24",regex=True)

    cen_file_name = start_time.strftime('%Y%m%d') + '.cen'
    cen_time_sm.to_csv(outDir+cen_file_name, sep=' ', header=False, index=False)


def make_cen_file_tcvitals(ref_time, analysis_time, tol, total_s, lat, lon, u, v, outDir):

    """
    Creates a center file for samurai from a tcvitals file, rewriting Michael's perl script into python

    Parameters
    ----------
    ref_time : pandas datetime value
        Datetime object defining the time when the domain reaches the reference lat/lon 
    analysis_time : pandas datetime value
        Datetime object defining the samurai analysis time
    tol : int
        Time in minutes defining the start before the samurai analysis time
    total_s : int
        Duration of the samurai analysis, in seconds
    lat : float
        Reference latitude
    lon : float
        Reference longitude
    u : float
        Zonal velocity (m/s)
    v: float
        Meridional velocity
    outDir: str
        Path where center file will be written
    """

    # comment: right now, assuming center time is from tcvitals file, which is likely quite a bit earlier than the analysis time

    init_time = ref_time
    start_time = analysis_time - pd.to_timedelta(tol, unit='m')
    end_time = analysis_time + pd.to_timedelta(tol, unit='m')
    # init_time = ref_time - pd.to_timedelta(tol, unit='m')
    cen_time = pd.DataFrame(pd.date_range(start=init_time, end=end_time, freq='S').strftime('%H%M%S'), columns=['cen_time'])
    #cen_time = pd.DataFrame(pd.date_range(start=init_time, end=end_time, periods=total_s, freq='S').strftime('%H%M%S'), columns=['cen_time'])
    cen_time['u'] = u
    cen_time['v'] = v

    if (u == 0.) & (v == 0.):
        cen_time['lat'] = lat
        cen_time['lon'] = lon
    else:
        # set up vars and arrays
        latrad = np.radians(lat)
        delta_sec = (pd.date_range(start=init_time, end=end_time, freq='S') - ref_time).total_seconds()
        #delta_sec = (pd.date_range(start=init_time, periods=total_s, freq='S') - ref_time).total_seconds()
        print(delta_sec)

        # do math
        fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
        fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
        xc = (u*delta_sec)/1000.
        yc = (v*delta_sec)/1000.
        cen_time['lon'] = lon + xc/fac_lon
        cen_time['lat'] = lat + yc/fac_lat

    # select only dates within tolerate of analysis time
    mask = delta_sec >= (start_time - ref_time).total_seconds()
    cen_time_sm = cen_time.loc[mask]
    print(cen_time)
    print(cen_time_sm)

    # Samurai struggles with 24 hour clock, for times after midnight, add integers so that they're greater than 23
    # right now doesn't account for full range of possibilities, UPDATE ********
    if (cen_time_sm.cen_time.iloc[0][0:2] == '23') & (cen_time_sm.cen_time.iloc[-1][0:2] == '00'):
        cen_time_sm['cen_time'] = cen_time_sm.cen_time.replace(r"^00",r"24",regex=True)

    cen_file_name = start_time.strftime('%Y%m%d') + '.cen'
    cen_time_sm.to_csv(outDir+cen_file_name, sep=' ', header=False, index=False)


def modify_param_file(ref_time, inFile, outFile):

    """
    Update and write new parameter file from master file.

    Parameters
    ----------
    ref_time : pandas datetime object
        Reference time to create output directory and define ref time in param file
    inFile : str
        Path to master parameter file
    outFile : str
        Path to write new parameter file

    Returns
    -------
    analysis_dir
        Path where samurai analysis will be written
    """

    # Read in the file
    with open(inFile, 'r') as file :
        filedata = file.read()

    # Replace the target string
    analysis_dir = 'samurai_output_all/samurai_output_'+ref_time.strftime('%Y%m%d%H%M')
    filedata = filedata.replace('samurai_output_', analysis_dir)
    filedata = filedata.replace('xx:xx:xx', ref_time.strftime('%H:%M:%S'))

    # Write the file out again
    with open(outFile, 'w') as file:
        file.write(filedata)

    return analysis_dir



#%% main code

# grab info from tcvitals or flight+ file
parser = argparse.ArgumentParser()
parser.add_argument("CENPATH", help="TC Vitals directory", type=str)
parser.add_argument("CENFN", help="TC Vitals filename", type=str)
parser.add_argument("STORM", help="storm name (all caps)", type=str)
parser.add_argument("ANALYSISTIME", help="samurai analysis datetime (YYYYMMDDHHMM)", type=str)
parser.add_argument("--CENTYPE", default="tcvitals", help="center type (tcvitals or fplus)", type=str)
args = parser.parse_args()

if args.CENTYPE == "fplus":

    print('using flight+ data for center info')

    fplus_path = args.CENPATH+'/fplus/'
    fplus_fn = args.CENFN 
    
    fplus = Dataset(fplus_path+fplus_fn)
    fplus_centime = fplus['FL_WC_wind_center_time_offset']
    fplus_cenlat = fplus['FL_WC_wind_center_time_latitude']
    fplus_cenlon = fplus['FL_WC_wind_center_time_longitude']

    samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)
    tcvitals_time = samurai_time.floor('6h')
    
    # grab time from flight+ file?
    tcvitals_path = args.CENPATH+'/tcvitals/'
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

    make_cen_file_fplus(samurai_time, 1, 120, fplus_centime, fplus_cenlat, fplus_cenlon, u, v, './testing/')

elif args.CENTYPE == "tcvitals":

    print('using tcvitals data for center info')

    tc_vital = []
    tcvitals_path = args.CENPATH+'/tcvitals/'
    tcvitals_fn = args.CENFN 
    #tcvitals_path = '/bell-scratch/jcdehart/hot/JHT_Michael_Test/TC_Vitals/'
    #tcvitals_fn = 'syndat_tcvitals.2018'i

    # grab all lines that contain storm
    file = open(tcvitals_path+tcvitals_fn)
    searchwds = [args.STORM, args.ANALYSISTIME[0:8], args.ANALYSISTIME[8:12]]
    #searchwds = ['MICHAEL','20181010','1200']
    for line in file: 
        if all(word in line for word in searchwds):
            tc_vital.append(line)

    # grab variables from TC Vitals file, character numbers taken from EMC website
    # https://www.emc.ncep.noaa.gov/mmb/data_processing/tcvitals_description.htm
    storm_lat = float(tc_vital[0][33:36])/10.
    storm_lon = float(tc_vital[0][38:42])/10.
    storm_dir = float(tc_vital[0][44:47])
    storm_motion = float(tc_vital[0][48:51])/10.

    # convert storm motion from met degrees to standard math degrees
    storm_dir_rot = 90 + (360 - storm_dir)
    if storm_dir_rot > 360:
        storm_dir_rot = storm_dir_rot - 360

    # calculate u and v components
    u = storm_motion*np.cos(np.radians(storm_dir_rot))
    v = storm_motion*np.sin(np.radians(storm_dir_rot))

    #print([storm_lat, storm_lon, storm_dir, storm_motion, storm_dir_rot, u, v])

    # get datetime object from TC Vitals time
    center_time = pd.to_datetime(searchwds[1]+searchwds[2], format='%Y%m%d%H%M', utc=True)
    samurai_time = pd.to_datetime(args.ANALYSISTIME, format='%Y%m%d%H%M', utc=True)

    # create center file
    make_cen_file_tcvitals(center_time, samurai_time, 1, 120, storm_lat, storm_lon, u, v, './testing/')

else:
    print('error: check center type name')

