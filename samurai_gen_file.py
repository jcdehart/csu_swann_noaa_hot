import pandas as pd
import numpy as np
import argparse


def make_cen_file(ref_time, sam_start, sam_end, lat, lon, u, v, outDir):

    """
    Creates a center file for samurai from a tcvitals file, rewriting Michael's perl script into python

    Parameters
    ----------
    ref_time : pandas datetime value
        Datetime object defining the time when the domain reaches the reference lat/lon 
    sam_start : pandas datetime value
        Datetime object defining the samurai start time
    sam_end : pandas datetime value
        Datetime object defining the samurai end time
    lat : float
        Reference latitude
    lon : float
        Reference longitude
    u : float
        Zonal velocity (m/s)
    v: float
        Meridional velocity (m/s)
    outDir: str
        Path where center file will be written
    """

    # init_time = reference time corresponding to lat/lon, start_time/end_time specified by user
    init_time = ref_time
    start_time = sam_start
    end_time = sam_end

    # set up date range array, beginning depends on when earliest time is
    if (init_time < start_time):
        dt_range = pd.date_range(start=init_time, end=end_time, freq='S')
    else:
        dt_range = pd.date_range(start=start_time, end=end_time, freq='S')

    cen_time = pd.DataFrame(dt_range.strftime('%H%M%S'), columns=['cen_time'])
    cen_time['v'] = v
    cen_time['u'] = u
    cen_time['dt'] = dt_range

    # do some manipulating to place datetime in format samurai expects (hours after midnight increase to 24, 25, 26...)
    # **** CHANGE TO FIXED VERSION ****
    diff = (dt_range - pd.to_datetime(str(dt_range[0].year)+str(dt_range[0].month)+str(dt_range[0].day), format='%Y%m%d', utc=True)).days
    cen_time['new_hour'] = (dt_range.hour + diff.values*24).astype(str) # caculate relative hour
    cen_time['new_hour'] = cen_time['new_hour'].str.zfill(2) # pad zeros
    cen_time['cen_time_sam'] = cen_time.apply(lambda x:x['cen_time'].replace(x['cen_time'][0:2], x['new_hour'], 1), axis=1)

    # produce lat/lon arrays, based on storm/domain motion
    if (u == 0.) & (v == 0.):
        cen_time['lat'] = lat
        cen_time['lon'] = lon
        latlon = [lat, lon]
    else:
        # set up vars and arrays
        latrad = np.radians(lat)
        delta_sec = (dt_range - ref_time).total_seconds()

        # do math
        fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
        fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
        xc = (u*delta_sec)/1000.
        yc = (v*delta_sec)/1000.
        cen_time['lat'] = lat + yc/fac_lat
        cen_time['lon'] = lon + xc/fac_lon
        latlon = [cen_time['lat'].loc[delta_sec == 0], cen_time['lon'].loc[delta_sec == 0]]

    # select only dates within user-defined range of analysis time
    mask = delta_sec >= (start_time - ref_time).total_seconds()
    cen_time_sm = cen_time.loc[mask]

    # rearrange columns to desired order
    column_titles = ['cen_time_sam','lat','lon','v','u']
    cen_time_final = cen_time_sm.reindex(columns = column_titles)

    # print(cen_time_final)

    # write file
    cen_file_name = start_time.strftime('%Y%m%d') + '.cen'
    cen_time_final.to_csv(outDir+cen_file_name, sep=' ', header=False, index=False)

    return(latlon)


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
    analysis_dir = 'samurai_output/samurai_output_'+ref_time.strftime('%Y%m%d%H%M')
    filedata = filedata.replace('samurai_output_', analysis_dir)
    filedata = filedata.replace('xx:xx:xx', ref_time.strftime('%H:%M:%S'))

    # Write the file out again
    with open(outFile, 'w') as file:
        file.write(filedata)

    return analysis_dir


