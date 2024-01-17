#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:22:00 2023

@author: jcdehart
"""

def motion_calcs(storm_dir, storm_motion):

    import numpy as np

    # convert storm motion from met degrees to standard math degrees
    storm_dir_rot = 90 + (360 - storm_dir)
    if storm_dir_rot > 360:
        storm_dir_rot = storm_dir_rot - 360

    # calculate u and v components
    u_motion = storm_motion*np.cos(np.radians(storm_dir_rot))
    v_motion = storm_motion*np.sin(np.radians(storm_dir_rot))
    
    # check for precision errors
    if (np.abs(u_motion) < 1e-3):
        u_motion = np.round(u_motion)
    if (np.abs(v_motion) < 1e-3):
        v_motion = np.round(v_motion)

    return u_motion, v_motion, storm_dir_rot


def center_fplus(args, samurai_time):

    import numpy as np
    import pandas as pd

    fplus_path = args.CENPATH+'/fplus/'
    fplus_fn = args.CENFN 
    
    fplus = Dataset(fplus_path+fplus_fn)
    fplus_centime = fplus['FL_WC_wind_center_time_offset']
    fplus_cenlat = fplus['FL_WC_wind_center_time_latitude']
    fplus_cenlon = fplus['FL_WC_wind_center_time_longitude']
    
    # FIX CENTER TIME ***********
    # fplus_centime normally is an array with the center by flight...
    center_time = pd.to_datetime(fplus_centime, unit='s', utc=True)


    # does flight plus have motion and direction????

    return center_time, fplus_cenlat, fplus_cenlon


# def center_simplex? chris's code???? *********


def center_tcvitals(args):

    import numpy as np
    import pandas as pd

    tc_vital = []
    tcvitals_path = args.CENPATH+'/tcvitals/'+args.CENTIME[:8]+'/'
    tcvitals_fn = args.CENFN 
    #tcvitals_path = '/bell-scratch/jcdehart/hot/JHT_Michael_Test/TC_Vitals/'
    #tcvitals_fn = 'syndat_tcvitals.2018'i

    if args.STORM[0:2] == 'AL':
        basin = 'L'
    elif args.STORM[0:2] == 'EP':
        basin = 'E'

    storm_id = args.STORM[2:4]+basin
    print(storm_id)

    # grab all lines that contain storm
    file = open(tcvitals_path+tcvitals_fn)
    searchwds = [storm_id, args.CENTIME[0:8], args.CENTIME[8:12]]
    #searchwds = ['MICHAEL','20181010','1200']
    for line in file: 
        if all(word in line for word in searchwds):
            tc_vital.append(line)

    print(tc_vital)

    center_time = pd.to_datetime(searchwds[1]+searchwds[2], format='%Y%m%d%H%M', utc=True)
    
    # grab variables from TC Vitals file, character numbers taken from EMC website
    # https://www.emc.ncep.noaa.gov/mmb/data_processing/tcvitals_description.htm
    storm_lat_r = float(tc_vital[0][33:36])/10.
    storm_lat_hemi = tc_vital[0][36]
    storm_lon_r = float(tc_vital[0][38:42])/10.
    storm_lon_hemi = tc_vital[0][42]
    storm_dir = float(tc_vital[0][44:47])
    storm_motion = float(tc_vital[0][48:51])/10.
    storm_intens = float(tc_vital[0][67:69])
    storm_rmw = float(tc_vital[0][70:73])

    # convert W, S to -
    if storm_lat_hemi == 'S':
        storm_lat = -1*storm_lat_r
    else:
        storm_lat = storm_lat_r

    if storm_lon_hemi == 'W':
        storm_lon = -1*storm_lon_r
    else:
        storm_lon = storm_lon_r

    u_motion, v_motion, storm_dir_rot = motion_calcs(storm_dir, storm_motion)
    
    return storm_lat, storm_lon, storm_intens, storm_rmw, storm_dir, storm_motion, center_time, u_motion, v_motion, storm_dir_rot


def center_adeck(args, samurai_time):

    import pandas as pd
    import numpy as np
    from os import system

    adeck = []
    adeck_path = args.CENPATH+'/adeck/'+args.CENTIME[0:4]+'/'
    adeck_fn = 'a'+args.STORM.lower()+args.CENTIME[0:4]+'.dat'

    system('gunzip '+adeck_path+adeck_fn+'.gz')
    print('unzipping  '+adeck_path+adeck_fn+'.gz')

    # grab all lines that contain storm
    file = open(adeck_path+adeck_fn)
    searchwds = ['OFCL', args.CENTIME[0:10], '34,']
    for line in file: 
        if all(word in line for word in searchwds):
            adeck.append(line)

    adeck2 = [x.split(',') for x in adeck]
    cols = ['BASIN', 'CY', 'YYYYMMDDHH', 'TECHNUM/MIN', 'TECH', 'TAU', 'LatN/S', 'LonE/W', 'VMAX', 'MSLP', 'TY', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'POUTER', 'ROUTER', 'RMW', 'GUSTS', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED', 'STORMNAME', 'DEPTH', 'SEAS', 'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4', 'USERDEFINED']
    cols_drop = ['BASIN', 'CY', 'TECHNUM/MIN', 'TECH', 'TY', 'POUTER', 'ROUTER', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'STORMNAME', 'DEPTH', 'SEAS', 'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4', 'USERDEFINED']

    df = pd.DataFrame(adeck2) # not sure why this went away.... .transpose()
    df.columns = cols
    df = df.drop(columns=cols_drop)
    df2 = pd.to_datetime(df.YYYYMMDDHH.str.strip(), format='%Y%m%d%H', utc=True) + pd.to_timedelta(df.TAU.str.strip().astype('int'), "h")
    df2 = pd.concat([df2,df[['VMAX','RAD1','RAD2','RAD3','RAD4','DIR','SPEED']].apply(pd.to_numeric, errors='coerce', axis=1)], axis=1)
    df2 = df2.rename(columns={0: "dt"})
    #print(df2)

    # convert lat/lon to good format
    df2['lat'] = df['LatN/S'].str.strip().str[:-1].astype('float')/10.
    df2['lon'] = df['LonE/W'].str.strip().str[:-1].astype('float')/10.
    df2['lat'][df['LatN/S'].str.strip().str[-1:] == 'S'] = df2['lat'][df['LatN/S'].str.strip().str[-1:] == 'S']*-1
    df2['lon'][df['LonE/W'].str.strip().str[-1:] == 'W'] = df2['lon'][df['LonE/W'].str.strip().str[-1:] == 'W']*-1

    # add new time and sort and interpolate
    df2.loc[len(df2), 'dt'] = samurai_time
    #print(df2)
    df2 = df2.sort_values(by=['dt']).reset_index(drop=True)
    df2['sin'] = df2['DIR'].apply(np.radians).apply(np.sin) 
    df2['cos'] = df2['DIR'].apply(np.radians).apply(np.cos) 

    # pandas version weirdness?????
    try:
        df2 = df2.interpolate()
    except:
        df2['VMAX'] = df2['VMAX'].interpolate()   
        df2['RAD1'] = df2['RAD1'].interpolate()   
        df2['RAD2'] = df2['RAD2'].interpolate()   
        df2['RAD3'] = df2['RAD3'].interpolate()   
        df2['RAD4'] = df2['RAD4'].interpolate()   
        df2['DIR'] = df2['DIR'].interpolate()   
        df2['SPEED'] = df2['SPEED'].interpolate()   
        df2['lat'] = df2['lat'].interpolate()   
        df2['lon'] = df2['lon'].interpolate()   
        df2['sin'] = df2['sin'].interpolate()   
        df2['cos'] = df2['cos'].interpolate()
        df2['DIR2'] = np.round(np.arctan2(df2['sin'],df2['cos'])*180./np.pi)

    #print(df2)


    # grab index of time and needed variables
    index = df2.loc[df2['dt'] == samurai_time].index[0]
    storm_lat = df2.loc[index,'lat']
    storm_lon = df2.loc[index,'lon']
    storm_dir = df2.loc[index,'DIR2']
    storm_motion = df2.loc[index,'SPEED']/1.94 # convert to m/s for consistency
    storm_intens = df2.loc[index,'VMAX']/1.94 # convert to m/s for consistency

    u_motion, v_motion, storm_dir_rot = motion_calcs(storm_dir, storm_motion)
    
    return storm_lat, storm_lon, storm_intens, storm_dir, storm_motion, df2.loc[index], u_motion, v_motion, storm_dir_rot


def modify_obj_jl_file(inFile, outFile, rmw_guess, sam_analysis):

    """
    Update and write new objective_simplex julia file from master file.

    Parameters
    ----------
    inFile : str
        Path to master julia script
    outFile : str
        Path to write new julia script

    Returns
    -------
    analysis_dir
        Path where samurai analysis will be written
    """

    # Read in the file
    with open(inFile, 'r') as file :
        filedata = file.read()

    # Replace the target string
    #analysis_dir = 'samurai_output/samurai_output_'+ref_time.strftime('%Y%m%d%H%M')
    #filedata = filedata.replace('samurai_output_', analysis_dir)
    filedata = filedata.replace('./samurai_path_yy', sam_analysis)
    filedata = filedata.replace('xx', str(rmw_guess))
    #filedata = filedata.replace('xx,xx,xx', str(x_guess)+','+str(y_guess)+','+str(rmw_guess))

    # Write the file out again
    with open(outFile, 'w') as file:
        file.write(filedata)

    #return analysis_dir
