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


def read_vdm(file, mode):
    
    import pandas as pd
    import re
    
    ## open file and read relevant lines
    # *** add other lines later ***
    f = open(file, "r")
    vdm = f.readlines()
    vdm_time = vdm[4]
    vdm_loc = vdm[5]
    vdm_header = vdm[3]
    vdm_flight = vdm[24]
    
    ## get flight, storm names
    
    header = vdm_header.split()
    flight = vdm_flight.split()
    storm_code = header[3]
    flight_code = flight[2]
    storm_name = flight[3]
    
    ## extract center time
    
    ddhhmmss = re.sub('[^0-9]','', vdm_time)
    
    # grab yeah, month info from filename
    filedt = file[-16:-4]
    vdm_file_time = pd.to_datetime(filedt, format='%Y%m%d%H%M', utc=True)
    
    # create datetime object
    # check if day is the same for midnight clock shift
    if vdm_file_time.day == int(ddhhmmss[:2]):
        vdm_center_time = pd.to_datetime(filedt[:6] + ddhhmmss[:6], format='%Y%m%d%H%M', utc=True)
        
    elif (vdm_file_time - pd.Timedelta(1, "D")).day == int(ddhhmmss[:2]):
        newday = (vdm_file_time - pd.Timedelta(1, "D")).strftime(format='%Y%m%d')
        vdm_center_time = pd.to_datetime(newday[:6] + ddhhmmss[:6], format='%Y%m%d%H%M', utc=True)
        
    else:
        print('some other problem - debug')
        
    ## extract lat lon
    
    lat = float(vdm_loc[3:8])
    lon = float(vdm_loc[15:21])
    if vdm_loc[13:14] == 'S':
        lat = lat*-1
    if vdm_loc[26:27] == 'W':
        lon = lon*-1
    
    ## *** can add additional code later!! ***
    # other vars to consider: dropsonde sfc pressure/winds, inbound/outbound max winds/rmw/time, 
    
    # basically one set of outputs if using VDM to start workflow, 
    # another if you want all the storm and flight info
    # will likely modify in the future ! *******
    if mode == 'trigger':
        leg_start = vdm_center_time - pd.Timedelta(45,unit='m')
        leg_end = vdm_center_time + pd.Timedelta(45,unit='m')
        
        print(storm_code[:4])
        print(leg_start.strftime('%Y%m%d%H%M'))
        print(leg_end.strftime('%Y%m%d%H%M'))
        print(lat)
        print(lon)
    elif mode == 'full':
        return (vdm_center_time, lat, lon, storm_code, storm_name, flight_code)


def run_wc(hdobs):

    import center_funcs
    import numpy as np

    peaks, properties, willfunc, wdir_rel = center_funcs.find_peaks(hdobs.wsp.values, hdobs.wdir.values, hdobs.dval.values, 0, 0) # change u_tc/v_tc?????
    peaks_refined = peaks.astype(int)
    window = 50
    approaches = len(peaks)
    peaks_refined = center_funcs.refine_peaks_minima(peaks_refined, willfunc)
    dt_wc, lon_wc_old, lat_wc_old = center_funcs.peaks_wc(peaks_refined, approaches, hdobs.lat.values, hdobs.lon.values, wdir_rel, hdobs.dt)
    dt_wc_inds = dt_wc[::2]

    if len(dt_wc) > 2:
        # check pressure and height vals
        hdobs_p = np.round(hdobs.p[dt_wc_inds]/50)*50
        if len(np.unique(hdobs_p)) == 1:
            lower_ind = np.argmin([hdobs.hgt.values[x] for x in dt_wc_inds])
            lon_wc = lon_wc_old[lower_ind]
            lat_wc = lat_wc_old[lower_ind]
        else:
            lower_ind = np.argmin([hdobs.hgt.values[x] for x in dt_wc_inds])
            lon_wc = lon_wc_old[lower_ind]
            lat_wc = lat_wc_old[lower_ind]
            print('centers greater than 50 hPa apart, revisit algoritm')
            print(hdobs.p[dt_wc_inds])
            print(hdobs.hgt[dt_wc_inds])
    else:
        lon_wc = lon_wc_old[dt_wc_inds[0]]
        lat_wc = lat_wc_old[dt_wc_inds[0]]

    return(lat_wc, lon_wc, dt_wc)

def read_hdob_file(f):
    
    import os
    import pandas as pd
    
    try:
        return pd.read_csv(f,sep=' ', skip_blank_lines=True, skiprows=[0,1,2,3,24,25,26], header=None, dtype=str)
    except:
        print('file has issue: '+f)
        os.remove(f)


def identify_hdob_files(all_files, storm, start_time, end_time, inDir):

    import numpy as np
    import pandas as pd
    import os
    import subprocess

    dfs_init = [read_hdob_file(f) for f in all_files]
    dfs = [x for x in dfs_init if x is not None] # remove instances with None
    files = [file for file, x in zip(all_files, dfs_init) if x is not None]
    good = []
    planes = []
    missions = []
    bad_plane = None
    bad_plane_files = []

    for i in range(len(dfs)): 

        if np.isin(files[i],bad_plane_files):
            print('file is from landing plane: '+files[i])
            planes.append(bad_plane)
            continue

        # if np.isin(files[i], notread):
        #     print('file not read due to error: '+files[i])
        #     continue

        dfs[i][0] = files[i][-17:-9]+dfs[i][0] # add YYYYMMDD to existing time strings

        # altitude check to see if transit flight mixed in
        # STILL NEED TO FIX MULTIPLE PLANES IN STORM AT ONCE *******
        
        f = open(files[i])
        lines = f.readlines()
        plane = lines[3].split()[0]
        mission = lines[3].split()[1]
        planes.append(plane)

        hdob_alt = dfs[i][4].astype('float')
        if (np.nanmean(hdob_alt) > 5000):
            print('altitude > 5 km ('+str(np.nanmean(hdob_alt))+'), possibly transit, deleting file '+files[i])
            os.system('rm -rf '+files[i])
            continue

        # altitude check for landing plane (altitude < 1000)
        if ((hdob_alt < 1000).any()):

            hdob_dt = pd.to_datetime(dfs[i][0],format='%Y%m%d%H%M%S',utc=True)

            if (hdob_dt.iloc[0] > end_time) | (hdob_dt.iloc[-1] < start_time):
                print('hdob file out of order - not within time range, deleting file '+files[i])
                os.system('rm -rf '+files[i])
                continue

            print('plane altitude < 1 km, '+plane+' likely landing, will delete files from plane')
            bad_plane = plane

            # use grep to find files with plane
            landing_files = subprocess.check_output('grep -l '+plane+' '+inDir+'/*.hdob',shell=True).decode().strip().split('\n')

            if len(landing_files) > 0:

                for lf in range(len(landing_files)):
                    print('deleting file '+landing_files[lf])
                    os.system('rm -rf '+landing_files[lf])
                    bad_plane_files.append(landing_files[lf])

                continue

            else:
                print('files probably already deleted')
        
        # make sure file all correspond to same storm
        if (os.system('grep '+storm+' '+files[i]) != 256):
            good.append(i)
            missions.append(mission)
        elif (os.system('grep TDR '+files[i]) != 256):
            good.append(i)
            missions.append(mission)
            print('storm name not found, but TDR string found - unnamed storm?')
        else:
            print('storm name not found in HDOBS')
            # maybe delete files?

    # remove any remaining files from descending aircraft
    # should rename vars lol
    if bad_plane is not None:
        planes_good = [planes[i] for i in good] # select plane names that meet altitude reqs
        only_instorm_flights = [i for i, x in enumerate(planes_good) if x != bad_plane] # indices that aren't a landing plane
        good_final = [good[i] for i in only_instorm_flights] # remove any good indices that were with a landing plane
        missions_final = [missions[i] for i in only_instorm_flights]
    else:
        good_final = good
        missions_final = missions


    return(dfs, good_final, missions_final)


def read_hdobs(plane, storm, analysis_type, start_time, end_time):

    # assumes files have already been moved into samurai_parent/samurai_input dir

    import numpy as np
    import pandas as pd
    import glob

    if analysis_type == 'HDOBS':
        inDir = './hdobs_parent/hdobs_input'
    elif analysis_type == 'SAMURAI':
        inDir = './samurai_parent/samurai_input'

    # sort files by name, read in files using pandas, add YYYYMMDD from file name to time, concat files together
    all_files = sorted(glob.glob(inDir+'/*'+plane+'*.hdob'))
    # files = sorted(glob.glob(inDir+'/*'+plane+'*.hdob')) # set in case some were deleted

    dfs, good_final, missions = identify_hdob_files(all_files, storm, start_time, end_time, inDir)

    # get mission
    if len(list(set(missions))) == 1:
        mission = list(set(missions))
    else:
        mission = max(set(missions), key=missions.count)

    # pare down dataframes
    dfs_good = [dfs[i] for i in good_final]
    full_ts = pd.concat(dfs_good,ignore_index=True)

    # remove any lines where final column is NaN - likely a missing value in earlier column, remove in case
    full_ts = full_ts[full_ts.iloc[:,12].notna()].reset_index(drop=True)

    # create new dataframe with actual values we want, starting with datetime
    hdobs_all = pd.DataFrame(data={'dt':pd.to_datetime(full_ts[0],format='%Y%m%d%H%M%S',utc=True)})

    # correct for crossing of midnight - should be LARGER than last value? probably fix in future....
    hdobs_all.loc[(hdobs_all.dt > hdobs_all.dt[len(hdobs_all.dt)-1]), 'dt'] = hdobs_all.dt[hdobs_all.dt > hdobs_all.dt[len(hdobs_all.dt)-1]] - pd.Timedelta(1,'day')

    if hdobs_all.dt.diff().max() <= pd.Timedelta(60,'s'):
        print('max difference of 60 seconds - conversion good')
    else:
        print(hdobs_all.dt.diff().max())
        print('issue with hdobs timing order')

    # convert lat/lon to good format (degrees and then MINUTES #facepalm)
    hdobs_all['lat'] = full_ts[1].str.strip().str[:2].astype('float') + full_ts[1].str.strip().str[2:-1].astype('float')/60.
    hdobs_all['lon'] = full_ts[2].str.strip().str[:3].astype('float') + full_ts[2].str.strip().str[3:-1].astype('float')/60.
    hdobs_all.loc[full_ts[1].str.strip().str[-1:] == 'S','lat'] = hdobs_all.loc[full_ts[1].str.strip().str[-1:] == 'S','lat']*-1
    hdobs_all.loc[full_ts[2].str.strip().str[-1:] == 'W','lon'] = hdobs_all.loc[full_ts[2].str.strip().str[-1:] == 'W','lon']*-1

    # grab pressure, convert to hPa (assuming all pressure below 1000 for now)
    hdobs_all['p'] = full_ts[3].str.replace('////','999').astype('float')/10. ### might need to check for errors in the future *****

    # grab height, keep in m
    hdobs_all['hgt'] = full_ts[4].astype('float')

    # grab wind speed and dir (dir already in met coords), speed is in knots
    hdobs_all['wdir'] = full_ts[8].str[0:3].replace('///','999').astype('float')
    hdobs_all['wsp'] = full_ts[8].str[3:].replace('///','999').astype('float')

    # get sfmr
    hdobs_all['sfmr'] = full_ts[10].str.replace('///','999').astype('float')

    # deal with flags
    hdobs_all['flag_pos'] = full_ts[12].str[0].astype('float')
    hdobs_all['flag_met'] = full_ts[12].str[1].astype('float')

    # set questionable wind data to nan
    hdobs_final = hdobs_all.copy(deep=True)
    hdobs_final[hdobs_all == 999] = np.nan
    hdobs_final.loc[((hdobs_all.flag_met == 2) | (hdobs_all.flag_met == 4) | (hdobs_all.flag_met == 6) | (hdobs_all.flag_met == 9)), 'wsp'] = np.nan
    hdobs_final.loc[((hdobs_all.flag_met == 3) | (hdobs_all.flag_met == 5) | (hdobs_all.flag_met == 6) | (hdobs_all.flag_met == 9)), 'sfmr'] = np.nan

    # remove any entries above 650 hPa to focus on low-level flight data and bad positional flag
    hdobs = hdobs_final[(hdobs_all.p > 605.) & (hdobs_all.flag_pos == 0)].reset_index()
    
    # calculate D-value
    # **** may need to use adaptive value here if legs contain two levels... *******
    z_p_700 = 3012
    z_p_850 = 1457
    med_p = hdobs['p'].median() # get median value (hope this weeds out outliers)
    if 100*np.round(med_p/100) == 700:
        hdobs['dval'] = hdobs['hgt'] - z_p_700
    elif (med_p > 800) | (med_p < 900):
        hdobs['dval'] = hdobs['hgt'] - z_p_850
    else:
        print('need additional levels')


    return(hdobs, mission[0])


def center_tcvitals(args):

    import numpy as np
    import pandas as pd
    import os

    centime_firstguess = pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True).floor('6H')
    tc_vital = []
    tcvitals_path = args.CENPATH+'/tcvitals/'+centime_firstguess.strftime('%Y%m%d')+'/'
    tcvitals_fn = args.CENFN.replace('XX',centime_firstguess.strftime('%H'))
    #tcvitals_path = '/bell-scratch/jcdehart/hot/JHT_Michael_Test/TC_Vitals/'
    #tcvitals_fn = 'syndat_tcvitals.2018'i

    if args.STORM[0:2] == 'AL':
        basin = 'L'
    elif args.STORM[0:2] == 'EP':
        basin = 'E'

    storm_id = args.STORM[2:4]+basin
    print(storm_id)

    # check if real time files exist, otherwise go to archive
    if os.path.isfile(tcvitals_path+tcvitals_fn):
        file = open(tcvitals_path+tcvitals_fn)
        centime = centime_firstguess
        print('tcvitals time: '+centime_firstguess.strftime('%Y%m%d%H%M'))
    else:
        print('orig time file does not exist, trying 6 hours before')

        # reset path and names and try for time 6 hours earlier
        centime_secondguess = centime_firstguess - pd.Timedelta(hours=6)
        tcvitals_path_early = args.CENPATH+'/tcvitals/'+centime_secondguess.strftime('%Y%m%d')+'/'
        tcvitals_fn_early = args.CENFN.replace('XX',centime_secondguess.strftime('%H'))
        
        if os.path.isfile(tcvitals_path_early+tcvitals_fn_early):
            file = open(tcvitals_path_early+tcvitals_fn_early)
            centime = centime_secondguess
            print('tcvitals time: '+centime_secondguess.strftime('%Y%m%d%H%M'))

        else:
            # assuming that data is in the archive
            print('neither time exists - moving to 2023 archive')
            tcvitals_path_archive = args.CENPATH+'/tcvitals/archive/'
            tcvitals_fn_archive = 'syndat_tcvitals.'+centime_firstguess.strftime('%Y')
            file = open(tcvitals_path_archive+tcvitals_fn_archive)

            # set 4 hours as required threshold for using first guess
            if (pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True) - centime_firstguess) / pd.Timedelta(1, 'h') > 4:
                centime = centime_firstguess
            else:
                centime = centime_secondguess
            

    # grab all lines that contain storm
    searchwds = [storm_id, centime.strftime('%Y%m%d'), centime.strftime('%H%M')]
    #searchwds = [storm_id, args.CENTIME[0:8], args.CENTIME[8:12]]
    #searchwds = ['MICHAEL','20181010','1200']
    for line in file: 
        if all(word in line for word in searchwds):
            tc_vital.append(line)

    #print(tc_vital)

    center_time = pd.to_datetime(searchwds[1]+searchwds[2], format='%Y%m%d%H%M', utc=True)
    
    # grab variables from TC Vitals file, character numbers taken from EMC website
    # https://www.emc.ncep.noaa.gov/mmb/data_processing/tcvitals_description.htm
    storm_name = tc_vital[0][9:18].strip()
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
    
    return storm_lat, storm_lon, storm_intens, storm_rmw, storm_dir, storm_motion, center_time, u_motion, v_motion, storm_dir_rot, storm_name


def center_adeck(args, samurai_time):

    import pandas as pd
    import numpy as np
    from os import system

    centime = pd.to_datetime(args.STARTTIME,format='%Y%m%d%H%M',utc=True).floor('6H')
    adeck = []
    adeck_path = args.CENPATH+'/adeck/'+centime.strftime('%Y')+'/'
    adeck_fn = 'a'+args.STORM.lower()+centime.strftime('%Y')+'.dat'
    #adeck_path = args.CENPATH+'/adeck/'+args.CENTIME[0:4]+'/'
    #adeck_fn = 'a'+args.STORM.lower()+args.CENTIME[0:4]+'.dat'

    system('gunzip '+adeck_path+adeck_fn+'.gz')
    print('unzipping  '+adeck_path+adeck_fn+'.gz')

    # grab all lines that contain storm
    file = open(adeck_path+adeck_fn)
    searchwds = ['OFCL', centime.strftime('%Y%m%d%H'), '34,']
    #searchwds = ['OFCL', args.CENTIME[0:10], '34,']
    for line in file: 
        if all(word in line for word in searchwds):
            adeck.append(line)

    if len(adeck) == 0:
        print('no data at center time, trying 6 hours before')
        centime_secondguess = centime - pd.Timedelta(hours=6)
        print(centime_secondguess.strftime('%Y%m%d%H'))
        
        searchwds2 = ['OFCL', centime_secondguess.strftime('%Y%m%d%H'), '34,']
        file = open(adeck_path+adeck_fn)
        for line in file: 
            if all(word in line for word in searchwds2):
                print(line)
                adeck.append(line)

        print(adeck)

    adeck2 = [x.split(',') for x in adeck]
    cols = ['BASIN', 'CY', 'YYYYMMDDHH', 'TECHNUM/MIN', 'TECH', 'TAU', 'LatN/S', 'LonE/W', 'VMAX', 'MSLP', 'TY', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'POUTER', 'ROUTER', 'RMW', 'GUSTS', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED', 'STORMNAME', 'DEPTH', 'SEAS', 'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4', 'USERDEFINED']
    cols_drop = ['BASIN', 'CY', 'TECHNUM/MIN', 'TECH', 'TY', 'POUTER', 'ROUTER', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'STORMNAME', 'DEPTH', 'SEAS', 'SEASCODE', 'SEAS1', 'SEAS2', 'SEAS3', 'SEAS4', 'USERDEFINED']
    cols_sm = ['BASIN', 'CY', 'YYYYMMDDHH', 'TECHNUM/MIN', 'TECH', 'TAU', 'LatN/S', 'LonE/W', 'VMAX', 'MSLP', 'TY', 'RAD', 'WINDCODE', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'POUTER', 'ROUTER', 'RMW', 'GUSTS', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS', 'DIR', 'SPEED','newline']
    cols_drop_sm = ['BASIN', 'CY', 'TECHNUM/MIN', 'TECH', 'TY', 'POUTER', 'ROUTER', 'EYE', 'SUBREGION', 'MAXSEAS', 'INITIALS','newline']

    df = pd.DataFrame(adeck2) # not sure why this went away.... .transpose()
    if df.shape[1] == 28:
        df.columns = cols_sm
        df = df.drop(columns=cols_drop_sm)
    else:
        df.columns = cols
        df = df.drop(columns=cols_drop)

    df2 = pd.to_datetime(df.YYYYMMDDHH.str.strip(), format='%Y%m%d%H', utc=True) + pd.to_timedelta(df.TAU.str.strip().astype('int'), "h")
    df2 = pd.concat([df2,df[['VMAX','RAD1','RAD2','RAD3','RAD4','DIR','SPEED']].apply(pd.to_numeric, errors='coerce', axis=1)], axis=1)
    df2 = df2.rename(columns={0: "dt"})
    #print(df2)

    # convert lat/lon to good format
    df2['lat'] = df['LatN/S'].str.strip().str[:-1].astype('float')/10.
    df2['lon'] = df['LonE/W'].str.strip().str[:-1].astype('float')/10.
    df2.loc[df['LatN/S'].str.strip().str[-1:] == 'S','lat'] = df2.loc[df['LatN/S'].str.strip().str[-1:] == 'S','lat']*-1
    df2.loc[df['LonE/W'].str.strip().str[-1:] == 'W','lon'] = df2.loc[df['LonE/W'].str.strip().str[-1:] == 'W','lon']*-1
    #df2['lon'][df['LonE/W'].str.strip().str[-1:] == 'W'] = df2['lon'][df['LonE/W'].str.strip().str[-1:] == 'W']*-1

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
