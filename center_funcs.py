#
# Program to find center of storm using the Willoughby-Chelmow method
#
#
# Import modules | define routine
from netCDF4 import Dataset
import numpy as np
import time, sys
from scipy import signal, interpolate
from math import radians, degrees, sin, cos, asin, acos, sqrt
import pandas as pd

#### functions

def great_circle(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6371 * (
        acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    )


def find_peaks(wspd_f, wdir_f, dval_f, u_tc, v_tc):
    # Convert wind to Tangential and Radial components
    md = 270 - wdir_f # from meteorological direction to math direction
    md_rad = md * np.pi / 180.0 # from degrees to radians
    u_f = wspd_f * np.cos(md_rad)
    v_f = wspd_f * np.sin(md_rad)
    #
    # Storm relative wind
    u_rel = u_f - u_tc
    v_rel = v_f - v_tc
    #
    # Get wind direction of the storm-relative wind
    wdir_rel = 270 - 180.0 * np.arctan2(v_rel, u_rel) / np.pi
    ind = np.where(wdir_rel >= 360.0)
    wdir_rel[ind] = wdir_rel[ind] - 360.0
    ind = np.where(wdir_f == 360.0)
    wdir_f[ind] = 0.0
    ind = np.logical_and(np.abs(u_rel)==0, np.abs(v_rel)==0)
    wdir_rel[ind] = 0.0
    #
    wspd_rel = np.sqrt(u_rel * u_rel + v_rel * v_rel)
    #
    # Calculate the Willoughby function
    willfunc = wspd_rel * wspd_rel + 9.8 * dval_f
    #
    # all local minima in the Willoughby function
    peaks, properties = signal.find_peaks(-willfunc, prominence=(800, None))

    return peaks, properties, willfunc, wdir_rel


def refine_peaks_nhc(peaks, approaches, dt_f, latsub, lonsub, date_bt, lat_bt, lon_bt):
    
    peaks_refined = np.array([])
    
    print('Refining centers based on best track')
    for cpo in range(0, approaches):
        latsub = lat_f[peaks[cpo]-50:peaks[cpo]+51]
        lonsub = lon_f[peaks[cpo]-50:peaks[cpo]+51]
        
        # Interpolate best track data locally to do sanity check
        dtsub = dt_f[peaks[cpo]-50:peaks[cpo]+51] # apply to other variables
        dtsub_ts = (dtsub - date_bt[0]).dt.total_seconds() # just convert to seconds from first best track time
        date_bt_ts = (date_bt - date_bt[0]).dt.total_seconds()
        f = interpolate.interp1d(date_bt_ts, lat_bt)
        latbtsub = f(dtsub_ts)
        f = interpolate.interp1d(date_bt_ts, lon_bt)
        lonbtsub = f(dtsub_ts)
    	
        d1 = np.sqrt((lonbtsub-lonsub)*(lonbtsub-lonsub) + (latbtsub-latsub)*(latbtsub-latsub))
        locind = np.argmin(d1)
        d = great_circle(lonsub[locind], latsub[locind], lonbtsub[locind], latbtsub[locind])
        if (d <= 20):
            peaks_refined = np.append(peaks_refined, peaks[cpo])

    return peaks_refined


def refine_peaks_minima(peaks_refined, willfunc):

    # Go through sample, and any cases where two local minima are found within close
    #  proximity, collect the closest of the points
    if len(peaks_refined > 1):
        distance_peaks = peaks_refined[1:]-peaks_refined[:-1]
        indduplicate = np.where(distance_peaks < 1000)[0]
        #
        if indduplicate.size > 0:
            ind_peaks_delete = np.array([])
            j = 0
            for i in range(0, len(peaks_refined)):
                if peaks_refined[i] == peaks_refined[indduplicate[j]]:
                    will1 = willfunc[peaks_refined[indduplicate[j]]]
                    will2 = willfunc[peaks_refined[indduplicate[j]+1]]
                    if will1 > will2:
                        ind_peaks_delete = np.append(ind_peaks_delete,indduplicate[j])
                    else:
                        ind_peaks_delete = np.append(ind_peaks_delete,indduplicate[j]+1)
                    
                    if j < len(indduplicate)-1:
                        j = j + 1
	 #
	 # Get rid of bad data
            peaks_refined = np.delete(peaks_refined, ind_peaks_delete.astype(int))

    if len(peaks_refined > 1):
        distance_peaks = peaks_refined[1:]-peaks_refined[:-1]
        indduplicate = np.where(distance_peaks < 1000)[0]
	#
        if indduplicate.size > 0:
            ind_peaks_delete = np.array([])
            j = 0
            for i in range(0, len(peaks_refined)):
                if peaks_refined[i] == peaks_refined[indduplicate[j]]:
                    will1 = willfunc[peaks_refined[indduplicate[j]]]
                    will2 = willfunc[peaks_refined[indduplicate[j]+1]]
                    if will1 > will2:
                        ind_peaks_delete = np.append(ind_peaks_delete,indduplicate[j])
                    else:
                        ind_peaks_delete = np.append(ind_peaks_delete,indduplicate[j]+1)
                    if j < len(indduplicate)-1:
                        j = j + 1
	 #
	 # Get rid of bad data
            peaks_refined = np.delete(peaks_refined, ind_peaks_delete.astype(int))

    return peaks_refined


def peaks_wc(peaks_refined, approaches, lat_f, lon_f, wdir_rel, dt_f):

    lon_wc = np.array([])
    lat_wc = np.array([])
    dt_wc = np.array([])
    #
    approaches = len(peaks_refined)
    #
    for cpo in range(0, approaches):
        i = 0
        
        # deal with peaks that are within 50 indices of either end of time series
        if (peaks_refined[cpo]-50 < 0):
            st = 0
        else:
            st = peaks_refined[cpo]-50

        if (peaks_refined[cpo]+51 >= len(lat_f)):
            en = -1
        else:
            en = peaks_refined[cpo]+51

        latsub = lat_f[st:en]
        lonsub = lon_f[st:en]
        wdirsub = wdir_rel[st:en]
	
	# Date information
        dtsub = dt_f[st:en].reset_index()
	
	# Initialize matrices
        b0 = 0
        b1 = 0
	
	# fill in matrices with values
        sn = np.sin(wdirsub * np.pi / 180.0)
        cs = np.cos(wdirsub * np.pi / 180.0)
        tm0 = cs * cs
        tm1 = cs * sn
        tm2 = sn * sn
        m00 = np.sum(tm2)
        m01 = np.sum(tm1)
        m10 = np.sum(tm1)
        m11 = np.sum(tm0)
        b0 = np.matmul(tm1, latsub) + np.matmul(tm2, lonsub)
        b1 = np.matmul(tm1, lonsub) + np.matmul(tm0, latsub)
        dts = m00 * m11 - m10 * m01
        dty = m00 * b1 - m01 * b0
        dtx = m10 * b1 - m11 * b0
	
        rellon = -dtx / dts
        rellat = dty / dts
	
	# Find best time to mark center estimate
        difflon = lonsub - rellon
        difflat = latsub - rellat
        errorl = np.sqrt(difflon * difflon + difflat * difflat)
        #print(dtsub)
        #print(dt_f)
        imin = np.argmin(errorl)
        #print(imin)
        #print(errorl)
        dt_wc = np.append(dt_wc, dtsub.iloc[imin])
	
        lon_wc = np.append(lon_wc, rellon)
        lat_wc = np.append(lat_wc, rellat)

    return dt_wc, lon_wc, lat_wc


def read_flight_plus(file_in):

    # Variables needed for algorithm
    # 
    #  Time, wind, lat, lon, altitude, dval
    #
    # Need to set storm motion vector
    #
    # Read in flight-level data
    f = Dataset(file_in, 'r', format = 'NETCDF4')
    time_f = f.variables['FL_PLATFORM_yyyymmddhhmmss'][:]
    #time_f = f.variables['all_yyyymmddhhmmss'][:]
    wspd_f = np.array(f.variables['FL_PLATFORM_wind_speed_30s_average'][:]) # m/s
    #wspd_f = np.array(f.variables['all_wind_speed'][:]) # m/s
    wdir_f = np.array(f.variables['FL_PLATFORM_wind_direction'][:]) # degrees
    #wdir_f = np.array(f.variables['all_wind_direction'][:]) # degrees
    dval_f = np.array(f.variables['FL_PLATFORM_deviation_value'][:]) # m
    #dval_f = np.array(f.variables['all_deviation_value'][:]) # m
    lat_f = np.array(f.variables['FL_PLATFORM_latitude'][:])
    #lat_f = np.array(f.variables['all_platform_latitude'][:])
    lon_f = np.array(f.variables['FL_PLATFORM_longitude'][:])
    #lon_f = np.array(f.variables['all_platform_longitude'][:])
    f.close()
    ind = np.where(wspd_f >= 0)
    time_f = time_f[ind]
    wspd_f = wspd_f[ind]
    wdir_f = wdir_f[ind]
    dval_f = dval_f[ind]
    lat_f = lat_f[ind]
    lon_f = lon_f[ind]

    ind = np.where(dval_f >= -4000)
    time_f = time_f[ind]
    wspd_f = wspd_f[ind]
    wdir_f = wdir_f[ind]
    dval_f = dval_f[ind]
    lat_f = lat_f[ind]
    lon_f = lon_f[ind]
    #
    # Get more specific time information
    dt_f = pd.to_datetime(time_f,format='%Y%m%d%H%M%S',utc=True)

    return wspd_f, wdir_f, dval_f, lat_f, lon_f, dt_f


def read_nhc(file_bt):

    # Read in best track
    lat_bt = np.array([])
    lon_bt = np.array([])
    time_bt = np.array([])
    
    with open(file_bt, 'r') as datafile:
        for line in datafile:
            split_line = line.split() # get line of text from text file
            if int(split_line[11].replace(",","")) <= 35:
                time_bt = np.append(time_bt, split_line[2].replace(",",""))
                lat_bt = np.append(lat_bt, split_line[6].replace(",",""))
                lon_bt = np.append(lon_bt, split_line[7].replace(",",""))

    lat_bt = [float(s[:-1])/10.0 for s in lat_bt]
    lon_bt = [-float(s[:-1])/10.0 for s in lon_bt]
    datetime_bt = pd.to_datetime(time_bt,format='%Y%m%d%H%M%S',utc=True) # convert to pandas datetime

    return lat_bt, lon_bt, datetime_bt


def write_ncfile(lon_wc, lat_wc, yr_wc, mo_wc, da_wc, hr_wc, mn_wc, sc_wc):

    # Put resulting data into output file
    ncenters = len(lon_wc)
    
    # Open new file and define variables
    dataset = Dataset(file_out, 'w', format = 'NETCDF4')
    ncenters = dataset.createDimension('ncenters', ncenters)
    lon_center = dataset.createVariable('Longitude Center', np.float32, ('ncenters'))
    lat_center = dataset.createVariable('Latitude Center', np.float32, ('ncenters'))
    yr_center = dataset.createVariable('Year Center', np.int32, ('ncenters'))
    mo_center = dataset.createVariable('Month Center', np.int32, ('ncenters'))
    da_center = dataset.createVariable('Day Center', np.int32, ('ncenters'))
    hr_center = dataset.createVariable('Hour Center', np.int32, ('ncenters'))
    mn_center = dataset.createVariable('Minute Center', np.int32, ('ncenters'))
    sc_center = dataset.createVariable('Second Center', np.int32, ('ncenters'))
    
    # Variable attributes
    lon_center.units = 'degrees east'
    lat_center.units = 'degrees west'
    
    # Fill in variables
    lon_center[:] = lon_wc
    lat_center[:] = lat_wc
    yr_center[:] = yr_wc
    mo_center[:] = mo_wc
    da_center[:] = da_wc
    hr_center[:] = hr_wc
    mn_center[:] = mn_wc
    sc_center[:] = sc_wc
    
    dataset.close()

    return


def run_wc_code(file_in):
    ####### main code
    # Temporary assigned file [will eventually be fed automated data]
    #file_in = '/bell-scratch/jcdehart/hot/ingest_dir/center_data/ncar/flightplus_test/FLIGHT-RT_AL132023_LEE_17_AF309_1713A.nc'
    #file_in = 'FLIGHT_2019_AL022019_BARRY_L1_v1.3.nc'
    #'./ingest_dir/center_data/adeck/2023'
    #file_bt = 'bal022019.dat'

    # read in flight+ data
    wspd_f, wdir_f, dval_f, lat_f, lon_f, dt_f = read_flight_plus(file_in)

    # define filenames for I/O
    suffix = '.nc'
    file_base = file_in[6:-11]
    #file_out = ('Centers' + file_base + '.nc')

    u_tc = 0
    v_tc = 0

    # read NHC data for sanity check on center
    #lat_bt, lon_bt, datetime_bt = read_nhc(file_bt) # does this need to be adeck for realtime?

    # find initial peaks
    peaks, properties, willfunc, wdir_rel = find_peaks(wspd_f, wdir_f, dval_f, u_tc, v_tc)

    # Loop through potential centers and get rid of junk ones
    window = 50
    approaches = len(peaks)

    # refine based on NHC best track/realtime
    #peaks_refined = refine_peaks_nhc(peaks, approaches, dt_f, latsub, lonsub, date_bt, lat_bt, lon_bt):
    #peaks_refined = peaks_refined.astype(int)

    # if not refining based on NHC... (#### fix in future)
    peaks_refined = peaks.astype(int)

    # refine peaks based on nearby minima
    peaks_refined = refine_peaks_minima(peaks_refined, willfunc)

    # run W-C algorithm
    print('Refining centers based Willoughby and Chelmow')
    dt_wc, lon_wc, lat_wc = peaks_wc(peaks_refined, approaches, lat_f, lon_f, wdir_rel, dt_f)

    # might be an error here? may need an index-based approach ***********
    print('W-C center lon: '+str(lon_wc))
    print('W-C center lat: '+str(lat_wc))
    print(dt_wc)

    # plot figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize = (7, 7))
    ax.plot(dval_f)
    ax.plot(willfunc)
    ax.plot(peaks, willfunc[peaks], 'x')
    ax.plot(peaks_refined, willfunc[peaks_refined], 'o')
    #plt.ylim(0, 10000)
    #plt.show()
    plt.savefig('./willoughby.png',dpi=100)

    return dt_wc, lon_wc, lat_wc
