def read_netcdf(dir, file, ncvars, alt_fl, cen):

    """
    Reads 3-D NetCDF file, recenters data, grabs location/wind variables and calculates radii/angle.

    Parameters
    ----------
    dir : str
        Directory that contains NetCDF file
    file : str
        NetCDF file name
    ncvars : dict
        Variable name dictionary {alt, x, y, lon, lat, u, v}
    alt_fl : float
        Altitude of flight-level data.
    cen : array
        Center location, in format [x, y]
    """

    import numpy as np
    from netCDF4 import Dataset

    # read NetCDF (e.g., samurai, TC radar) file and reshape output arrays
    ncfile = Dataset(dir+file)
    alt = ncfile[ncvars['alt']][:].data
    alt_lev = (alt == alt_fl)

    # cartesian file only 
    x = ncfile[ncvars['x']][:].data
    y = ncfile[ncvars['y']][:].data
    X_grid, Y_grid = np.meshgrid(x - cen[0], y - cen[1], indexing='xy') 
    lon_nc = ncfile[ncvars['lon']][:].data
    lat_nc = ncfile[ncvars['lat']][:].data
    u_storm = np.squeeze(ncfile[ncvars['u']][:].data[0,alt_lev,:,:])
    v_storm = np.squeeze(ncfile[ncvars['v']][:].data[0,alt_lev,:,:])
    u_storm[u_storm == -999] = np.nan
    v_storm[u_storm == -999] = np.nan
    rd = np.sqrt(X_grid**2 + Y_grid**2)
    th_nc = np.arctan2(Y_grid, X_grid) # radians
    th = th_nc*180./np.pi
    th[th < 0] = th[th < 0] + 360 # degrees

    return u_storm, v_storm, lon_nc, lat_nc, th, th_nc, rd, X_grid, Y_grid


def calc_wspd_earth(u_storm, v_storm, u_motion, v_motion, addmotion):

    import numpy as np

    if addmotion == True:
        # calculate earth-relative components and magnitude
        u_earth = u_storm + u_motion # u and v motion from tcvitals file 
        v_earth = v_storm + v_motion
        wspd_earth = np.sqrt(u_earth**2 + v_earth**2)
    elif addmotion == False:
        wspd_earth = np.sqrt(u_storm**2 + v_storm**2)

    return wspd_earth


def prep_hdobs_data(hdobs, x_plane, y_plane):

    """
    Prep HDOBs data.

    Parameters
    ----------
    hdobs : pandas dataframe
        HDOBs data that includes windspeed (in kts)
    x_plane : array
        Distance of plane from storm center in x direction (km)
    y_plane : array
        Distance of plane from storm center in y direction (km)
    """

    import numpy as np

    # create theta/radius grids 
    rd = np.sqrt(x_plane**2 + y_plane**2)
    th_r = np.arctan2(y_plane, x_plane)
    th = th_r*180./np.pi

    wspd_earth = hdobs.wsp.values/1.94 # CONVERTING TO M/S NEEDED FOR ALEX'S MODEL ******
    hdobs_rmw = rd[np.unravel_index(np.nanargmax(wspd_earth),np.shape(wspd_earth))] 

    return rd, th, wspd_earth, hdobs_rmw


def process_nn_vars(radii, rmw, theta, storm_dir, storm_intens, storm_motion, flight_wind, alt_plane, HDOBS):

    # manipulate variables for use in neural net

    # input units below 
    # radii: km
    # rmw: km
    # theta: degrees (math reference frame where 0 is to the right)
    # storm_dir: degrees (met reference frame, where 0 is north)
    # storm_intens: kts
    # storm_motion magnitude: m/s
    # flight_wind: m/s  

    import numpy as np

    ## RMW
    # normalize wrt RMW and ravel data
    r_norm = radii.ravel(order='C')/rmw

    ## THETA
    # convert samurai theta from math degrees (i.e., 0 = right) to met degrees (i.e., 0 = north)
    theta_met = 90 + (360 - theta)
    theta_met[theta_met > 360] = theta_met[theta_met > 360] - 360

    # rotate theta with respect to storm motion and convert to radians
    theta_motionrel = theta_met - storm_dir
    theta_motionrel[theta_motionrel < 0] = theta_motionrel[theta_motionrel < 0] + 360
    theta_nr = np.radians(theta_motionrel.ravel(order='C'))

    ## FLIGHT LEVEL WIND
    # grab flight level wind (i.e., 3 km) and reshape
    FL_wind = flight_wind.ravel(order='C')

    ## REMAINING
    # set up 2-D arrays of repeating scalar values for flight level (make automatic), best track vmax (knots), storm motion magnitude (m/s)
    alt = np.zeros_like(FL_wind)
    if HDOBS == True:
        alt = alt_plane # m, actual plane height
    else:
        alt[:] = alt_plane*1000 # m, median plane height

    BT_Vmax = np.zeros_like(FL_wind)
    BT_Vmax[:] = storm_intens*1.94 # convert to knots
    SM_mag = np.zeros_like(FL_wind)
    SM_mag[:] = storm_motion
    RMW_arr = np.zeros_like(FL_wind)
    RMW_arr[:] = rmw

    # normalized r (r/rmw), theta (wrt motion) x2, wind, altitude, vmax, storm motion magnitude
    X_ratio = np.asarray([r_norm, theta_nr, theta_nr, FL_wind, alt, BT_Vmax, SM_mag, RMW_arr])  #double up on angle for sin and cosine
    X_ratio[1,:] = np.sin(X_ratio[1,:]) # converted to radians above
    X_ratio[2,:] = np.cos(X_ratio[2,:])

    return X_ratio, r_norm
