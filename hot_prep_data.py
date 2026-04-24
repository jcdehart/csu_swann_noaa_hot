def postprocess_swann_af(r_norm, wspd_earth, predict, rd):

    import numpy as np

    predict[r_norm < 0.3] = np.nan # remove data within radius of 0.3*RMW where SWANN shouldn't be applied

    # reshape arrays and mask orig missing data
    sfc_wind_pred = wspd_earth*predict.T[0] # multiply reduction factor and flight-level wind

    # grab flight-level storm-relative data and remove bad data
    sfc_wind_pred[np.isnan(wspd_earth)] = np.nan
    sfc_wind_pred[wspd_earth*1.94 < 20] = np.nan 
    sfc_wind_pred[np.isnan(wspd_earth)] = np.nan
    print('/n')
    print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

    # grab RMW
    swann_rmw = rd[np.unravel_index(np.nanargmax(sfc_wind_pred),np.shape(sfc_wind_pred))]

    # unit conversion
    sfc_wind_pred_ms = sfc_wind_pred*1.94 # convert to m/s

    return sfc_wind_pred, swann_rmw, sfc_wind_pred_ms


def postprocess_swann_sam(r_norm, wspd_earth, predict, u_storm, v_storm, rd, sam_rmw, th_nc):

    import numpy as np

    predict[r_norm < 0.3] = np.nan # remove data within radius of 0.3*RMW where SWANN shouldn't be applied

    # reshape arrays and mask orig missing data
    sfc_wind_pred = wspd_earth*np.reshape(predict,u_storm.shape,order='C') # multiply reduction factor and flight-level wind

    # grab flight-level storm-relative data and remove bad data
    mag_3km = wspd_earth
    sfc_wind_pred[np.isnan(mag_3km)] = np.nan
    sfc_wind_pred[mag_3km*1.94 < 20] = np.nan
    sfc_wind_pred[np.isnan(mag_3km)] = np.nan # can i remove this???
    mag_3km[(rd/sam_rmw < 0.3)] = np.nan
    print('/n')
    print('predicted max sfc wind: '+str(np.nanmax(sfc_wind_pred)))

    # get RMW of max point
    swann_rmw = np.nanmin(rd[sfc_wind_pred == np.nanmax(sfc_wind_pred)]) # think this should just be a point location... *******

    # censor out boundaries with spectral ringing
    mag_3km[:4,:] = np.nan; mag_3km[-4:,:] = np.nan; mag_3km[:,:4] = np.nan; mag_3km[:,-4:] = np.nan
    sfc_wind_pred[:4,:] = np.nan; sfc_wind_pred[-4:,:] = np.nan; sfc_wind_pred[:,:4] = np.nan; sfc_wind_pred[:,-4:] = np.nan

    # convert wind speed to u and v, assuming inflow angle is 22.6, from zhang and uhlhorn 2012
    u_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.cos(th_nc) - sfc_wind_pred*np.sin(np.radians(90-22.6))*np.sin(th_nc)
    v_nc = sfc_wind_pred*np.cos(np.radians(90-22.6))*np.sin(th_nc) + sfc_wind_pred*np.sin(np.radians(90-22.6))*np.cos(th_nc)

    return mag_3km, sfc_wind_pred, swann_rmw, u_nc, v_nc


def grab_p3_alt(hdobs):

    import numpy as np

    # grab plane altitude manually
    med_hgt = (500*(hdobs.hgt/500).round()).median()

    if med_hgt == 1500.:
        alt_plane = 1.5
        print('plane hgt = '+str(alt_plane))
    elif med_hgt == 3000.:
        alt_plane = 3.
        print('plane hgt = '+str(alt_plane))
    else:
        hgt_options = np.array([1500, 3000])
        alt_plane = hgt_options[np.argmin(np.abs(med_hgt - hgt_options))]/1000
        print('plane hgt not within 500 m of options')
        print('using plane hgt = '+str(alt_plane))
        print('actual med plane hgt = '+str(med_hgt/1000))  

    return alt_plane


def vmax_calcs_af(alt_plane, hdobs, sfc_wind_pred):

    import numpy as np

    hdobs_fl_vmax = np.nanmax(hdobs.wsp)
    swann_hdobs_vmax = np.nanmax(sfc_wind_pred*1.94) # convert to kts

    # determine simplified franklin reduction based on altitude of peak HDOBs wind
    med_hgt = 500*(alt_plane[np.nanargmax(hdobs.wsp)]/500).round()

    if med_hgt == 1500.:
        sf_frac = 0.8
    elif med_hgt == 3000.:
        sf_frac = 0.9
    else:
        hgt_options = np.array([1500., 3000.])
        alt_tmp = hgt_options[np.argmin(np.abs(med_hgt - hgt_options))]
        
        if alt_tmp == 1500.:
            sf_frac = 0.8
        elif alt_tmp == 3000.:
            sf_frac = 0.9

    simp_frank = sf_frac*hdobs_fl_vmax    

    return hdobs_fl_vmax, swann_hdobs_vmax, simp_frank       


def vmax_calcs_sam(alt_plane, wspd_earth, hdobs, sfc_wind_pred, sfc_wind_pred_ac):

    import numpy as np

    if alt_plane == 1.5:
        sf_frac = 0.8
    elif alt_plane == 3.0:
        sf_frac = 0.9

    sam_fl_vmax = np.nanmax(wspd_earth*1.94) # convert to kts
    hdobs_fl_vmax = np.nanmax(hdobs.wsp) # already in kts
    swann_sam_vmax = np.nanmax(sfc_wind_pred*1.94) # convert to kts
    swann_hdobs_vmax = np.nanmax(sfc_wind_pred_ac*1.94) # convert to kts
    simp_frank = sf_frac*sam_fl_vmax

    return sam_fl_vmax, hdobs_fl_vmax, swann_sam_vmax, swann_hdobs_vmax, simp_frank


def create_fig_str(storm_name, mission, leg_start, leg_end, lat, lon, rmw, simp_frank, plane):

    """
    Creates figure title and info box strings.

    Parameters
    ----------
    storm_name : str
        Storm name (e.g., ANDREW)
    mission : str
        Flight code
    leg_start : pandas datetime
        Start time of leg
    leg_end : pandas datetime
        End time of leg
    lat : float
        Center latitude
    lon : float
        Center longitude
    rmw : float
        Radius of maximum wind (km)
    simp_frank : float
        Simplified Franklin estimate of max surface wind (kts)
    plane : str
        Plane (N: NOAA P3, A: Air Force)
    """

    import numpy as np

    if plane == 'N':
        input = 'Inputs: HRD TDR, HDOBS'
        centype = 'SAM' # modify if updated? ******
    elif plane == 'A':
        input = 'Inputs: HDOBS'
        centype = 'VDM' # modify if updated? ******

    figtitle = storm_name + ' | ' + leg_start.strftime('%Y%m%d') + ' | ' + mission + ' | ' + leg_start.strftime('%H:%M') + ' to ' + leg_end.strftime('%H:%M') + ' UTC'

    textstr = '\n'.join((
        input,
        centype + ' Center: %.2f N, %.2f W' % (lat, np.abs(lon),), # currently assuming negative longitudes **********
        'RMW: %.1f (nm)' % (rmw/1.852,), # converting from km to nm
        'Simp. Franklin: %.1f (kt)' % (simp_frank,), ))
    
    return figtitle, textstr


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
