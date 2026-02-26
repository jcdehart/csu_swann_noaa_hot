

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
