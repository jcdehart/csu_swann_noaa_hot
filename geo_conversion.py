

def latlon(cenlon, cenlat, dom_x, dom_y):

    import numpy as np

    latrad = np.radians(cenlat)

    # do math
    fac_lat = 111.13209 - 0.56605 * np.cos(2.0 * latrad) + 0.00012 * np.cos(4.0 * latrad) - 0.000002 * np.cos(6.0 * latrad)
    fac_lon = 111.41513 * np.cos(latrad) - 0.09455 * np.cos(3.0 * latrad) + 0.00012 * np.cos(5.0 * latrad)
    dom_lon = cenlon + (dom_x)/fac_lon # distance in km
    dom_lat = cenlat + (dom_y)/fac_lat

    return(dom_lon, dom_lat)


def xy(lat, lon, lat0, lon0):

    import numpy as np

    # Approximate radius of earth in km
    R = 6373.0
    lat_plane = np.radians(lat)
    lon_plane = np.radians(lon)
    lat_cen = np.radians(lat0)
    lon_cen = np.radians(lon0)

    dlon = lon_plane - lon_cen
    dlat = lat_plane - lat_cen
    a = np.sin(dlat / 2)**2 + np.cos(lat_cen) * np.cos(lat_plane) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    bearing = np.arctan2(np.sin(lon_plane-lon_cen)*np.cos(lat_plane), np.cos(lat_cen)*np.sin(lat_plane)-np.sin(lat_cen)*np.cos(lat_plane)*np.cos(lon_plane-lon_cen))
    #bearing is in north-facing coords...
    
    x = distance*np.cos(-1*(bearing-(np.pi/2)))
    y = distance*np.sin(-1*(bearing-(np.pi/2)))
    return(x,y)