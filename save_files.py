def save_txt(lat, lon, fl_vmax, swann_vmax, rmw, simp_frank, inDir, args, analysis_time, analysis_type):

    if analysis_type == 'SAM':
        f = open(inDir+'txt_output/'+args.STORM+'_'+analysis_time+'_data_samurai.txt','w')
        lines = ['Inputs: HRD TDR, HDOBS\n', 'SAMURAI Center: '+str(lat)+', '+str(lon)+'\n', 'SAMURAI Vmax (kts): '+str(fl_vmax)+'\n', 'SWANN Vmax (kts): '+str(swann_vmax), 'SWANN RMW (nm): '+str(rmw/1.852), 'Simplified Franklin (kts): '+str(simp_frank)]
        f.writelines(lines)
        f.close()
    elif analysis_type == 'HDOBS':
        f = open(inDir+'txt_output/'+args.STORM+'_'+analysis_time+'_data_hdobsonly.txt','w')
        lines = ['Inputs: HDOBS\n', 'W-C Center: '+str(lat)+', '+str(lon)+'\n', 'HDOBS Vmax (kts): '+str(fl_vmax)+'\n', 'SWANN Vmax (kts): '+str(swann_vmax)+' ', 'SWANN RMW (nm): '+str(rmw/1.852)+' ', 'Simplified Franklin (kts): '+str(simp_frank)]
        f.writelines(lines)
        f.close()

def save_1d_netcdf(hdobs, u_nc, v_nc, samurai_time, args):

    from netCDF4 import Dataset
    import numpy as np
    import pandas as pd

    # open file
    ncfile_sfc = Dataset('./nn_output/HOT_HDOBS_sfc_analysis_'+args.STORM+'_'+samurai_time.strftime('%Y%m%d%H%M')+'.nc',mode='w',format='NETCDF4') 
        
    # set up metadata
    ncfile_sfc.title='CSU Predicted Surface Wind'
    ncfile_sfc.subtitle="Generated using CSU SWANN"

    # set up variables
    nclat = ncfile_sfc.createVariable('latitude', np.float32, ('time'))
    nclat.units = 'degrees_north'
    nclat.long_name = 'latitude'
    nclon = ncfile_sfc.createVariable('longitude', np.float32, ('time'))
    nclon.units = 'degrees_east'
    nclon.long_name = 'longitude'
    nctime = ncfile_sfc.createVariable('time', np.float64, ('time'))
    nctime.units = 'seconds since 1970-01-01'
    nctime.long_name = 'time'
    # Define a 3D variable to hold the data
    ncu = ncfile_sfc.createVariable('u_wind',np.float64,('time')) # note: unlimited dimension is leftmost
    ncu.units = 'm s-1' 
    ncu.standard_name = 'eastward_wind' # this is a CF standard name
    ncu.long_name = 'U component of the predicted surface wind'
    ncv = ncfile_sfc.createVariable('v_wind',np.float64,('time')) # note: unlimited dimension is leftmost
    ncv.units = 'm s-1' 
    ncv.standard_name = 'northward_wind' # this is a CF standard name
    ncv.long_name = 'V component of the predicted surface wind'

    # save data to arrays 
    nclat[:] = hdobs.lat.values
    nclon[:] = hdobs.lon.values
    nctime[:] = (hdobs.dt  - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
    ncu[:] = u_nc
    ncv[:] = v_nc
        
    ncfile_sfc.close()

def save_2d_netcdf(lat_nc, lon_nc, u_nc, v_nc, samurai_time, analysis_time, args):

    from netCDF4 import Dataset
    import numpy as np
    import pandas as pd

    # open file
    ncfile_sfc = Dataset('./nn_output/HOT_SAMURAI_sfc_analysis_'+args.STORM+'_'+analysis_time+'.nc',mode='w',format='NETCDF4') 

    # define dimensions
    # are these two-dimensional?? (could do a simple, x/y)
    y_dim = ncfile_sfc.createDimension('latitude', len(lat_nc))     # latitude axis
    x_dim = ncfile_sfc.createDimension('longitude', len(lon_nc))    # longitude axis
    time_dim = ncfile_sfc.createDimension('time', 1) # unlimited axis (can be appended to)

    # set up metadata
    ncfile_sfc.title='CSU Predicted Surface Wind'
    ncfile_sfc.subtitle="Generated using CSU SWANN"

    # set up variables
    nclat = ncfile_sfc.createVariable('latitude', np.float32, ('latitude'))
    nclat.units = 'degrees_north'
    nclat.long_name = 'latitude'
    nclon = ncfile_sfc.createVariable('longitude', np.float32, ('longitude'))
    nclon.units = 'degrees_east'
    nclon.long_name = 'longitude'
    nctime = ncfile_sfc.createVariable('time', np.float64, ('time',))
    nctime.units = 'seconds since 1970-01-01'
    nctime.long_name = 'time'
    # Define a 3D variable to hold the data
    ncu = ncfile_sfc.createVariable('u_wind',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
    ncu.units = 'm s-1' 
    ncu.standard_name = 'eastward_wind' # this is a CF standard name
    ncu.long_name = 'U component of the predicted surface wind'
    ncv = ncfile_sfc.createVariable('v_wind',np.float64,('time','latitude','longitude')) # note: unlimited dimension is leftmost
    ncv.units = 'm s-1' 
    ncv.standard_name = 'northward_wind' # this is a CF standard name
    ncv.long_name = 'V component of the predicted surface wind'

    # save data to arrays and reshape data into 2-D array
    nclat[:] = lat_nc # (MAYBE?!) 
    nclon[:] = lon_nc # (MAYBE?!)
    nctime[:] = (samurai_time - pd.Timestamp("1970-01-01", tz='UTC')) // pd.Timedelta('1s')
    ncu[:,:,:] = u_nc[np.newaxis,:,:] # check dimensions
    ncv[:,:,:] = v_nc[np.newaxis,:,:] # check dimensions

    ncfile_sfc.close()

def plot_image_2pan(x_plane, y_plane, sfc_wind_pred, hdobs,
               radii_vals_str, radii_vals, echo_edges, textstr, vmax_table, figtitle, args, imDir, samurai_time):

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.font_manager import FontProperties
    import numpy as np
    plt.rcParams.update({'mathtext.default':  'regular' })

    vmax_col_labels = ['HDOBS\nVmax (kt)']
    vmax_row_labels = ['FL','SWANN']

    radii_col_labels = ['NE','SE','SW','NW']
    radii_row_labels = ['R34','R50','R64']

    fig = plt.figure(figsize=(8.5,3.5))
    gs = fig.add_gridspec(1,3)
    f_ax4 = fig.add_subplot(gs[0, :-1])
    f_ax5 = fig.add_subplot(gs[0, -1])
    f_ax4.plot(hdobs.dt, hdobs.sfmr, 'k',hdobs.dt, hdobs.wsp, 'r')
    f_ax4.plot(hdobs.dt, sfc_wind_pred*1.94, color='#1E4D2B')
    f_ax4.plot(hdobs.dt.values[0], hdobs.wsp.values[0], 'kx') # flight start
    f_ax4.plot(hdobs.dt.values[-1], hdobs.wsp.values[-1], 'ko') # flight end
    axins = f_ax4.inset_axes(
        [0.02, 0.78, 0.15, 0.2], xticklabels=[], yticklabels=[])
    axins.plot(x_plane, y_plane,'r')
    axins.plot(x_plane[0], y_plane[0],'kx')
    axins.plot(x_plane[-1], y_plane[-1],'ko')
    axins.plot(0, 0,'k*')

    f_ax5.text(-0.075, 0.99, textstr, transform=f_ax5.transAxes, fontsize=10,verticalalignment='top')
    my_table = f_ax5.table(cellText=np.round(vmax_table,decimals=1), rowLabels=vmax_row_labels,
                        colLabels=vmax_col_labels, bbox=[0.15,0.3,0.4,0.375])
    for (row, col), cell in my_table.get_celld().items():
        if (row == 2):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.get_text().set_color('#1E4D2B')

    my_table2 = f_ax5.table(cellText=radii_vals_str, rowLabels=radii_row_labels,
                        colLabels=radii_col_labels, bbox=[0.15,-0.025,0.8,0.3])

    for (row, col), cell in my_table2.get_celld().items():
        if (row == 0) | (col == -1):
            continue
        if ((radii_vals[row-1,col]/echo_edges[col]) > 0.95):
            cell.set_text_props(fontproperties=FontProperties(style='italic',weight='ultralight'))
            cell.get_text().set_color('red')
    f_ax5.set_axis_off()
    f_ax4.legend(['SFMR','FL','SWANN'], loc='lower right')
    f_ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    f_ax4.grid(True)
    f_ax4.set_ylabel('wind speed (kt)')
    plt.suptitle(figtitle,y=0.94)
    fig.savefig(imDir+args.STORM+'_'+samurai_time.strftime(format='%Y%m%d%H%M')+'_2pan.png', dpi=200, bbox_inches='tight')


def plot_image_4pan(x_plot, y_plot, rd, x_plane, y_plane, sfc_wind_pred, mag_3km, sfc_wind_pred_ac, hdobs, swann_rmw,
               radii_vals_str, radii_vals, echo_edges, textstr, vmax_table, figtitle, args, imDir, analysis_time):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    import matplotlib.colors as colors
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    import numpy as np
    plt.rcParams.update({'mathtext.default':  'regular' })

    vmax_col_labels = ['HDOBS\nVmax (kt)','SAMURAI\nVmax (kt)']
    vmax_row_labels = ['FL','SWANN']

    radii_col_labels = ['NE','SE','SW','NW']
    radii_row_labels = ['R34','R50','R64']

    colors1 = plt.cm.Blues(np.linspace(0.2, 0.8, 7,endpoint=False)+0.5/7.)
    colors2 = plt.cm.Greens(np.linspace(0.2, 0.8, 8,endpoint=False)+0.5/8.)
    colors3 = plt.cm.YlOrRd(np.linspace(0.0, 0.25, 7,endpoint=False)+0.5/7.)
    colors4 = plt.cm.Reds(np.linspace(0.5, 0.8, 8,endpoint=False)+0.5/8.)
    colors5 = plt.cm.RdPu(np.linspace(0.3, 0.9, 14,endpoint=False)+0.5/14.)

    # combine them and build a new colormap
    cs = np.vstack((colors1, colors2, colors3, colors4, colors5))

    bounds = np.hstack((np.arange(20,34,2),np.arange(34,50,2),np.arange(50,64,2),np.arange(64,96,4),np.arange(96,200,8)))
    norm = colors.BoundaryNorm(boundaries=bounds,ncolors=len(bounds))
    spd_ticks = [20,34,50,64,83,96,113,137]

    mymap = colors.ListedColormap(cs)

    line = Line2D([0], [0], label='RMW', color='k', linestyle='--')

    fig = plt.figure(figsize=(8.5,7))
    gs = fig.add_gridspec(3,3,height_ratios=[1.0,0.05,1.0])
    f_ax1 = fig.add_subplot(gs[0, 0])
    f_ax2 = fig.add_subplot(gs[0, 1])
    f_ax3 = fig.add_subplot(gs[0, 2])
    f_ax4 = fig.add_subplot(gs[2, :-1])
    f_ax5 = fig.add_subplot(gs[2, -1])
    c1 = f_ax1.contourf(x_plot/1.852, y_plot/1.852, sfc_wind_pred*1.94, levels=bounds, norm=norm, cmap=mymap, extend='max');
    c2 = f_ax2.contourf(x_plot/1.852, y_plot/1.852, mag_3km*1.94, levels=bounds, norm=norm, cmap=mymap, extend='max');
    t1 = f_ax1.contour(x_plot/1.852, y_plot/1.852, sfc_wind_pred*1.94,colors=['k','k','k'],
                    linewidths=[0.35,0.7,1.15], levels=[83,113,137])
    t2 = f_ax2.contour(x_plot/1.852, y_plot/1.852, mag_3km*1.94,colors=['k','k','k'],
                    linewidths=[0.35,0.7,1.15], levels=[83,113,137])
    ln2, = f_ax2.plot(x_plane/1.852,y_plane/1.852,'k')
    f_ax2.legend([ln2],['flight path'])
    f_ax2.plot(x_plane[0]/1.852,y_plane[0]/1.852,'kx') # flight start
    f_ax2.plot(x_plane[-1]/1.852,y_plane[-1]/1.852,'ko') # flight end
    c3 = f_ax3.contourf(x_plot/1.852, y_plot/1.852, sfc_wind_pred/mag_3km, levels=np.arange(0.75,1.05,0.05), cmap='coolwarm', extend='both')
    f_ax3.contour(x_plot/1.852, y_plot/1.852, rd/1.852, levels=np.array([swann_rmw/1.852]), colors='k', linestyles='dotted');
    f_ax3.legend([line],['RMW'])
    f_ax1.set_aspect('equal')
    f_ax2.set_aspect('equal')
    f_ax3.set_aspect('equal')
    f_ax4.plot(hdobs.dt, hdobs.wsp, 'r')
    f_ax4.plot(hdobs.dt, sfc_wind_pred_ac*1.94, color='#1E4D2B')
    f_ax4.plot(hdobs.dt.values[0], hdobs.wsp.values[0], 'kx') # flight start
    f_ax4.plot(hdobs.dt.values[-1], hdobs.wsp.values[-1], 'ko') # flight end
    #f_ax4.plot(hdobs.dt, hdobs.sfmr, 'k',hdobs.dt, hdobs.wsp, 'r')
    f_ax5.text(-0.075, 0.99, textstr, transform=f_ax5.transAxes, fontsize=10,verticalalignment='top')
    my_table = f_ax5.table(cellText=np.round(vmax_table,decimals=1), 
                        rowLabels=vmax_row_labels, colLabels=vmax_col_labels,
                        bbox=[0.15,0.3,0.8,0.375])
    for (row, col), cell in my_table.get_celld().items():
        if (row == 2):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.get_text().set_color('#1E4D2B')

    my_table2 = f_ax5.table(cellText=radii_vals_str, # convert radii from km to nm
                        rowLabels=radii_row_labels, colLabels=radii_col_labels,
                        bbox=[0.15,-0.025,0.8,0.3])

    for (row, col), cell in my_table2.get_celld().items():
        if (row == 0) | (col == -1):
            continue
        if ((radii_vals[row-1,col]/echo_edges[col]) > 0.95):
            cell.set_text_props(fontproperties=FontProperties(style='italic',weight='ultralight'))
            cell.get_text().set_color('red')

    f_ax5.set_axis_off()
    f_ax4.legend(['HDOBS FL','HDOBS SWANN'])
    #f_ax4.legend(['SFMR (kt)','FL (kt)'])
    f_ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    f_ax1.set_xlim([-85,85]); f_ax1.set_ylim([-85,85])
    f_ax2.set_xlim([-85,85]); f_ax2.set_ylim([-85,85])
    f_ax3.set_xlim([-85,85]); f_ax3.set_ylim([-85,85])
    f_ax1.set_xticks([-80, -40, 0, 40, 80]); f_ax1.set_yticks([-80, -40, 0, 40, 80]);
    f_ax2.set_xticks([-80, -40, 0, 40, 80]); f_ax2.set_yticks([-80, -40, 0, 40, 80]);
    f_ax2.set_yticklabels([])
    f_ax3.set_xticks([-80, -40, 0, 40, 80]); f_ax3.set_yticks([-80, -40, 0, 40, 80]);
    f_ax3.set_yticklabels([])
    f_ax4.grid(True)
    f_ax1.xaxis.set_minor_locator(AutoMinorLocator()); f_ax1.yaxis.set_minor_locator(AutoMinorLocator())
    f_ax2.xaxis.set_minor_locator(AutoMinorLocator()); f_ax2.yaxis.set_minor_locator(AutoMinorLocator())
    f_ax3.xaxis.set_minor_locator(AutoMinorLocator()); f_ax3.yaxis.set_minor_locator(AutoMinorLocator())
    f_ax1.set_xlabel('distance from center (nm)'); f_ax1.set_ylabel('distance from center (nm)');
    f_ax2.set_xlabel('distance from center (nm)');
    f_ax3.set_xlabel('distance from center (nm)');
    f_ax1.set_title('SWANN SFC wind (kt)');
    f_ax2.set_title('SAMURAI FL wind (kt)');
    f_ax3.set_title('ratio: SFC/FL');
    f_ax4.set_ylabel('wind speed (kt)');
    plt.suptitle(figtitle,y=0.915)
    cb1 = plt.colorbar(mappable=c1,cax=fig.add_subplot(gs[1,:2]), orientation='horizontal',ticks=spd_ticks)
    cb1.ax.set_title('');
    cb1.add_lines(t1)
    cb3 = plt.colorbar(mappable=c3,cax=fig.add_subplot(gs[1,2]), orientation='horizontal', ticks=[0.75, 0.85, 0.95, 1.05])
    cb3.ax.set_title('');
    fig.savefig(imDir+args.STORM+'_'+analysis_time+'_4pan.png', dpi=200, bbox_inches='tight')

