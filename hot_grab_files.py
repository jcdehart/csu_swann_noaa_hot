import glob
import pandas as pd
import numpy as np
import os
import subprocess


def create_dataframe(inDir, start_time, end_time):

    if inDir.find('hrd_radials') != -1:
        ext = '[0-9].list.gz'
    elif inDir.find('hdobs') != -1:
        ext = '[0-9].txt'
    else:
        print("issue with inDir")

    files = sorted(glob.glob(inDir+'/'+str(start_time.year)+'/*'+ext, recursive=True))
    df_orig = pd.DataFrame(files,columns=['path'])
 
    if inDir.find('hrd_radials') != -1:
        start = len(inDir) + 6
    elif inDir.find('hdobs') != -1:
        start = df_orig['path'][0].find('.'+str(start_time.year)) + 1

    df_orig['datetime'] = pd.to_datetime(df_orig['path'].str[start:start+12], format='%Y%m%d%H%M', utc=True)
    
    return df_orig


def shrink_df(df, start_time, end_time, storm_name, mission_code, af):

    # reducing to a 1-min buffer for now
    #mask = (df['datetime'] >= (start_time - pd.Timedelta(5,unit='m'))) & (df['datetime'] <= (end_time + pd.Timedelta(5,unit='m')))
    mask = (df['datetime'] >= (start_time - pd.Timedelta(10,unit='m'))) & (df['datetime'] <= (end_time + pd.Timedelta(10,unit='m'))) # with 10-min buffer, removing for now
    df_sm = df.loc[mask].reset_index(drop=True)
    print(start_time)
    print(end_time)
        
    if df_sm.path.str.contains('hdobs').iloc[0]:
        # basically get identifying info about each hdob file
        hdob_header = []
    
        print(len(df_sm.path))
        # loop through path and grab header
        for i in df_sm.path:
            test = subprocess.Popen('grep HDOB '+i,shell=True, stdout=subprocess.PIPE,close_fds=False).stdout #no clue what this means haha
            hdob_header.append(test.read().decode().rstrip().split()) # add to list and separate by whitespace
            #print(i)

        # only retain paths that match the storm name passed to function
        df_sm = df_sm.join(pd.DataFrame(hdob_header,columns=['plane','mission','storm','hdob','num','date'],dtype=str))

        if mission_code is not None:
            df_storm = df_sm[(df_sm.storm == storm_name) & (df_sm.mission == mission_code)]
        else:
            df_storm = df_sm[(df_sm.storm == storm_name)]

        # check to see if storm hasn't been named yet
        if df_storm.shape[0] == 0:
            print('data frame empty, testing TDR flag for unnamed storm')
            df_storm = df_sm[df_sm.storm == 'TDR']

        
        # depending on plane, restrict file list to relevant plane
        AF_mask = df_storm.plane.str.contains('AF')
    
        if af:
            df_plane = df_storm.loc[AF_mask]
        else:
            df_plane = df_storm.loc[~AF_mask]

    else:
        df_plane = df_sm
        print('not hdobs, returning dataframe')



    return df_plane


def copy_files(df, outdir):

    for i in np.arange(len(df)):
        os.system('cp '+df.path.iloc[i]+' '+outdir)

    print('copied over '+str(i+1)+' files to samurai_input')

    return


