import glob
import pandas as pd
import numpy as np
import os


def create_dataframe(inDir, start_time, end_time):

    if inDir.find('hrd_radials') != -1:
        ext = '[0-9].list.gz'
    elif inDir.find('hdobs') != -1:
        ext = '.txt'
    else:
        print("issue with inDir")

    files = sorted(glob.glob(inDir+'/'+str(start_time.year)+'/*'+ext, recursive=True))
    df = pd.DataFrame(files,columns=['path'])
 
    if inDir.find('hrd_radials') != -1:
        start = len(inDir) + 6
    else:
        start = df['path'][0].find('.'+str(start_time.year)) + 1
    
    df['datetime'] = pd.to_datetime(df['path'].str[start:start+12], format='%Y%m%d%H%M', utc=True)

    return df


def shrink_df(df, start_time, end_time):

    mask = (df['datetime'] >= (start_time - pd.Timedelta(10,unit='m'))) & (df['datetime'] <= (end_time + pd.Timedelta(10,unit='m')))
    df_sm = df.loc[mask]

    return df_sm


def copy_files(df, outdir):

    for i in np.arange(len(df)):
        os.system('cp '+df.path.iloc[i]+' '+outdir)

    print('copied over '+str(i+1)+' files to samurai_input')

    return


