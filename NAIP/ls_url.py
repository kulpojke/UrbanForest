#/* cSpell:disable */
#%%
import urllib.request
import pandas as pd

url = 'ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/NED/LPC/projects/USGS_LPC_CA_LosAngeles_2016_LAS_2018/laz/'

#%%
def ls_url(url, outpath):
    '''Makes a list of files found at url and writes it to file.
    '''
    contents = urllib.request.urlopen(url).read().splitlines()
    filez = []
    for l in contents:
        filez.append(url + str(l).rsplit(' ')[-1].rstrip('\''))
    df = pd.DataFrame()
    df['file'] = filez
    df.to_csv(outpath,sep=' ', header=False, index=False)


