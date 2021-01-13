#/* cSpell:disable */
# runs in conda env: py3-env1

#%%
import pandas as pd
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

#%%
path = '/media/data/UrbanTree/TreeData/all_clean_LAcounty_sunset.hdf'
#%%

def whatever():


    # Read the file, drop rows with  NA in the DBH measurements
    path = '/media/data/UrbanTree/TreeData/all_clean_LAcounty_sunset.hdf'
    # read the hdf
    la = pd.read_hdf(path, key='data')
    # select desired columns
    cols=['ID', 'LATITUDE', 'LONGITUDE', 'DBH_LO', 'DBH_HI', 'CREATED',
        'UPDATED', 'SOURCE', 'Name_matched', 'Zone']
    la = la[cols]
    # drop NAs
    la.dropna(how='any', axis=0, subset=['DBH_LO', 'DBH_HI'], inplace=True)