#/* cSpell:disable */
# see requirements.txt

#%% [markdown]
# NAIP imagery comes as mrSID.
# Before running this convert to tif. Download the lizardtech converter from https://www.extensis.com/support/developers. Then download newist sdk. Extract it then  ```cd MrSID_DSDK-9.5.4.4709-rhel6.x86-64.gcc531/Raster_DSDK/bin/```. Then ```./mrsiddecode -wf -i infile.sid -o outfile.tif```

#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import xarray as xr

# %%
filepath = '/media/data/UrbanTree/NAIP'
infile = os.path.join(filepath, 'ortho_1-1_hn_s_ca037_2018_1.tif')
outfile = os.path.join(filepath, 'ortho_1-1_hn_s_ca037_2018_1_ndvi.tif')

#%% [markdown]
# Fin out information about the geotiff.

#%%
with rio.open(infile) as dataset:
    width, height = dataset.shape
    count = dataset.count
    desc = dataset.descriptions
    metadata = dataset.meta
    driver = dataset.driver # driver used to open
    proj = dataset.crs
    gt = dataset.transform
    dtype =  metadata['dtype']
    

#%%
with rio.open(outfile, 'w', width=width, height=height, dtype=dtype, count=count, driver='GTiff', crs=proj) as dst:
    with rio.open(infile) as src:
        for i, window in src.block_windows(1):
            naip_data = src.read(window=window)
            naip_ndvi = es.normalized_diff(naip_data[2], naip_data[0]) # 
            naip_ndvi = naip_ndvi.astype(dtype) #TODO: will be three in real [3] - [2]
            dst.write(naip_ndvi, window=window, indexes=1)
# %%

# %%
