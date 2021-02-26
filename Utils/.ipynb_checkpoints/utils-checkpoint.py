import fiona
import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd
import numpy as np


def checkFix_proj(image_path, shape_path, crs=None):
    '''Checks to make sure the crs is the same for image and shape.
    If they are not the same, transforms shp to the image crs, or optionaly,
    ensures both are in specified crs.
    image_path -- str - path to tiff.
    shape_path -- str - path to shapefile.
    crs        -- str - crs to be used. If None, uses img crs.
    '''
    # open the files
    shp = gdp.read_file(shape_path)
    with rasterio.open(image_path) as img:
        if crs == None:
            crs = img.crs
            print(crs)
            
    
    