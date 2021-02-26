import fiona
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd
import numpy as np


def clip_raster(image_path, shape_path, out_path, out_crs=None):
    '''clips image with shp'''

    # ensure files conform to the desired crs
    image_path, shape_path = checkFix_proj(image_path, shape_path, crs=out_crs)

    # read the shape
    with fiona.open(shape_path, 'r') as shp:
        geoms = [feature['geometry'] for feature in shp]

    # read the tif
    with rasterio.open(image_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
        out_meta = src.meta

    # write clipped file
    out_meta.update({'driver': 'GTiff', 'height': out_image.shape[1],
                     'width': out_image.shape[2], 'transform': out_transform})

    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(out_image)

    
def checkFix_proj(image_path, shape_path, crs=None):
    '''Checks to make sure the crs is the same for image and shape.
    If they are not the same, transforms shp to the image crs, or optionaly,
    ensures both are in specified crs.
    image_path -- str - path to tiff.
    shape_path -- str - path to shapefile.
    crs        -- str - crs to be used. If None, uses img crs.
    '''
    # housecleaning supplies
    shp_path, img_path = shape_path, image_path
    skip = False
    global del_dir
    del_dir = False
    
    # open the files and determine what crs to use
    shp = gpd.read_file(shape_path)
    with rasterio.open(image_path) as img:
        if crs == None:
            crs = img.crs
            crs = crs.to_string()
            skip = True

    # transform shp if necessary and write to file
    if shp.crs != crs:
        # make the path to write transformed img to
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
            del_dir = True
        shp_path = 'tmp/shp_tranformed.shp'
            
        # transform and write
        shp = shp.to_crs(crs)
        shp.to_file(filename=shp_path)
    # transform the image if necessary
    if not skip:

        # make the path to write transformed img to
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
            del_dir = True
        img_path = 'tmp/img_tranformed.tif'
        
        # transform and write img
        with rasterio.open(image_path) as img:
            transform, width, height = calculate_default_transform(img.crs, crs, img.width, img.height, *img.bounds)
            kwargs = img.meta.copy()
            kwargs.update({'crs' : crs, 'transform': transform, 'width': width, 'height': height})
            with rasterio.open(img_path, 'w', **kwargs) as dst:
                for band in range(1, img.count + 1):
                    reproject(
                        source=rasterio.band(img, band),
                        destination=rasterio.band(dst, band),
                        src_transform=img.transform,
                        src_crs=img.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest)
        
    return(img_path, shp_path)
