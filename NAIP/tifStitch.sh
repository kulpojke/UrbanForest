#!/bin/bash

DIR="$1"

gdalbuildvrt ${DIR}/mosaic.vrt ${DIR}/*.tif

gdal_translate -of GTiff -co "COMPRESS=LZW"  -co "TILED=YES" -co  "BIGTIFF=YES" ${DIR}/mosaic.vrt ${DIR}/mosaic.tif
