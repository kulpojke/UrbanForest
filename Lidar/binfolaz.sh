#!/bin/bash
# batch info for laz files, is wgere the name came from, but no it does much more:
# finds the info; makes pipelines; runs pipelines to create chm, dsm, dtm; 
# reprojects to EPSG4326; makes a big ol' mosaic.
# DIR=path to infiles

PARAMS=""
while (( "$#" )); do
  case "$1" in
  -h | --help)
    echo "SYNOPSIS"  
    echo "     binfozip [DIR]"
    echo "DESCRIPTION"
    echo "     Runs pdal info to find original extent of files,"
    echo "     then builds pdal pipelines to make DTM, DSM, CHM,"
    echo "     and normed las, saves pipeline as json. Finaly runs"
    echo "     pdal pipeline on each json using GNU parallel then delets json files."
    echo "     Mandatory argument, DIR is the path to directory holding the .laz files"
    echo "     Don't include / on the end of the path to directory"
    shift
    ;;
  *) # preserve positional arguments
    PARAMS="$PARAMS $1"
    shift
  esac
done

eval set -- "$PARAMS"

DIR="$1"

echo "Making pipelines in ${DIR}"

ls ${DIR}/*.laz | parallel -j+0 --eta 'bash binf.sh ${DIR}{}' 

echo "Running pipelines" 
ls ${DIR}/*.json | parallel -j+0 --eta 'pdal pipeline {}'

rm ${DIR}/*.json

echo "Reprojecting tifs"

ls ${DIR}/*.tif | parallel -j+0 --eta 'gdalwarp -t_srs EPSG:4326 -tr 0.000001 0.000001 {} {.}_EPSG4326.tif '