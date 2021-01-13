#!/bin/bash
# called by binfolaz to construct json pipelines for pdal
# FNAME= file name

FNAME="$1"

# make filenames
BASE=(${FNAME//./ })
DSM="${BASE}_dsm.tif"
DTM="${BASE}_dtm.tif"
CHM="${BASE}_chm.tif"
NRM="${BASE}_normed.las"

# define smrf parameters.
# TODO: are these good params?
SLOPE=0.1
WINDOW=5

# determine bounds of infile
BOUNDS=$(pdal info $FNAME | jq '.stats.bbox.native.bbox |[[.minx, .maxx], [.miny, .maxy]]' | sed '0,/\[/{s/\[/\(/}' | sed ':a;N;$!ba;s/\n//g' | sed 's/\(.*\)\]/\1\)/' | tr -d ' ')

# make the pipeline
JSONFILE="{
      \"pipeline\":[
        \"$FNAME\",
        {
          \"type\":\"filters.assign\",
        \"assignment\" : \"Classification[:]=0\"
        },
        {
          \"type\":\"filters.elm\",
          \"threshold\":2.0
        },
        {
          \"type\":\"filters.outlier\"
        },
        {
          \"type\":\"filters.estimaterank\",
          \"knn\":8,
          \"thresh\":0.01
        },
        {
          \"type\":\"filters.smrf\",
          \"ignore\":\"Classification[7:7]\",
          \"slope\":\"$SLOPE\",
          \"window\":\"$WINDOW\",
          \"cell\":1,
          \"returns\": \"first,last,only\",
          \"threshold\":0.45,
          \"scalar\":1.2
        },
        {
          \"type\": \"writers.gdal\",
          \"resolution\": 1,
          \"gdaldriver\":\"GTiff\",
          \"output_type\": \"idw\",
          \"radius\": 6,
          \"data_type\": \"float32\",
          \"bounds\": \"$BOUNDS\",
          \"filename\":\"$DSM\"
        },
        {
          \"type\": \"writers.las\",
          \"compression\": \"laszip\",
          \"scale_x\": \"0.01\",
          \"scale_y\": \"0.01\",
          \"scale_z\": \"0.01\",
          \"offset_x\": \"auto\",
          \"offset_y\": \"auto\",
          \"offset_z\": \"auto\",
          \"filename\": \"$NRM\"
        },
        {
        \"type\":\"filters.range\",
        \"limits\":\"Classification[2:2]\"
        },
        {
          \"type\": \"writers.gdal\",
          \"resolution\": 1,
          \"gdaldriver\":\"GTiff\",
          \"output_type\": \"min\",
          \"output_type\": \"idw\",
          \"radius\": 6,
          \"data_type\": \"float32\",
          \"bounds\": \"$BOUNDS\",
          \"filename\":\"$DTM\"
        }
      ]
    }"

echo $JSONFILE > ${BASE}.json

