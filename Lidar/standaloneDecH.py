#/* cSpell:disable */
# runs in pycrown++ env
''' This is an implementaion of DecH, described in:

DecHPoints: A New Tool for Improving LiDAR Data Filtering in Urban Areas
Sellers, Chester Andrew; Cordero, Miguel; Miranda, David
ISSN: 2512-2789 , 2512-2819; DOI: 10.1007/s41064-019-00088-7
Journal of photogrammetry, remote sensing and geoinformation science. , 2020 '''

#%%
import pdal, sys
from string import Template
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import richdem as rd
import xarray as xr
import pandas as pd
import time
import dask.array as da
import matplotlib.pyplot as plt

clk = time.CLOCK_MONOTONIC

def timer_decor(func):
    def inner(*args, **kwargs):
        t0 =  time.clock_gettime(clk)
        func(*args, **kwargs)
        dt =  time.clock_gettime(clk) - t0
        dt = round(dt, 3)
        name = func.__name__
        print(f'It took {dt} seconds to run {name}')
    return(inner)

#%% [markdown] 
# We will use some files, but for now one file.

#%%
path = '/media/data/UrbanTree/6249/'
f = f'{path}USGS_LPC_CA_LosAngeles_2016_L4_6429_1820a_LAS_2018.laz'

#%% [markdown]  
# First we will use a PDAL (*CITE PDAL*) pipeline to load the
# pointcloud and do some preliminary filtering of outliers. After loading the
# file three filters are applied.  ```filters.elm``` which removes low value noise, ```filters.outlier``` which filter out noise in general, and ```filters.assign``` which we simply use to assign a classification of 0 to all points.

#%% # TODO: dtch elm and outliers, not needed
def make_pipe(path, file_name, slope=0.1, window=5):
  '''Returns pipeline for creating a DSM, DTM, and CHM'''
  base = file_name.rstrip('.laz')
  dsm = f'{base}_dsm.tif'
  dtm = f'{base}_dtm.tif'
  chm = f'{base}_chm.tif'
  las = f'{base}_norm.las'

  t = Template('''
  {
      "pipeline":[
        "${infile}",
        {
          "type":"filters.assign",
          "assignment" : "Classification[:]=0"
        }
      ]
    }''')

  pipe = t.substitute(infile=f, name_dsm=dsm, name_dtm=dtm, name_las=las, name_chm=chm, slope_=slope, window_=window)
  return(pipe)

pipe = make_pipe(path, f)
pipeline = pdal.Pipeline(pipe)
pipeline.validate()
count = pipeline.execute()
S = pipeline.arrays[0]
metadata = pipeline.metadata
log = pipeline.log
sh = S.shape
print(f'S is a numpy structured array of shape {sh}.')

# %% [markdown] 
# After running the pipeline we end up with a point cloud, ```S```, which is a 1D numpy structured array. Each entry in the array corresponds to a point in the pointcloud and is stored as a structured datatype which is a composition of simple datatypes.  In this case the fileds of the dtype are

#%%
S.dtype.descr
 
#%% [markdown]
# TODO:Unfortunately the X, Y and Z coordinates in the pointcloud here are given in feet.  So lets convert them to m.

#%% [markdown]
# Despite the outlier filters we used from PDAL there are some obvious outliers still in the data. First lets get rid of obvious outliers from the point cloud.
#%%
h, bins, patches = plt.hist(S['Z'], bins=1000)


#%% [markdown]
# The histogram plotted above shows that the values are gamma distributed-ish with some severe outliers going down to about -500. If we zoom in on the lower end of common values we see that there is a very sharp dropoff at somewhere around 36.

#%%
blw = len(S['Z'][S['Z']<36])
in36_37 = len(S['Z'][(S['Z']>36) & (S['Z']<37)])

plt.hist(S['Z'], bins=1000);
plt.xlim([35, 38]);
plt.title(f'There are {blw} points with Z below 36 and {in36_37} with Z between 36 and 37');

#%% [markdown]
# Let's make an occurance threshold based on the histograms. Points below the main mass of the histogram which fall into bins with a count smaller than the threshold will be culled. This is sensitive to choice of threshold and the bin size of the hist. Based on the above histogram (note the title) we will choose 2000 as the threshold, and stick with 1000 bins. 

#%%
thresh = 2000
cut = bins[np.argmax(h>thresh)]
S = S[S['Z']>=cut]
h, bins, patches = plt.hist(S['Z'], bins=1000)


# %% [markdown]
# Points which lie in the long tail of the histogram, i.e. high points, will be dropped by the next step in creating the DTM.
#
# ***DecHPoints***
#
# Next we will implement the DecHPoints algortihm which decimates the
# highest points in a region so that classsifcation of ground points will be more
# straightforward afterwards. 
#
# The inputs for DecHPoints are a point cloud (loaded above as ```S```), window size ```c``` and the maximum allowed residual ```δh```. The window size is the maximum size of an area without groudpoints, P$_{g}$, which should e something like the shortest side of the larget building (assuming rectangular buildings).  ```δh``` is the maximum vertical distance allowed between a point and the reference surface for it to be considered a ground point.

# %% In feet, yuck
c =60
δh = 1

#%% [markdown]
# The first step of DecH is to automatically calculate some variables. ```d``` is 
# the average point densityfor the extent of the cloud. Further explanation of variables can be found in Sellers in Fig 3. and section 3.1.

#%%
ns = S.shape[0] # number of points
d = ns / ((S['X'].max() - S['X'].min()) *  (S['Y'].max() - S['Y'].min()))
npoints = 4
resolution = np.sqrt(npoints / d)
# create initial DTM raster
buf = c // 3
width  = S['X'].max() - S['X'].min() + buf
height = S['Y'].max() - S['Y'].min() + buf
# numer of cells
N_x = int(np.ceil(width  / resolution))
N_y = int(np.ceil(height / resolution))
# offset to shift coordinates
x_offset = -S['X'].min() + buf/2
y_offset = -S['Y'].min() + buf/2
#craeate a new field 'grid' in S to store raster location 
new_dt = np.dtype(S.dtype.descr + [('gridX', '<u4'), ('gridY', '<u4'), ('resid', 'f4')])
new_arr = np.empty(S.shape, dtype=new_dt)
for desc in S.dtype.descr:
    new_arr[desc[0]] = S[desc[0]]
S = new_arr

#%%
# make an array full of high values 
pre_φdtm = np.full((N_x, N_y), 100000.0)#, chunks=(1000, 1000))
# locate each point on grid...

@np.vectorize
def grid_loc(x, y, z, x_offset, y_offset, resolution):
    # locate each point on grid and assign min value to each cell
    global pre_φdtm
    xi = int(np.ceil((x + x_offset) / resolution))
    yi = int(np.ceil((y + y_offset) / resolution))
    pre_φdtm[xi, yi] = min(pre_φdtm[xi, yi], z)
    return(xi, yi)

S['gridX'] , S['gridY'] = grid_loc(S['X'], S['Y'], S['Z'],  x_offset, y_offset, resolution)
# ... and assign min value to each cell
for row in S:
  pre_φdtm[row['gridX'], row['gridY']] = min(pre_φdtm[row['gridX'], row['gridY']], row['Z'])

plt.hist(pre_φdtm[pre_φdtm<100000], bins=50);


#%% [markdown]
# then the raster can be smoothed interpolating with a ```c```$\times$```c```
# window across the raster with each cell taking the lowest value from
# its neighbourhood. In this case we have used a 1/3 overlap of the
# interpolation window rather than the 50% overlap ussed by Selles et al.

#%%
def interp_min(arr, c):
    buf = int(c // 3)
    new_arr = np.full_like(arr, 100000.0)
    for i in range(buf, arr.shape[0] - buf):
        for j in range(buf, arr.shape[1] - buf):
            new_arr[i, j] = arr[i-buf:i+buf, j-buf:j+buf].min()
    return(new_arr)

pre_φdtm[pre_φdtm<0] = 100000.0
φdtm = interp_min(pre_φdtm, c)

#del pre_φdtm

φdtm[φdtm == 100000] = np.NaN
plt.imshow(φdtm, cmap='Greys') # displays better if first φdtm[φdtm == 100000] = np.NaN
plt.title(f'c = {c}')

#%% [markdown]
# Next we calculate the slope surface ```φslope```; max slope parameter, ```Slmin```; and min slope parameter,```Slmax```. In this case, Slmin
# and Slmax are established from the 65% and 90% quantiles of the cell
# values of the slope surface. We will use the handy TerrainAttribute function from RichDEM to find. ```φslope``'.

#%%
rda = rd.rdarray(φdtm, no_data=np.NaN)
φslope = rd.TerrainAttribute(rda, attrib='slope_riserun')
Slmin, Slmax = np.nanpercentile(φslope, [65, 90])

#%% [markdown]
# Then we classify points as ground if they are within δh of the reference surface given
# by  φdtm, and calculate the penetrability surfac, eφpnt.#%% [markdown]
# Then we classify points as ground if they are within δh of the reference surface given
# by  φdtm, and calculate the penetrability surfac, eφpnt.

#%%
# Classify as ground (2) if δi <= δh
S['Classification'] = (S['Z'] - φdtm[S['gridX'], S['gridY']] <= δh) * 2

# create φpnt
#φpnt = np.full_like(φdtm, np.NaN)
#x, y = φdtm.shape[0], φdtm.shape[1]
#for i in range(x):
#  print(f'Calculating row {i} of {x}')
#  for j in range(y):
#    points   = S[(S['gridX']==i) & (S['gridY']==j)].shape[0]
#    G_points = S[(S['gridX']==i) & (S['gridY']==j) & (S['Classification']==2)].shape[0]
#    if points > 0:
#      φpnt[i, j] = G_points / points


#%% Delayed version
#from dask import delayed, compute
#from dask.diagnostics import ProgressBar
#
#φpnt = np.full_like(φdtm, np.NaN)
#count = len(φdtm)
#
#def pnt(slice, S):
#  '''eats tuples of enumerated slices), e.g. (10, array([..]) )'''
#  global φpnt
#  i = slice[0]
#  for j in range(len(slice[1])):
#    points   = S[(S['gridX']==i) & (S['gridY']==j)].shape[0]
#    G_points = S[(S['gridX']==i) & (S['gridY']==j) & (S['Classification']==2)].shape[0]
#    if points > 0:
#      φpnt[i, j] = G_points / points


#skivor = [(i, φdtm[i]) for i in range(φdtm.shape[0])]
#slices = [delayed(pnt)(skiva, S) for skiva in skivor]
#with ProgressBar():
#  zults = compute(*slices)

#%% numba version
from numba import jit, stencil


@jit(nopython=True, parallel=True)
def inner_loop(slice, S, i):
  for j in range(len(slice)):
    query    = S[(S['gridX']==i) & (S['gridY']==j)]
    points   = query.shape[0]
    G_points = query['Classification'].sum() / 2
    if points > 0:
      slice[j] = G_points / points
  return(slice)


φpnt = np.full_like(φdtm, np.NaN)
xdim, ydim = φpnt.shape[0], φpnt.shape[1] 
t0 =  time.clock_gettime(clk)
for i in range(xdim):
  dt =  round(time.clock_gettime(clk) - t0, 1)
  print(f'On {i} of {xdim} at {dt} seconds')
  φpnt[i, :] = inner_loop(φpnt[i, :], S, i)



#%% [markdown]
# ***Decimating the Point Cloud***
#
# The next step is to decimate the point cloud.  First the highest points within
# the searh window will be selected, then within the window of analysis it will be
# determined if the selected points are gnon-ground.



#%% 
L = 1
O = 0.5
while L <= 2:
  if L ==1:
    H = c
  else:
    H = 0.75 * c
  # Define search and analysis windows, as well as theoretical points/search window
  Wa = H
  Ws = (1.5 * H)**2
  NSs = d * Ws
  # In the paper they use blc, trc : here I use tlc, brc
  buf = Ws / 2
  regions = []
  for x in range(0, np.nanmax(S['gridX']), int(Ws * O)):
    for y in range(0, np.nanmax(S['gridY']), int(Ws * O)):
      regions.append([x - buf, y - buf , x + buf , y + buf])
  for region in regions:
    Ss = S[(S['gridX']>region[0]) & (S['gridX']<region[2]) & (S['gridY']>region[1]) & (S['gridY']<region[3])]
    if len(Ss) >= 0.9 * NSs:
      pass
  L=10

# %%
