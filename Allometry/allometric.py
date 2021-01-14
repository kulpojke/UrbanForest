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
# Prepare dataset for use in testing, get allo equations

# read the file, drop rows with  NA in the DBH measurements 
path = '/media/data/UrbanTree/TreeData/SanDiegoCounty.csv'
# read the csv
df = pd.read_csv(path, usecols=['ID', 'LATITUDE', 'LONGITUDE',
                                  'DBH_LO', 'DBH_HI', 'CREATED',
                                  'UPDATED', 'SOURCE', 'Name_matched'])

# drop NAs
df.dropna(how='any', axis=0, subset=['DBH_LO', 'DBH_HI', 'CREATED', 'UPDATED'], inplace=True)

# Allometric equations from :
#
# McPherson, E. Gregory; van Doorn, Natalie S.; Peper, Paula J. 2016. Urban tree database.
# Fort Collins, CO: Forest Service Research Data Archive. Updated 21 January 2020.
# https://doi.org/10.2737/RDS-2016-0005
#
# 'Apps min' and 'Apps max' give the input range (cm) that the authors feel 
#  that the equations are reliable
# 'InlEmp' and 'SoCalC' are Climate zones where the eqs are different.
#  SoCalC reference city is Santa Monica, InlEmp is Claremont,
#  see Table 1, p16 for further Climate zone details

allo_df = pd.read_csv('/media/data/UrbanTree/Allometric/Data/TS6_Growth_coefficients.csv',
usecols=['Region', 'Scientific Name', 'Independent variable', 'Predicts component ', 'EqName', 'Units of predicted components',
'EqName', 'a', 'b', 'c', 'd', 'e', 'Apps min', 'Apps max'])
 

def mcpherson_eqs():
    '''returns dict of equations from table 3 (p24) of McPherson 2020
    functions use np so as to be vectorized'''

    eq_dict = {'lin'        : (lambda a, b, c, d, e, x, mse: a + b * (x)), 
                'quad'      : (lambda a, b, c, d, e, x, mse: a + b * x + c * x**2),
                'cub'      : (lambda a, b, c, d, e, x, mse: a + b * x + c * x**2 + d * x**3),
                'quart'     : (lambda a, b, c, d, e, x, mse:a + b * x + c *x**2 + d * x**3 + e * x**4), 
                'loglogw1' : (lambda a, b, c, d, e, x, mse: np.exp(a + b * np.log(np.log(x + 1) + (mse/2)))),
                'loglogw2' : (lambda a, b, c, d, e, x, mse: np.exp(a + b * np.log(np.log(x + 1)) + (np.sqrt(x) + (mse/2)))),
                'loglogw3' : (lambda a, b, c, d, e, x, mse: np.exp(a + b * np.log(np.log(x + 1)) + (x) + (mse/2))),
                'loglogw4' : (lambda a, b, c, d, e, x, mse: np.exp(a + b * np.log(np.log(x + 1)) + (x**2) + (mse/2))),
                'expow1'    : (lambda a, b, c, d, e, x, mse: np.exp(a+ b * (x) + (mse/2))),
                'expow2'    : (lambda a, b, c, d, e, x, mse: np.exp(a + b * (x) + np.sqrt(x) + (mse/2))),
                'expow3'    : (lambda a, b, c, d, e, x, mse: np.exp(a + b * (x) + (x) + (mse/2))),
                'expow4'    : (lambda a, b, c, d, e, x, mse: np.exp(a + b * (x) + (x**2) + (mse/2)))}

    return(eq_dict)

eq_dict = mcpherson_eqs()
#%%
# capitalize names
df['Name_matched'] = df.Name_matched.str.capitalize()

#%%
# Find all the trees with over 100 occurances 
trees = df.Name_matched.value_counts()
trees = list(trees.where(trees > 100).dropna().index)

#%%
# drop trees we do not have equations for
ns = allo_df['Scientific Name'].unique()
scrap = []
for n in trees:
    if n not in ns:
        scrap.append(n)
trees = [s for s in trees if s not in scrap]
df = df.loc[df.Name_matched.isin(trees)]

#%%
# converts inch hi, low into cm hi and low floats
@np.vectorize
def convert(x, y): 
    return(x * 2.54, y * 2.54)

df['dbh_low'], df['dbh_high'] = convert(df.DBH_LO, df.DBH_HI)
df.drop(['DBH_LO', 'DBH_HI'], axis=1, inplace=True)

#%%
# Change date fields to dateTime type
df['CREATED'] = pd.to_datetime(df.CREATED)
df['UPDATED'] = pd.to_datetime(df.UPDATED)
df.info()

# %%
df.head()
# %%
