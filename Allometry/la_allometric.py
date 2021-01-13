

#/* cSpell:disable */
# runs in conda env: py3-env1

#%%
import pandas as pd
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection

#%%
# read the file, drop rows with  NA in the DBH measurements 
path = '/media/data/UrbanTree/TreeData/LA_Trees_Clean_05012020.csv'
# read the csv
la = pd.read_csv(path, usecols=['ID', 'LATITUDE', 'LONGITUDE',
                                'DBH_LO', 'DBH_HI', 'CREATED',
                                'UPDATED', 'SOURCE', 'Name_matched',
                                'HEIGHT_LO', 'HEIGHT_HI', 'STEMS',
                                'TRUNKS', 'CANOPY', 'zip_code',
                                'place_name', 'NAME', 'CityType',
                                'Pop2010', 'Land_sqmi', 'DateIncorp',
                                'zcta', 'tot_pop', 'GEOID', 'HD_NAME',
                                'SPA_NAME', 'HD_2012', 'SPA_2012'])
# filter it to only Los Angeles
la = la[la.NAME == 'Los Angeles']
# drop NAs
la.dropna(how='any', axis=0, subset=['DBH_LO', 'DBH_HI'], inplace=True)
#%% [markdown]
# Lets see what the DBH bins in the data set look like.

#%%
@np.vectorize
def temp_range(low, high):
    l = str(low)
    h = str(high)
    return(f'{l}-{h}')

la['temp_range'] = temp_range(la.DBH_LO, la.DBH_HI)

la.temp_range.value_counts()

#%% [markdown]
# From the print out above we can see that there are some odd ranges of DBH that do not fit well with the others.  We will standardize the bins at six inch intervals.  Some of the bins are already fitted to this interval (0-6, 6-12, 12-18, 18-24, 24-30, 30-36, 36-42) some will fall entirely within the intervals (0-3, 3-6, 7-12, 13-18, 19-24, 25-30) and some will span the intervals(7-18, 12-24, 13-30). For each entry we will sample a uniform distribution on the given range to assign a point.  This will assign each tree a size within the proper range.  We will then use the point for assigning new bins. The point may be useful for ploting later.

#%%
la.drop('temp_range', axis=1, inplace=True) #housekeeping
la['sim_dbh'] = np.random.uniform(low=la.DBH_LO, high=la.DBH_HI)

def get_range(x):
    ages = {'0-6' : [0,6], '6-12' : [6,12], '12-18' : [12, 18], '18-24' : [18, 24], '24-30' : [24, 30], '30-36' : [30, 36], '36-42' : [36, 42] }
    for key, val in ages.items():
        lo, hi = val
        if lo < x <= hi:
            return(key)

la['dbh_range'] = [get_range(x) for x in la.sim_dbh.values]

#%% [markdown]
# Now lets llok at the DBH distribution of all the trees in the dataset. 

#%%
fig_path = '/media/data/UrbanTree/Figures/'

count = la[['ID', 'dbh_range']].groupby(by=['dbh_range']).count()
count.columns = ['Population']
count['DBH range'] = count.index

br_plt = sns.barplot(x='Population' ,y='DBH range', color='seagreen', data=count, order = ['36-42', '30-36', '24-30', '18-24', '12-18', '6-12', '0-6'])
_ = br_plt.set(ylabel='DBH range (inches)', title = 'All Species')
plt.savefig('{}hist_all.pdf'.format(fig_path))

#%% [markdown]
# Now to make a figure showint the dbh distribution of the top ten species.


#%%

species_list = list(la.Name_matched.unique())

top_ten = list(la.Name_matched.value_counts().head(10).index)
top_ten.sort()
t10 = la.loc[la.Name_matched.isin(top_ten)][['Name_matched', 'dbh_range']]

t10['count'] = 1
t10 = t10.pivot_table(index=['Name_matched', 'dbh_range'], values='count', aggfunc=np.sum)
t10.reset_index(inplace=True)
#t10 = t10.pivot(index='Name_matched', columns='dbh_range', values='count')
#t10 = t10.transpose()

#sns.relplot(x=t10.index, y=t10.columns, row_order=top_ten, col_order=['0-6', '6-12', '12-18', '18-24', '24-30', '30-36', '36-42'], data=t10, sizes=(40, 400))
t10['padd'] = t10['count'] / 10000
fig = plt.figure(figsize=(10.5, 8))
x = sorted(t10.Name_matched.unique())
y = ['0-6', '6-12', '12-18', '18-24', '24-30', '30-36', '36-42', ' ', 'blank2', 'blank3']
s = plt.scatter(x, y, s = 0)
s.remove

for row in t10.itertuples():
    bbox_props = dict(boxstyle='circle,pad={}'.format(row.padd), fc='g', alpha=0.5,  ec='w', lw=2)
    plt.annotate(str(row.count), xy = (row.Name_matched, row.dbh_range), bbox=bbox_props, ha='center', va='center', zorder = 2, clip_on = True)

fig.autofmt_xdate()
plt.ylim(top=' ')
plt.ylabel('DBH range (inches)')
plt.xlabel('Species')
plt.tight_layout()
plt.savefig('{}top_ten.pdf'.format(fig_path))
#ax = sns.heatmap(t10, cbar=False, cmap='binary')
#ax.invert_yaxis()


#%% [markdown]
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

#%%
allo_df = pd.read_csv('/media/data/UrbanTree/Allometric/Data/TS6_Growth_coefficients.csv',
usecols=['Region', 'Scientific Name', 'Independent variable', 'Predicts component ', 'EqName', 'Units of predicted components',
'EqName', 'a', 'b', 'c', 'd', 'e', 'Apps min', 'Apps max'])

allo_df = allo_df.loc[allo_df.Region.isin(['InlEmp', 'InlVal', 'NoCalC', 'SoCalC'])]
allo_df = allo_df.loc[allo_df['Predicts component '] == 'age']  

#%%  [markdown]
# dict of equations from table 3 (p24) of McPherson 2020
# functions use np so as to be vectorized

#%%
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

#%% [markdown]
# Find all the trees with over 100 occurances, determine which ones we have equations for and convert DBH values to centimeters.

#%%
trees = la.Name_matched.value_counts()
trees = list(trees.where(trees > 100).dropna().index)
# capitalise the genus
Trees = [s.capitalize() for s in trees]
# find out which ones we have equations for
ns = allo_df['Scientific Name'].unique()
scrap = []
for n in Trees:
    if n not in ns:
        scrap.append(n)
Trees = [s for s in Trees if s not in scrap]
trees = [t.lower() for t in Trees] # i should have just capitalized name_mathched long ago
la = la.loc[la.Name_matched.isin(trees)]

# converts inch hi, low into cm hi and low floats
@np.vectorize
def convert(x, y): 
    return(x * 2.54, y * 2.54)

la['dbh_lo_cm'], la['dbh_hi_cm'] = convert(la.DBH_LO, la.DBH_HI)

# for now
la['region'] = 'SoCalC'

#%% [markdown]
# Use the allometric equations to calculate the max age and min age.  mse values are found in the c column


#%%
new_la = []
for tree in Trees:
    try:
        row = allo_df.loc[(allo_df['Scientific Name'] == tree) & (allo_df['Region'] == 'SoCalC')][['EqName', 'a', 'b', 'c', 'd', 'e', 'Apps min', 'Apps max']].values[0]

        eq, a, b, c, d, e, apps_min, apps_max = row

        mse = c # mse values are found in the c column

        # make df of trees of species with dbh in the viable range for the the eq
        df = la.loc[(la.Name_matched == tree.lower()) & (la.DBH_LO > apps_min) & (la.DBH_HI < apps_max)]

        df['age_max'] = eq_dict[eq](a, b, c, d, e, df.dbh_hi_cm, mse)
        df['age_min'] = eq_dict[eq](a, b, c, d, e, df.dbh_lo_cm, mse)
        new_la.append(df)
    except IndexError:
        pass

new_la = pd.concat(new_la)

#%% [markdown]
# Print a list of the tree species that we now have age estimates for.

#%%
names = list(new_la.Name_matched.unique())
names.sort()
names

#%% [markdown]
# then make violin plots showing the age distribution of the trees we have estimates for.

#%%

df = new_la[['ID', 'Name_matched', 'age_max', 'age_min']]
df['age_max'] = df['age_max'] + 3
df['age_min'] = df['age_min'] - 3
df['age_sim'] = df['age_sim'] = np.random.uniform(df.age_max, df.age_min)
df.drop(['age_min', 'age_max'], axis=1, inplace=True)
df = df.loc[df.Name_matched.isin(names[:4])]
df['Name_matched'] = np.vectorize(str.capitalize)(df.Name_matched)
df = pd.melt(df, id_vars=['ID', 'Name_matched'], value_vars=['age_sim'])
df.columns = ['ID', 'Species', 'variable', '~Age (yrs)']
#sns.swarmplot(x='Species', y='~Age (yrs)', data=df, size=0.5)
sns.violinplot(x='Species', y='~Age (yrs)', data=df, inner='box', cut=0, scale='count', bw=0.3)
sns.despine(offset=10, trim=True);
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('{}allo1.pdf'.format(fig_path, bbox_inches='tight'))


#%%
df = new_la[['ID', 'Name_matched', 'age_max', 'age_min']]
df['age_max'] = df['age_max'] + 3
df['age_min'] = df['age_min'] - 3
df['age_sim'] = df['age_sim'] = np.random.uniform(df.age_max, df.age_min)
df.drop(['age_min', 'age_max'], axis=1, inplace=True)
df = df.loc[df.Name_matched.isin(names[4:8])]
df['Name_matched'] = np.vectorize(str.capitalize)(df.Name_matched)
df = pd.melt(df, id_vars=['ID', 'Name_matched'], value_vars=['age_sim'])
df.columns = ['ID', 'Species', 'variable', '~Age (yrs)']
sns.violinplot(x='Species', y='~Age (yrs)', data=df, inner='box', cut=0, scale='count', bw=0.3)
sns.despine(offset=10, trim=True);
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('{}allo2.pdf'.format(fig_path, bbox_inches='tight'))

#%%
df = new_la[['ID', 'Name_matched', 'age_max', 'age_min']]
df['Name_matched'] = np.vectorize(str.capitalize)(df.Name_matched)
df['Age_estimate'] =  df.age_min + (df.age_max - df.age_min) / 2
names = df.Name_matched.unique()
E = []
med = []
stand_dev = []
for name in names:
    E.append(df[df.Name_matched == name].Age_estimate.mean())
    med.append(df[df.Name_matched == name].Age_estimate.median())
    stand_dev.append(df[df.Name_matched == name].Age_estimate.std())

age_estimates = pd.DataFrame()
age_estimates['Species'] = names
age_estimates['Age_estimate'] = E
age_estimates['Median'] = med
age_estimates['sd'] = stand_dev
age_estimates = age_estimates.round(1)

age_estimates.to_csv('{}LA_tree_age_estimates.csv'.format(fig_path))
# %% [markdown]
# ###

#%%
# this is not interesting
fig = plt.figure(figsize=(10.5, 8))
sns.scatterplot(x='LONGITUDE', y='sim_dbh', hue='Name_matched' ,data=new_la.sample(frac=0.01))
plt.tight_layout()

# %%


# %%
