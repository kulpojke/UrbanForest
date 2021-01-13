#%%
import json  

#%%
zone_dict = {'CenFla' : [26],
             'GulfCo' : [27, 28],
             'InlEmp' : [18, 19, 20, 21],
             'InlVal' : [7, 8, 9, 14],
             'InterW' : [2, 10],
             'LoMidW' : [35],
             'MidWst' : [36, 41, 43],
             'NMtnPr' : [1,44,45],
             'NoCalC' : [15, 16, 17],
             'NoEast' : [34, 37, 38, 39, 40, 42], 
             'PacfNW' : [4, 5, 6],
             'Piedmt' : [29, 30, 31, 32, 33],
             'SoCalC' : [22, 23, 24],
             'SWDsrt' : [11, 12, 13],
             'TpIntW' : [3],
             'Tropic' : [25]}

#%% make sure zones are all unique
z =[x for val in zone_dict.values() for x in val]
assert len(z) == len(list(set(z)))

# %%
with open('zones.json', 'w') as of:
    json.dump(zone_dict, of)


# %%
