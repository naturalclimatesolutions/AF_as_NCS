import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import geopandas as gpd
from shapely.geometry import Point
from copy import deepcopy
import re


# colors
col_1dec = '#dbaff0'
col_2dec = '#cd74f7'
col_3dec = '#8306bf'


##########################
# load and preprocess data
##########################
# read in our points collected from the lit
af_locs = pd.read_excel(('../fig3_maps_fig5_curr_vs_poten/AF_locations_from_papers/'
                         'Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx'),
                        sheet_name='S2 sites')

# subset columns
df = af_locs.loc[:, ['site.id', 'study.id', 'site.sitename', 'site.state',
                    'site.country', 'lat', 'lon',
                    'lat_deg', 'lat_min', 'lat_sec', 'N_S',
                    'long_deg', 'long_min', 'long_sec', 'E_W', 'other.reference']]

# convert 999s to NaNs
df = df.replace({999: np.nan})

# save length of raw data
orig_len = len(df)

# drop anything for which we could not or have not gathered coordinate
# information (in some way or another), and report summary info
has_dec_coords = np.logical_and(*[pd.notnull(df[col]) for col in ['lat',
                                                                  'lon']])
has_dms_coords = np.logical_and(*[pd.notnull(df[col]) for col in ['lat_deg',
                                                                  'long_deg']])
has_coords = np.logical_or(has_dec_coords, has_dms_coords)
print(('\n\n%0.2f%% OF ORIGINAL ROWS '
       '(%i of %i rows) HAVE NO COORDS\n'
       '\n' % ( 100*(1-np.sum(has_coords)/orig_len),
               np.sum(np.invert(has_coords)),
               orig_len)))

# drop anything with strings for coordinates (instead of single numbers;
# because they generally indicate that coordinates were given as a range of
# values, so precision generally very low, and at any rate not worth assessing
# here)
has_str_dec_coords = np.logical_or(*[[isinstance(c,
        str) for c in df[col]] for col in ['lat', 'lon']])
str_fn = lambda x: isinstance(x, str)
dms_cols = ['lat_deg', 'lat_min', 'lat_sec', 'long_deg', 'long_min', 'long_sec']
has_str_dms_coords = np.bool8(np.sum(np.stack([df[col].apply(
                                str_fn) for col in dms_cols]), axis=0))
has_str_coords = np.logical_or(has_str_dec_coords, has_str_dms_coords)
print(('\n\n%0.2f%% OF ORIGINAL ROWS '
       '(%i of %i rows) HAD COORDS EXPRESSED AS RANGES OR '
       'OTHERWISE EXPRESSED AS STRINGS\n'
       '\n') % (100*(np.sum(has_str_coords)/orig_len),
                np.sum(has_str_coords),
                orig_len))

# drop anything that used Google Maps
used_google = []
for ref in df['other.reference']:
    if pd.notnull(ref) and isinstance(ref, str):
        res = re.search('google', ref, flags=re.IGNORECASE)
        used_google.append(res is not None)
    else:
        used_google.append(False)
assert len(used_google) == len(df)
print(('\n\n%0.2f%% OF ORIGINAL ROWS '
       '(%i of %i rows) HAD COORDS THAT WERE GATHERED USING '
       'GOOGLE MAPS OR GOOGLE EARTH\n'
       '\n') % (100*(np.sum(used_google)/orig_len),
                np.sum(used_google),
                orig_len))

keep_idxs = np.logical_and(np.logical_and(has_coords, np.invert(has_str_coords)),
                           np.invert(used_google))
print(('\n\nDROPPING %0.2f%% OF ORIGINAL ROWS '
       '(%i of %i rows) BECAUSE OF NO COORDS, STRING COORDS, '
       'OR COORDS DERIVED FROM GOOGLE\n'
       '\n') % (100*(1-np.sum(keep_idxs)/orig_len),
                orig_len - np.sum(keep_idxs),
                orig_len))
df = df[keep_idxs]

print('\n\nANALYZING %i ROWS FROM %i STUDIES\n\n' % (len(df),
                                                len(df['study.id'].unique())))

# make sure there are no remaining strings in the non-DMS coord columns
for i, row in df.iterrows():
    for col in dms_cols:
        assert (pd.isnull(row[col]) or
                isinstance(row[col], int) or
                isinstance(row[col], float)), '%s' % str(row)

# function to convert DMS coords to decimal-degree coords
def convert_dms_2_dec(d, m, s, dir_str):
    assert dir_str.strip().upper() in ['E', 'W', 'N', 'S'], dir_str
    # need at least degrees in order to calculate
    assert pd.notnull(d)
    # set missing minutes or secords to 0s (which will still accurately
    # propagate through as lower precision than finer minute or second
    # distinctions)
    if pd.isnull(m):
        m = 0
    if pd.isnull(s):
        s = 0
    dec = d + ((m + (s/60))/60)
    if dir_str.strip().upper() in ['W', 'S']:
        dec *= -1
    return dec


# standardize coords by convert DMS to decimal degrees
standard_lat = []
standard_lon = []
standard = {'lat': standard_lat,
            'lon': standard_lon,
           }
dir_cols = {'lat': 'N_S',
            'lon': 'E_W',
           }
for i, row in df.iterrows():
    for dim in ['lat', 'lon']:
        if pd.notnull(row[dim]):
            standard[dim].append(row[dim])
        else:
            dec = convert_dms_2_dec(row[dim + ('g'*(dim=='lon')) + '_deg'],
                                    row[dim + ('g'*(dim=='lon')) + '_min'],
                                    row[dim + ('g'*(dim=='lon')) + '_sec'],
                                    row[dir_cols[dim]],
                                   )
            standard[dim].append(dec)
for dim, dim_list in standard.items():
    assert len(dim_list) == len(df)
    assert np.nan not in dim_list
    df[dim] = dim_list


##################################
# add coordinate-precision columns
##################################

# assess variance in divergence from 1:1 line as fn of geo coord precision
def estimate_coord_precision(coord):
    ct = 0
    # return 0 if no decimals
    if '.' not in str(coord):
        return ct
    str_dec = str(coord).split('.')[1]
    for i, dig in enumerate(str_dec):
        ct += 1
        if i+1 == len(str_dec):
            return ct
        elif i+2 == len(str_dec):
            if str_dec[i+1] == dig:
                return ct
            else:
                return ct+1
        else:
            if str_dec[i+2] == str_dec[i+1] == dig:
                return ct

def calc_coord_precision_column(df):
    lat_prec = [estimate_coord_precision(lat) for lat in df.lat]
    lon_prec = [estimate_coord_precision(lon) for lon in df.lon]
    prec_col = (np.array(lat_prec) + np.array(lon_prec))/2
    # reclass the numeric coord_prec
    class_bin_lefts= [1, 2, 3]
    class_bin_rights = [2, 3, 10000]
    class_bins = ['low', 'mod', 'high']
    prec_bin_col = []
    for val in prec_col:
        for bin, l, r in zip(class_bins, class_bin_lefts, class_bin_rights):
            if l<= val < r:
                prec_bin_col.append(bin)
    assert len(prec_bin_col) == len(prec_col)
    return prec_col, prec_bin_col

prec_col, prec_bin_col = calc_coord_precision_column(df)
df['coord_prec'] = prec_col
df['coord_prec_bin'] = prec_bin_col


###################################################################
# calculate percentage of measurements with each level of precision
###################################################################
pct_lt1dec = np.mean(df.coord_prec<1)
pct_1dec = np.mean((df.coord_prec>=1) * (df.coord_prec<2))
pct_2dec = np.mean((df.coord_prec>=2) * (df.coord_prec<3))
pct_3dec = np.mean(df.coord_prec>=3)
assert np.allclose(1, np.sum((pct_lt1dec, pct_1dec, pct_2dec, pct_3dec)))
print('\n\n%0.3f%% available coordinates < 1 decimal degrees precision!\n' % (100*pct_lt1dec))
print('\n\n%0.3f%% available coordinates >= 1 and < 2 decimal degrees precision\n' % (100*pct_1dec))
print('\n\n%0.3f%% available coordinates >= 2 and < 3 decimal degrees precision\n' % (100*pct_2dec))
print('\n\n%0.3f%% available coordinates >3 decimal degrees precision\n' % (100*pct_3dec))


#############
# make legend
#############
fig, ax = plt.subplots(1,1, figsize=(8,2))
precisions = {1: '11 km',
              2: '1.1 km',
              3: '110 m',
             }
pcts = {1: pct_1dec + pct_2dec + pct_3dec,
        2: pct_2dec + pct_3dec,
        3: pct_3dec,
       }
#sizes = {1: 1500, 2: 500, 3: 75}
sizes = {1: 1500, 2: 1500, 3: 1500}
for i, col in enumerate([col_1dec, col_2dec, col_3dec]):
    pct = pcts[i+1] * 100
    label = '%i decimal place%s = ~%s precision (%0.1f%%)'  % (i+1,
                                                    's' * (i>0),
                                                    precisions[i+1], pct)
    ax.scatter(1, 3-i, c=col, facecolor=col, edgecolor='black', marker='_', s=sizes[i+1],
               linewidth=3)
    ax.text(1.2, 3-i, label, size=18)

# activate latex text rendering, then use in legend title
ax.text(1.13, 4, 'coordinate precision (and % reporting)',
        size=18,
        weight='bold')
ax.set_xlim(0.8,3)
ax.set_ylim(0.3,4.6)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
# box around edge
ax.plot([*ax.get_xlim()]+[*ax.get_xlim()[::-1]]+[ax.get_xlim()[0]],
        [*np.repeat(ax.get_ylim(), 2)]+[ax.get_ylim()[0]], '-k', linewidth=5)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
fig.show()
fig.savefig('MRV_radii_legend.png', dpi=600)
