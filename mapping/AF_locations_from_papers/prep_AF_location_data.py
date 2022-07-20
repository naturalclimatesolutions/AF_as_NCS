import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import re


# params
n_coords_dec_for_dedup = 4 # NOTE: ~=100m, i.e., res of Lesiv data we're
                           #       calculating a global C estimate from
plot_it = False

# load data
fulldf = pd.read_excel('./Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx',
                         sheet_name='S2 sites')

# subset columns
df = fulldf.loc[:, ['site.id', 'study.id', 'site.sitename', 'site.state',
                    'site.country', 'lat', 'lon',
                    'lat_deg', 'lat_min', 'lat_sec', 'N_S',
                    'long_deg', 'long_min', 'long_sec', 'E_W', 'other.reference']]

# convert 999s to NaNs
df = df.replace({999: np.nan})

# rename columns (GEE doesn't like periods in them)
df.columns = [col.replace('.', '_') for col in df.columns]


# function for handling strings with degree and minute signs in them
def handle_deg_min_str(val):
    dir = re.split(' ', val.strip())[-1]
    assert dir in ['E', 'W', 'N', 'S']
    split_vals = [substr.strip() for substr in re.split('to', val)]
    assert len(split_vals) == 2
    dec_vals = []
    for subval in split_vals:
        deg, rest = re.split('°', subval)
        min, rest = re.split('′', rest)
        deg = float(deg)
        min = float(min)
        dec_val = deg + (min /60)
        if dir in ['W', 'S']:
            dec_val *= -1
        dec_vals.append(dec_val)
    assert len(dec_vals) == 2
    dec_val = np.mean(dec_vals)
    return dec_val


# function for handling strings with cardinal directions on the end
def handle_NSEW_str(val):
    num, dir = re.split(' ', val)
    num = num.strip()
    dir = dir.strip()
    dir = dir.upper()
    assert dir in ['N', 'S', 'E', 'W']
    num = float(num)
    if dir in ['W', 'S']:
        num *= -1
    return num


# try to use deg, min, sec, and cardinal direction columns to calculate
# decimal-degree lat, lon values, if decimal-degree coordinates are initially
# missing, else just return existing decimal-degree lat/lon values
def calc_dec_lat_lon(row):
    if pd.notnull(row['lat']) and pd.notnull(row['lon']):
        dec_vals = [row['lat'], row['lon']]
        for i, val in enumerate(dec_vals):
            if (isinstance(val ,str) and
                (re.search(' to ', val) or
                 re.search('\d+.* [NSEW]', val))):
                if re.search('°', val):
                    try:
                        new_val = handle_deg_min_str(val)
                    except Exception:
                        new_val = np.nan
                elif re.search('\d+.* [NSEW]', val):
                    try:
                        new_val = handle_NSEW_str(val)
                    except Exception:
                        new_val = np.nan
                else:
                    new_val = np.mean([float(n.strip(
                        )) for n in re.split('to', val)])
                dec_vals[i] = new_val
    else:
        try:
            dec_vals = []
            for dim, cols in {'lat': ['lat_deg', 'lat_min', 'lat_sec', 'N_S'],
                              'lon': ['long_deg', 'long_min', 'long_sec', 'E_W'],
                             }.items():
                # get a copy of the values
                vals = deepcopy(row[cols].values)
                # convert any range strings (e.g., '7 to 9') to averages
                for i, val in enumerate(vals):
                    if (isinstance(val ,str) and
                        re.search('^-?\d+\.?\d* to -?\d+\.?\d*$', val)):
                        new_val = np.mean([float(n.strip(
                            )) for n in re.split('to', val)])
                        vals[i] = new_val
                # break out values in separate variables, then use to calculate
                # decimal degrees
                deg, min, sec, dir = row[cols].values
                dec = deg + (min + (sec/60))/60
                if dir in ['W', 'S']:
                    dec *= -1
                dec_vals.append(dec)
        except Exception:
            dec_vals = [np.nan, np.nan]

    return dec_vals


# run checks and filters on geo-coordinate columns
keep = np.zeros(len(df))
assert len(keep) == df.shape[0]

standard_lat = []
standard_lon = []
for i, row in df.iterrows():
    try:
        new_lat_lon = calc_dec_lat_lon(row)
        for val in new_lat_lon:
            assert isinstance(val, float) or val is None, '%s' % val
    except Exception:
        new_lat_lon = [np.nan, np.nan]
    standard_lat.append(new_lat_lon[0])
    standard_lon.append(new_lat_lon[1])

df['lat'] = standard_lat
df['lon'] = standard_lon

# get rid of rows where lat or lon values are anomalous
# (indicating they were probably switched when inputted)
df = df[np.logical_and(np.abs(standard_lat) <= 90, np.abs(standard_lon) <= 180)]

# assert that all missing coords have now been dropped
assert np.sum(np.isnan(df['lat'])) == 0
assert np.sum(np.isnan(df['lon'])) == 0

# create rounded columns, to use for deduplication
df['lat_trunc'] = np.round(df['lat'] ,n_coords_dec_for_dedup)
df['lon_trunc'] = np.round(df['lon'] ,n_coords_dec_for_dedup)

# remove duplicate coords (out to specifed number of decimal places)
df = df.drop_duplicates(subset = ['lon_trunc', 'lat_trunc'])

# scatter points, if requested
if plot_it:
    plt.scatter(df['lon_trunc'], df['lat_trunc'])
    plt.show()

# turn into GeoDataFrame
gdf = gpd.GeoDataFrame(
        df.drop(['lon_trunc', 'lat_trunc'], axis=1),
        crs={'init': 'epsg:4326'},
        geometry=[Point(lonlat) for lonlat in zip(df.lon_trunc, df.lat_trunc)])

# write to shapefile
gdf.to_file('AF_locations_from_meta-analyses.shp', index=False)
