import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import re


# params
n_coords_dec_for_dedup = 3
plot_it = False

# load data
fulldf = pd.read_excel('./Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx',
                         sheet_name='S2 sites')

# subset columns
df = fulldf.loc[:, ['site.id', 'study.id', 'site.sitename', 'site.state',
                    'site.country', 'lat', 'lon', 'other.reference']]

# rename columns (GEE doesn't like periods in them)
df.columns = [col.replace('.', '_') for col in df.columns]

# drop rows with bad lats or lons
df = df[((pd.notnull(df.lon).values)*
               (pd.notnull(df.lat))*
               (pd.notnull(df['site_id'])))]
keep_rows = []
new_lons = []
new_lats = []
for i, row in df.iterrows():
    try:
        # average vals expressed as 'x1 to x2', then convert to float and round
        if (isinstance(row['lat'],str) and
            re.search('^-?\d+\.?\d* to -?\d+\.?\d*$', row['lat'])):
            new_lat = np.round(np.mean([np.float(n.strip()) for n in
                    re.split('to', row['lat'])]), n_coords_dec_for_dedup)
        # otherwise just convert to float and round
        else:
            new_lat = np.round(float(row['lat']), n_coords_dec_for_dedup)
        assert -90<=new_lat<=90
        if (isinstance(row['lon'],str) and
            re.search('^-?\d+\.?\d* to -?\d+\.?\d*$', row['lon'])):
            new_lon = np.round(np.mean([np.float(n.strip()) for n in
                    re.split('to', row['lon'])]), n_coords_dec_for_dedup)
        else:
            new_lon = np.round(float(row['lon']), n_coords_dec_for_dedup)
        assert -180<=new_lon<=180
        # save vals
        new_lons.append(new_lon)
        new_lats.append(new_lat)
        keep_rows.append(i)
    except Exception as e:
        print(e)
        print(row['lat'], row['lon'])
        print('-'*80)
        pass
df = df.loc[keep_rows,:]
df['lon_trunc'] = new_lons
df['lat_trunc'] = new_lats

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
