import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import rioxarray as rxr
import os

# load datasets
rast_datadir = '/media/deth/SLAB/TNC/tmp_lesiv_forest_mgmt_data'
lesiv_rast = rxr.open_rasterio(os.path.join(rast_datadir,
                                            'FML_v3.2.tif'),
                               masked=False,
                               cache=True,
                               chunks=(5,5),
                              )[0]

# keep only AF pixels
lesiv_rast = (lesiv_rast==53).astype(np.int8)
lesiv_rast = lesiv_rast.where(lesiv_rast==1, np.nan)

af_locs_datadir = ('/home/deth/Desktop/TNC/repos/agrofor_lit_review/'
                   'mapping/AF_locations_from_papers/')
af_locs = gpd.read_file(os.path.join(af_locs_datadir,
                                     'AF_locations_from_meta-analyses.shp'))

# figure out if each location is covered by Lesiv data or not
in_lesiv = []
for i, row in af_locs.to_crs(lesiv_rast.rio.crs).iterrows():
    lon, lat = [i[0] for i in row.geometry.coords.xy]
    lesiv_vals = lesiv_rast.sel(x=lon, y=lat, method='nearest').values
    assert lesiv_vals.size == 1
    lesiv_val = float(lesiv_vals)
    in_lesiv.append(pd.notnull(lesiv_val))
af_locs['in_lesiv'] = in_lesiv

# write to file
af_locs.to_file(os.path.join(af_locs_datadir,
                             'AF_locations_from_meta-analyses_W_LESIV.shp'))
