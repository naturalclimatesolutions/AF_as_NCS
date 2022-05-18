import numpy as np
import geopandas as gpd

# load lesiv AF area data
lesiv_area = gpd.read_file('./Lesiv_et_al_national_AF_areas.shp')

# calculate global area, in Mha (incoming data in m^2)
global_area = np.sum(lesiv_area['sum']) * (1/10000) /1e6
print(('\n\nLesiv et al. 2022 data estimates a total, global AF area '
       'of %0.2f Mha.\n\n') % global_area)

# load lesiv AF C data
lesiv_C = gpd.read_file('./Lesiv_et_al_national_AF_C_stocks.shp')

# calculate global AF C, in Pg (incoming data in Mg)
global_C = np.sum(lesiv_C['sum']) / 1e9
print(('\n\nLesiv et al. 2022 data, combined with WHRC global AGB maps, '
       'estimates a total, global AF AGC stock of '
       '%0.2f Pg.\n\n') % global_C)


