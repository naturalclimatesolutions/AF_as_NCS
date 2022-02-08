import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from shapely.geometry import Polygon
from scipy.stats import ttest_ind
from copy import deepcopy
import warnings

# plot params
save_it = False
suptitle_fontsize = 50
title_fontsize = 40
contour_axislab_fontsize = 10
contour_ticklab_fontsize = 7
annot_fontsize = 14
cbar_fontsize = 14
fig1_width = 5
fig1_height = 7
dpi = 400
n_ticklabels = 5
contour_alpha = 0.5
contour_linewidth = 0.1
contour_linecolor = 'gray'
contour_lims = [0,1]
vert_connector_linewidth = 0.25
subplots_adj_left=0.01
subplots_adj_bottom=0.01
subplots_adj_right=0.99
subplots_adj_top=0.99
subplots_adj_wspace=0.01
subplots_adj_hspace=0.01
min_x=0
max_x=1
min_y=0
max_y=1
savefig=True
savefig=True
x_buff=max_x/20
orientation='landscape'
savefig=True


# load datasets
rosenstock = gpd.read_file(('./rosenstock_et_al_2019_data/'
                            'rosenstock_et_al_2019_AF_NDCs_db.shp'))
chapman_C_agg = gpd.read_file(('./chapman_data_aggregated/'
                               'chapman_crop_and_pasture_country_agg.shp'))
chapman_ag_area_agg = gpd.read_file(('./chapman_data_aggregated/'
                                       './chapman_ag_land_area_country_agg.shp'))
af_locs = gpd.read_file(('AF_locations_from_papers/'
                         'AF_locations_from_meta-analyses.shp'))


# add continents to datasets
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# manually change some continent assignments (French Guiana, Russia)
countries.loc[countries['name'] == 'Russia', 'continent'] = 'Asia'
fake_row = countries.iloc[0,:]
france_poly = countries[countries.name == 'France'].geometry
french_guiana_poly = france_poly.intersection(Polygon([[-40,0], [-70,0],
                                                       [-70,20], [-40,20],
                                                       [-40,0]]))
real_france_poly = france_poly.intersection(Polygon([[-10,20], [30,20],
                                                       [30,60], [-10,60],
                                                       [-10,20]]))
countries = countries[np.invert(countries.name == 'France')]
france_row = deepcopy(fake_row)
france_row.name == 'France'
france_row.geometry = france_poly
france_row.continent = 'Europe'
countries = countries.append(gpd.GeoDataFrame({**france_row}))
french_guiana_row = deepcopy(fake_row)
french_guiana_row.name == 'French Guiana'
french_guiana_row.geometry = french_guiana_poly
french_guiana_row.continent = 'South America'
countries = countries.append(gpd.GeoDataFrame({**french_guiana_row}))

# create Central America and Caribbean
for country in ['Haiti', 'Dominican Rep.', 'Bahamas',
                'Panama', 'Costa Rica', 'Nicaragua',
                'Honduras', 'El Salvador',
                'Guatemala', 'Belize',
                'Puerto Rico', 'Jamaica',
                'Cuba', 'Trinidad and Tobago']:
    countries.loc[countries.name == country, 'continent'] = 'C. Am./Car.'

# dissolve to continents
continents = countries.dissolve('continent')

# shorten continent names
continents.index = ['Af.', 'Antarctica', 'Asia', 'C. Am./Car.',
                    'Eur.', 'N. Am.', 'Oceania', 'Seven seas', 'S. Am.']

def get_continents(df):
    conts = []
    for i, row in df.iterrows():
        intsxn = continents.geometry.intersection(row.geometry)
        try:
            if np.sum(np.invert(intsxn.is_empty)) == 1:
                match = intsxn[np.invert(intsxn.is_empty)]
                cont = match.index[0]
                conts.append(cont)
            elif np.sum(np.invert(intsxn.is_empty)) >1:
                area_match = intsxn[intsxn.geometry.area ==
                                        np.max(intsxn.geometry.area)]
                assert len(area_match) == 1, 'could not find 1 max-area continent!'
                cont = area_match.index[0]
                conts.append(cont)
            else:
                #print('\n\n\nfound 0 matching continents\n\n')
                #print(row.CNTRY_N)
                #print('\n\n\n')
                conts.append(np.nan)
        except Exception as e:
            print(intsxn)
            print(row.CNTRY_N)
            print(e)
            conts.append(np.nan)
    assert len(conts) == len(df), 'len conts %i, len df %i' % (len(conts), len(df))
    return conts


rosenstock['cont'] = get_continents(rosenstock)
chapman_C_agg['cont'] = get_continents(chapman_C_agg)
chapman_ag_area_agg['cont'] = get_continents(chapman_ag_area_agg)
af_locs['cont'] = get_continents(af_locs)



# utils functions
# CODE ADAPTED FROM:
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
def hex2rgba(hex, alpha=255):
    rgb = [int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    rgba = tuple(rgb + [alpha])
    return rgba


# merge rosenstock and chapman data
ndcs = []
ncs = []
namas = []
for i, row in chapman_C_agg.iterrows():
    iso3 = row['ISO_3DIGIT']
    subdf_rosenstock = rosenstock[rosenstock['cn_ISO3'] == iso3]
    if len(subdf_rosenstock) == 1:
        ndcs.append(subdf_rosenstock.iloc[0]['NDCmnt'])
        ncs.append(subdf_rosenstock.iloc[0]['NCmnt'])
        namas.append(subdf_rosenstock.iloc[0]['NAMAmnt'])
    elif len(subdf_rosenstock) == 0:
        ndcs.append(np.nan)
        ncs.append(np.nan)
        namas.append(np.nan)
    else:
        print(subdf_rosenstock)
        raise(ValueError)
chapman_C_agg['NDC'] = ndcs
chapman_C_agg['NC'] = ncs
chapman_C_agg['NAMA'] = namas


# add country-aggregated ag land area col to country-aggregated ag woody C,
# then get area-normalized woody C
chapman_ag_area_agg['ha'] = chapman_ag_area_agg['sum']/10_000
chapman_C_agg['ha_ag_land'] = chapman_ag_area_agg['ha']
chapman_C_agg['Mg_C_per_ha'] = chapman_C_agg['sum']/chapman_C_agg['ha_ag_land']


##### ANALYSIS 1:

# prep data
data_fig1 = chapman_C_agg.loc[:, ['ha_ag_land',
                                  'Mg_C_per_ha',
                                  'NDC',
                                  'CNTRY_NAME',
                                  'cont',
                                  'geometry']]
data_fig1 = data_fig1.dropna(axis=0, how='any')
data_fig1 = data_fig1[data_fig1['ha_ag_land']>0]
data_fig1['NDC_num'] = data_fig1['NDC']
data_fig1['NDC'] = data_fig1['NDC'].map(lambda x: {1:'yes', 0:'no'}[x])



# plot dists of mean woody C/m^2 land area for countries w/ and w/out AF in NDCs
fig_1 = plt.figure()
#fig_1 = plt.figure(dpi=dpi,
#                   figsize=(fig1_width, fig1_height))
#gs = fig.add_gridspec(nrows=9, ncols=5,
#                      height_ratios=[1,1,1,1,1.5,1,1,1,1])
fig_1.suptitle(('average crop and pasture woody C density in\n'
                'countries that do and do not mention AF in NDCs'))
ax = fig_1.add_subplot(111)
sns.set_theme(style="whitegrid")
ax = sns.violinplot(ax=ax,
                    x='NDC',
                    y='Mg_C_per_ha',
                    data=data_fig1,
                    inner='box',
                    #orient='h',
                    palette=['#d1caa7', '#b4edb5'],
                   )
cont_lookup = dict(zip(data_fig1.cont.unique(),
                       range(len(data_fig1.cont.unique()))))
rev_cont_lookup = dict(zip(range(len(data_fig1.cont.unique())),
                       data_fig1.cont.unique()))
x_offset_unit = 0.06
scat = ax.scatter(data_fig1['NDC_num'] + (x_offset_unit *
                                   (2+data_fig1['cont'].map(lambda cont:
                                                        cont_lookup[cont]))),
           data_fig1['Mg_C_per_ha'],
           c = data_fig1['cont'].map(lambda cont: cont_lookup[cont]),
           s = 60,
           cmap='Set3',
           alpha=0.75,
           edgecolor='black',
           linewidth=0.5,
           label=data_fig1.cont,
          )
#leg = ax.legend(*scat.legend_elements(), title='Continent')
ax.set_xlabel('mention agroforestry in NDC?')
ax.set_ylabel('mean crop and pasture woody C density\n($Mg C / ha$)')
fig_1.show()

for cont in data_fig1.cont.unique():
    for NDC_status in range(2):
        x = NDC_status + (x_offset_unit * (2+cont_lookup[cont]))
        y = -0.3
        ax.text(x-0.005, y, cont, fontdict={'fontsize': 10,
                                       'weight': 'bold',
                                       'rotation': 'vertical'})
        if len(data_fig1.loc[(data_fig1.cont == cont) &
                             (data_fig1.NDC_num == NDC_status)]) > 0:
            ax.plot([x, x],
                    [0, np.max(data_fig1.loc[(data_fig1.cont == cont) &
                                         (data_fig1.NDC_num == NDC_status),
                                         'Mg_C_per_ha'])],
                    color='black', linewidth=0.2, alpha=1)
        else:
            ax.plot([x,x], [0, 0.01], color='black', linewidth=0.2, alpha=1)

#for i, row in data_fig1.iterrows():
#    ax.text(row['NDC_num'], row['Mg_C_per_ha'], row['CNTRY_NAME'],
#            fontsize=6)

# t-test of significant diff between NDC and non-NDC groups
res = ttest_ind(data_fig1[data_fig1.NDC_num==1]['Mg_C_per_ha'],
                data_fig1[data_fig1.NDC_num==0]['Mg_C_per_ha'])
print(('\n\nt-test of sig. diff. between woody C ag-land density in NDC and '
       'non-NDC countries:\n\tt-stat: %0.3f\n\tp-value: '
       '%0.3f') % (res.statistic, res.pvalue))

ax.text(-0.4, -0.3, 't-stat: %0.3f\np-value: %0.3f' % (res.statistic,
                                                        res.pvalue),
        fontdict={'fontsize': 10, 'fontstyle': 'italic'})

#fig_1.subplots_adjust(left=subplots_adj_left,
#                    bottom=subplots_adj_bottom,
#                    right=subplots_adj_right,
#                    top=subplots_adj_top,
#                    wspace=subplots_adj_wspace,
#                    hspace=subplots_adj_hspace)


fig_1.show()

if save_it:
    fig_1.savefig('woody_C_in_AF_NDC_non-NDC_countries.png',
                  dpi=dpi, orientation='portrait')


#### ANALYSIS 2:
# plot AF sites on top of area-normalized ag woody C,
# with separate colormaps for NDC and non-NDC countries

fig2 = plt.figure()
fig2.suptitle('average woody C density and known agroforestry locations')

# set colormaps
cmaplist_NDC = []
for val, col in zip([0,1], ['#bee6bc', '#035900']):
    cmaplist_NDC.append((val, col))
cmap_NDC = LinearSegmentedColormap.from_list("custom", cmaplist_NDC)
cmaplist_nonNDC = []
for val, col in zip([0,1], ['#706a3f', '#edd20c']):
    cmaplist_nonNDC.append((val, col))
cmap_nonNDC = LinearSegmentedColormap.from_list("custom", cmaplist_nonNDC)
cmaps = {0: cmap_nonNDC,
         #1: 'Greens',
         1: cmap_NDC,
        }

# plot it
ax = fig2.add_subplot(111)
divider = make_axes_locatable(ax)
rcax = divider.append_axes("right", size="5%", pad=0.1)
lcax = divider.append_axes("left", size="5%", pad=0.1)
cax_dict = {0: lcax, 1:rcax}

# plot NDC and non-NDC countries' woody C density separately
cbar_title_lookup = {0: 'AF not in NDC', 1: 'AF in NDC'}
for NDC_status in range(2):
    subdf = data_fig1[data_fig1.NDC_num == NDC_status]
    if NDC_status == 1:
        edgecolor = '#003604'
        linewidth = 1.5
    else:
        edgecolor='black'
        linewidth=0.25
    map = subdf.plot('Mg_C_per_ha',
                     ax=ax,
                     cmap=cmaps[NDC_status],
                     edgecolor=edgecolor,
                     linewidth=linewidth,
                     legend=True,
                     legend_kwds={'label': 'Mg woody C/ha (crop & pasture)',
                                  'orientation': "vertical"},
                     cax=cax_dict[NDC_status])
rcax.set_title(cbar_title_lookup[1])
lcax.set_title(cbar_title_lookup[0])
lcax.yaxis.set_ticks_position('left')
lcax.yaxis.set_label_position('left')

# outline missing country boundaries
chapman_C_agg[pd.isnull(chapman_C_agg.NDC)].plot(edgecolor='black',
                                                 facecolor='white',
                                                 linewidth=0.25,
                                                 ax=ax)
# add locations
ax.scatter(af_locs.geometry.x, af_locs.geometry.y,
           c='black',
           s=5,
           alpha=1)

#ax.set_xlabel('$^{\circ} lon.')
#ax.set_ylabel('$^{\circ} lat.')
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

fig2.show()


if save_it:
    fig2.savefig('woody_C_density_and_known_AF_locs.png',
                  dpi=dpi, orientation='landscape')




#### ANALYSIS 3:
# compare resolutions







##########################################
##########################################
##########################################
##########################################
##########################################


