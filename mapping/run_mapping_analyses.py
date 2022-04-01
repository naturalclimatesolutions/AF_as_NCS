import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import palettable
import seaborn as sns
import rioxarray as rxr
from shapely.geometry import Polygon
from scipy.stats import ttest_ind
from scipy import stats
from copy import deepcopy
import warnings


# TODO:

    # clean up Chapman map and format and save 

    # tie up boxplots
        # nix fliers
        # format axes, labels, etc

    # need to plot or at least test separate slopes for each region
    # (folding Oceania into Asia I guess?)?



# plot params
save_it = True
map_minx = -13700000
map_maxx = 16500000
map_miny = -7000000
map_maxy = 8392644
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
savefig=True

# TODO:

    # replace my Chapman data with her original
        # read in:
            # decide if need sep crop and pasture maps or all combined okay
            # produce pair(s?) of current and potential maps

    # look into Brandt images

    # reread Sam's meeting notes and decide if any other maps needed

    # reach out to Sam and Susan with update
        # tell Sam about possibility of extracting NDC info later on,
        # but currently plugging in countries from IUCN 2018 (from Chapman)

    # put maps in draft fig, ping group for review

    # read Damien papers some more

    # reply to Damien email

    # rework co-benefits section a touch
        # private vs public, and private benefits main reasons for AF adoption
        # and maintenance

        # context dependence (need for local knowledge, research, extension)

        # trade-offs possible --> no hype! farmers need comprehensive info

    # look at AF survey results thus far

    # follow up with Susan to schedule C fig brainstorm

    # continue reworking text


# load datasets
rosenstock = gpd.read_file(('./rosenstock_et_al_2019_data/'
                            'rosenstock_et_al_2019_AF_NDCs_db.shp'))
chapman_C_agg = gpd.read_file(('./chapman_data_aggregated/'
                               'chapman_crop_and_pasture_country_agg.shp'))
chapman_ag_area_agg = gpd.read_file(('./chapman_data_aggregated/'
                                       './chapman_ag_land_area_country_agg.shp'))
roe = pd.read_excel('./Roe_et_al_SI.xlsx',
                    sheet_name='1. Sectoral mitigation-country',
                    skiprows=10)
ndc_contrib_raw = pd.read_csv(('./NDC_contribs/pathway_mitigation_potential_and_NDC'
                               '_targets_with_ISO3.csv'))
af_locs = gpd.read_file(('AF_locations_from_papers/'
                         'AF_locations_from_meta-analyses.shp'))

# load the IUCN AF NDC-mentions data from 2018 report
# (gleaned in a cleaned form from Millie Chapman's work:
#  https://raw.githubusercontent.com/milliechapman/treesincroplands/
#  master/data/IUCN_ndc_agroforestry.csv)
# then fold into the Rosenstock data to supplement it
iucn = pd.read_csv(('./rosenstock_et_al_2019_data/'
                    'supplemental_IUCN_NDC_data_from_Chapman_supps.csv'))
iucn = iucn.iloc[:, :3]
supplemented_NDCmnt = []
for i, row in rosenstock.iterrows():
    if pd.isnull(row['NDCmnt']):
        try:
            sub_iucn = iucn[iucn['ISO_A3'] == row['cn_ISO3']]
            assert len(sub_iucn) == 1
            NDCmnt = sub_iucn.iloc[0]['Agroforestry']
            supplemented_NDCmnt.append(NDCmnt)
        except Exception as e:
            # assign France's value to French Guiana
            if row.CNTRY_N == 'French Guiana':
                val = iucn[iucn.ISO_A3=='FRA']['Agroforestry'].values[0]
                print('ADDING FRENCH GUIANA:', val)
                supplemented_NDCmnt.append(val)
            # assign US value to Puerto Rico
            elif row.CNTRY_N == 'Puerto Rico':
                val = iucn[iucn.ISO_A3=='USA']['Agroforestry'].values[0]
                print('ADDING PUERTO RICO:', val)
                supplemented_NDCmnt.append(val)
            # Russia's IUCN code is wrong in the Chapman data for some reason
            # (ROM instead of RUS)
            elif row.CNTRY_N == 'Russia':
                val = iucn[iucn['Symbol ']=='Russia']['Agroforestry'].values[0]
                print('ADDING RUSSIA: ', val)
                supplemented_NDCmnt.append(val)
            else:
                print(('\n\nNOTE: NDCmnt is null in Rosenstock '
                       'for %s (ISO3: %s), but was not '
                       'succesfully found in '
                       'IUCN table\n\n\tError: %s\n\n') % (row.CNTRY_N,
                                                       row.cn_ISO3,
                                                       e))
                supplemented_NDCmnt.append(np.nan)
    else:
        supplemented_NDCmnt.append(row['NDCmnt'])
    assert len(supplemented_NDCmnt) == i+1
assert len(supplemented_NDCmnt) == len(rosenstock)
# replace the NDCmnt column in Rosenstock with the new, supplemented one
rosenstock['NDCmnt'] = supplemented_NDCmnt


# add continents to datasets
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# drop the one 'Seven seas (open ocean)' polygon
countries = countries[np.invert(countries.continent == 'Seven seas (open ocean)')]
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
france_row['name'] = 'France'
france_row['geometry'] = real_france_poly
france_row['continent'] = 'Europe'
france_row['iso_a3'] = 'FRA'
france_row['gdp_md_est'] = np.nan
france_row['pop_est'] = np.nan
countries = countries.append(gpd.GeoDataFrame({**france_row}))
french_guiana_row = deepcopy(fake_row)
french_guiana_row['name'] = 'French Guiana'
french_guiana_row['geometry'] = french_guiana_poly
french_guiana_row['continent'] = 'South America'
french_guiana_row['iso_a3'] = 'GUF'
french_guiana_row['gdp_md_est'] = np.nan
french_guiana_row['pop_est'] = np.nan
countries = countries.append(gpd.GeoDataFrame({**french_guiana_row}))
# create Central America and Caribbean
for country in ['Haiti', 'Dominican Rep.', 'Bahamas',
                'Panama', 'Costa Rica', 'Nicaragua',
                'Honduras', 'El Salvador',
                'Guatemala', 'Belize',
                'Puerto Rico', 'Jamaica',
                'Cuba', 'Trinidad and Tobago']:
    countries.loc[countries['name'] == country, 'continent'] = 'South America'

# dissolve to continents
continents = countries.dissolve('continent')
# shorten continent names
continents.index = ['Africa', 'Antarctica', 'Asia',
                    'Europe', 'N. America', 'Oceania',
                    'C. & S. America\n& Caribbean']
# set continents' color palette
cont_palette = [sns.color_palette('bright')[i] for i in [4,8,6,9,3,1]]
# make the pink less "AAARGHH!!!" and the yellow less "BAAART!!!"
cont_palette[1] = tuple(np.array((250, 250, 75))/255)
cont_palette[2] = tuple(np.array((255, 150, 248))/255)
# set color for each row
cont_colors = []
cont_ct = 0
for cont in continents.index:
    if cont == 'Antarctica':
        cont_colors.append((0,0,0))
    else:
        cont_colors.append(cont_palette[cont_ct])
        cont_ct += 1
continents['color'] = cont_colors

# load Chapman raster data
chap_rast = rxr.open_rasterio('./chapman_all_Mg_C_ha_ca2km_EPSG_8857.tif',
                              masked=True,
                              cache=False,
                              chunks=(5, 5),
                             )
#chap_rast = chap_rast.rio.clip_box(map_minx, map_miny, map_maxx, map_maxy)


# load the Chapman SI datasets, then merge onto countries to make spatial
# (NOTE: I found no good metadata doc for her SI data,
#        but I compared to my own datasets and confirmed that units are Mg
#        (biomass and C) and ha)
chapman_potential = pd.read_csv(('./chapman_supplemental_data/'
                                  'summary_potential_standing.csv'))
chapman_potential = pd.merge(countries, chapman_potential,
                                 left_on='iso_a3', right_on='ISO_A3',
                                 how='left')

# convert densities to Mg C (from Mg biomass)
for c in ['density_crop', 'density_pasture']:
    chapman_potential[c] = chapman_potential[c]/2

# merge Roe et al. 2021 estimates onto this, also expressed in Mg C
subroe = roe.loc[:, ['ISO', 'agrofor_techcum', 'agrofor_techden',
                        'agrofor_feascum', 'agrofor_feasden']]
for col in subroe.columns[1:]:
    subroe[col] = subroe[col] * 1e6 * (12.0107/(2 * 15.999))
potential = pd.merge(chapman_potential, subroe,
                     left_on='iso_a3', right_on='ISO', how='left')


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
chapman_potential['cont'] = get_continents(chapman_potential)
potential['cont'] = get_continents(potential)
af_locs['cont'] = get_continents(af_locs)



# utils functions
# CODE ADAPTED FROM:
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
def hex2rgba(hex, alpha=255):
    rgb = [int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    rgba = tuple(rgb + [alpha])
    return rgba


# merge rosenstock and potential data
ndcs = []
ncs = []
namas = []
for i, row in potential.iterrows():
    iso3 = row['iso_a3']
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
potential['NDC'] = ndcs
potential['NC'] = ncs
potential['NAMA'] = namas


# add area-weighted average ag woody C density
# (i.e., pasture woody C weighted by pasture land area, same or crop, then sum)
assert np.allclose(potential.area_crop + potential.area_pasture,
                   potential.total_area, equal_nan=True)
potential['wt_avg_density'] = (((potential['area_crop'] *
                                 potential['density_crop']) +
                                (potential['area_pasture'] *
                                 potential['density_pasture'])) /
                               potential['total_area'])

# prep NDC contributions analysis
# subset and rename cols
ndc_contrib = ndc_contrib_raw.loc[:, ['iso3',
                    'CountryGeography',
                    'Trees in Agriculture Lands [Chapman]',
                    'Cost-effective Trees in Agriculture Lands [Chapman]',
                    'Reforestation (GROA)',
                    'Cost-effective Reforestation (GROA)',
                    'Nutrient Management',
                    'Cost-effective Nutrient Management',
                    'Optimal Grazing Intensity',
                    'Cost-effective Optimal Grazing Intensity',
                    'Grazing Legumes',
                    'Cost-effective Grazing Legumes',
                    '(Sharon et al) Emissions Reduction Target',
                    '(Sharon et al) NDC Reduction Percent',
                    '(Sharon et al) Reference Year Emissions Rate',
                    '(Sharon et al) New Annual Emisions after target reached',
                    'NDC Summary',
                   ]]
ndc_contrib.columns = ['iso3',
              'geo',
              'tia',
              'tia_ce',
              'refor',
              'refor_ce',
              'nut',
              'nut_ce',
              'opt_graz',
              'opt_graz_ce',
              'leg_graz',
              'leg_graz_ce',
              'targ',
              'red_pct',
              'ref_yr',
              'new_emis',
              'ndc_summ'
              ]

# calculate proportion of NDC goals that could be met by AF
ndc_contrib['pct_tia'] = ndc_contrib['tia']/ndc_contrib['targ'] * 100
ndc_contrib['pct_tia_ce'] = ndc_contrib['tia_ce']/ndc_contrib['targ'] * 100
ndc_contrib['pct_refor'] = ndc_contrib['refor']/ndc_contrib['targ'] * 100
ndc_contrib['pct_refor_ce'] = ndc_contrib['refor_ce']/ndc_contrib['targ'] * 100
# calculate trees in ag as pct of refor and of other ag NCS
ndc_contrib['tia_as_pct_refor'] = ndc_contrib['tia']/ndc_contrib['refor'] * 100
ndc_contrib['tia_as_pct_refor_ce'] = ndc_contrib['tia_ce']/ndc_contrib['refor_ce'] * 100
ndc_contrib['tia_as_pct_nut'] = ndc_contrib['tia']/ndc_contrib['nut'] * 100
ndc_contrib['tia_as_pct_nut_ce'] = ndc_contrib['tia_ce']/ndc_contrib['nut_ce'] * 100
ndc_contrib['tia_as_pct_opt_graz'] = ndc_contrib['tia']/ndc_contrib['opt_graz'] * 100
ndc_contrib['tia_as_pct_opt_graz_ce'] = ndc_contrib['tia_ce']/ndc_contrib['opt_graz_ce'] * 100
ndc_contrib['tia_as_pct_leg_graz'] = ndc_contrib['tia']/ndc_contrib['leg_graz'] * 100
ndc_contrib['tia_as_pct_leg_graz_ce'] = ndc_contrib['tia_ce']/ndc_contrib['leg_graz_ce'] * 100
# get rid of infs resulting from division by 0
ndc_contrib.replace(np.inf, np.nan, inplace=True)






##### ANALYSIS OF DENSITY IN NDC AND NON-NDC COUNTRIES
# prep data
data_for_figs = potential.loc[:, ['area_crop',
                                  'area_pasture',
                                  'total_area',
                                  'wt_avg_density',
                                  'density_crop',
                                  'density_pasture',
                                  'potential_crop',
                                  'potential_pasture',
                                  'total_potential',
                                  'total_biomass',
                                  'agrofor_techcum',
                                  'agrofor_feascum',
                                  'agrofor_feasden',
                                  'ISO_A3',
                                  'NDC',
                                  'NAME_EN',
                                  'cont',
                                  'geometry']]
data_for_figs = data_for_figs[np.invert(pd.isnull(data_for_figs['NDC']))]
data_for_figs = data_for_figs[data_for_figs['total_area']>0]
data_for_figs['NDC_num'] = data_for_figs['NDC']
data_for_figs['NDC'] = data_for_figs['NDC'].map(lambda x: {1:'yes', 0:'no'}[x])
data_for_figs['agrofor_feasden_Mgha'] = data_for_figs['agrofor_feasden']/1e6



##########################################################################
# PLOT 0: plot known AF locs and continent and country bounds over Chapman
##########################################################################


def format_map_axes(ax, bcax, max_tickval=None):
    """
    Function to custom format map images
    """
    # set 'ocean' color
    ax.set_facecolor('#ebf5f4')
    # bound the longitude (cuts off Hawaii, but no data there, and otherwise
    # makes plot nicer; also cuts out Antarctica)
    ax.set_xlim((map_minx, map_maxx))
    ax.set_ylim((map_miny, map_maxy))
    # get rid of axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    # get rid of ticks and ticklabels on map axes
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # format colorbar axes
    bcax.xaxis.set_ticks_position('bottom')
    bcax.xaxis.set_label_text('woody C density ($Mg\ C\ ha^{-1}$)',
                              fontdict={'fontsize': 18})
    bcax.tick_params(labelsize=14)
    #bcax.set_title(cbar_title_lookup[0], loc='left',
        #               fontdict={'fontsize':14, 'fontweight': 'bold'})
    # fix ticklabels on colorbars
    ticklocs = bcax.xaxis.get_ticklocs()
    curr_ticklabs = bcax.xaxis.get_ticklabels()
    new_ticklabs = []
    for n, loc in enumerate(ticklocs):
        if n%2 == 0:
            new_ticklabs.append('%i' % loc)
    ticklocs = [tl for n, tl in enumerate(ticklocs) if n%2 == 0]
    bcax.set_xticks(ticklocs, new_ticklabs)


map_pal = palettable.cmocean.sequential.Algae_3.mpl_colormap
soil_color = palettable.cmocean.sequential.Speed_20.mpl_colormap(2)
map_pal.set_under(soil_color)
map_pal.set_bad('#ffffff00')
fig_0 = plt.figure(figsize=(12, 9))
ax = fig_0.add_subplot(111)
# axes at bottom for colorbar
divider = make_axes_locatable(ax)
bcax = divider.append_axes("bottom", size="7%", pad=0.1)
# plot land underneath
countries.to_crs(8857).plot(color='#f7f7f7',
                            linewidth=0.25,
                            edgecolor='none',
                            ax=ax,
                            zorder=0,
                            )
# plot Chapman data
chap_rast.squeeze().plot.imshow(cmap=map_pal,
                                vmin=5,
                                vmax=40,
                                add_labels=False,
                                add_colorbar=True,
                                cbar_ax=bcax,
                                cbar_kwargs={'orientation': 'horizontal'},
                                ax=ax,
                                extend='max',
                                zorder=1,
                               )
# plot countries and continents
countries.to_crs(8857).plot(color='none',
                            linewidth=0.25,
                            edgecolor='white',
                            ax=ax,
                            zorder=2,
                            )
for cont in continents.index.unique():
    if cont not in ['Antarctica', 'Seven Seas']:
        data = continents.reset_index()[continents.reset_index()['index'] == cont]
        data.to_crs(8857).plot(color='none',
                               linewidth=1.5-(1.25*(cont == 'Antarctica')),
                               edgecolor=data['color'].values[0],
                               ax=ax,
                               zorder=2,
                               )

# add locations
ax.scatter(af_locs.to_crs(8857).centroid.x,
           af_locs.to_crs(8857).centroid.y,
           c='black',
           s=35,
           marker='X',
           edgecolor='white',
           linewidth=0.6,
           alpha=1,
           zorder=3,
          )

# fix the colorbar and set ticks and labels
patches_0to5 = [Rectangle(xy=(0, 0), width=5, height=1)]
p = PatchCollection(patches_0to5, alpha=1, color=soil_color, zorder=0)
bcax.add_collection(p)
bcax.set_xlim((0, bcax.get_xlim()[1]))
bcax.set_ylim((0, 1))
bcax.set_xticks([0,5,10,20,30,40], ['0', '5', '10', '20', '30', '40+'])
bcax.axvline(x=5, ymin=0, ymax=1, linewidth=2, color='black')
# call map-formatting fn
format_map_axes(ax, bcax)
# asjust spacing
fig_0.subplots_adjust(left=0,
                      bottom=0.075,
                      right=1,
                      top=1,
                     )
fig_0.show()

if save_it:
    fig_0.savefig('ag_woody_C_and_known_AF_locs.png',
                 dpi=dpi, orientation='landscape')



#############################################################
# PLOT 1: current and potential density, by continent and NDC
#############################################################


data_for_figs_long = data_for_figs.melt(id_vars=['NDC', 'cont',
                                                 'agrofor_feascum', 'NAME_EN'],
                                        value_vars=['wt_avg_density',
                                                    'agrofor_feasden_Mgha'],
                                        var_name='when',
                                        value_name='density')
# remap 'when' values
when_vals = {'wt_avg_density': 'current',
             'agrofor_feasden_Mgha': 'potential',
            }
data_for_figs_long['when'] = [when_vals[val] for val in data_for_figs_long['when']]
# get names of countries in nth quantile for total feasible mitigation by 2050


fig_1 = plt.figure(figsize=(9,8))
ax = fig_1.add_subplot(1,1,1)
cont_tot_width = 1.4
space_curr_potent = 0.05
space_NDC_no_yes = 0.1
space_cont = 0.2
box_width = (cont_tot_width -
             space_cont -
             space_NDC_no_yes -
             (2*space_curr_potent)) / 4
assert np.allclose(cont_tot_width,
                    ((box_width*4) +
                     (space_curr_potent*2) +
                     space_NDC_no_yes +
                     space_cont))
positions_per_cont = [box_width/2 + space_cont/2]
positions_per_cont.append(positions_per_cont[0] + space_curr_potent + box_width)
positions_per_cont.append(positions_per_cont[1] + space_NDC_no_yes + box_width)
positions_per_cont.append(positions_per_cont[2] + space_curr_potent + box_width)
positions = []
box_vecs = []
colors = []
alphas = []
widths = [box_width/2] * (4 * len(conts))
no_NDC_box_ll_xs = [0]
ct = 0
conts = data_for_figs_long.cont.unique()
for i, cont in enumerate(conts):
    data = data_for_figs_long[data_for_figs_long['cont'] == cont]
    positions.extend([p + (i*cont_tot_width) for p in positions_per_cont])
    if i < (len(conts)-1):
        no_NDC_box_ll_xs.append(no_NDC_box_ll_xs[i] + cont_tot_width)
    for j, NDC_val in enumerate(['no', 'yes']):
        subdata = data[data['NDC'] == NDC_val]
        for k, when in enumerate(data_for_figs_long['when'].unique()):
            subsubdata = subdata[subdata['when'] == when]
            box_vec = subsubdata['density'].values
            box_vec = box_vec[pd.notnull(box_vec)]
            box_vecs.append(box_vec)
            colors.append(continents.loc[cont]['color'])
            alphas.append(1 - (0.7*k))
            ct += 1
bp = ax.boxplot(x=box_vecs,
                positions=positions,
                notch=False,
                patch_artist=True,
                widths=widths,
                flierprops={'marker': '.'},
               )
for box, color, alpha in zip(bp['boxes'], colors, alphas):
    box.set_facecolor(color)
    box.set_alpha(alpha)
# add Xs in 'yes-NDC' columns for N. Am. and Oceania
ax.scatter([np.mean(positions[6:8]),
            np.mean(positions[14:16])],
           [0,0],
           marker='x',
           color='black',
           s=200,
           linewidth=4,
          )
# make potential boxes and their whiskers dotted-lined
for box in bp['boxes'][1::2]:
    box.set_linestyle(':')
for n, whisker_n_cap in enumerate(zip(bp['whiskers'], bp['caps'])):
    if (n in range(2, len(bp['whiskers'])+4, 4) or
        n in range(3, len(bp['whiskers'])+4, 4)):
        whisker, cap = whisker_n_cap
        whisker.set_linestyle(':')
        cap.set_linestyle(':')
ymin, ymax = ax.get_ylim()
patches=[]
for val in [p-(box_width/2)-(space_cont/2) for p in positions[::4]]:
    ax.axvline(val, ymin, ymax, linestyle='-', color='black', linewidth=0.5)
    ax.axvline(val+(0.5*cont_tot_width), ymin, ymax, linestyle='-',
                   color='black', linewidth=0.5)
    patch = Rectangle(xy=(val, ax.get_ylim()[0]),
                      width=0.5*cont_tot_width,
                      height=np.diff(ax.get_ylim()),
                     )
    patches.append(patch)
p = PatchCollection(patches, alpha=0.1, color='black', zorder=0)
ax.add_collection(p)
ax.set_xlim(positions[0]-(box_width/2)-(space_cont/2),
            positions[-1]+(box_width/2)+(space_cont/2))
ax.set_xticks([p+(box_width/2)+(space_NDC_no_yes/2) for p in positions[1::4]],
              conts)
ax.tick_params(labelsize=12)
ax.set_xlabel('continent', fontdict={'fontsize':18})
ax.set_ylabel('woody C density ($Mg\ C\ ha^{-1}$)', fontdict={'fontsize':18})
fig_1.subplots_adjust(left=0.08,
                      bottom=0.1,
                      right=0.98,
                      top=0.98,
                     )
fig_1.show()
if save_it:
    fig_1.savefig('current_and_potential_boxplots.png', dpi=700)



# t-test of significant diff between NDC and non-NDC groups
res = ttest_ind(data_for_figs[data_for_figs.NDC_num==1]['wt_avg_density'],
                data_for_figs[data_for_figs.NDC_num==0]['wt_avg_density'],
               nan_policy='omit')
print(('\n\nt-test of sig. diff. between woody C ag-land density in NDC and '
       'non-NDC countries:\n\tt-stat: %0.3f\n\tp-value: '
       '%0.5f') % (res.statistic, res.pvalue))



############################################
# PLOT 2: current woody C density vs HDI/GDP
############################################


def scale_markersizes(vals, min_marksize=20, max_marksize=250, transform=None):
    if transform == 'log':
        vals = np.log10(vals)
    # NOTE: perhaps makes most sense to scale with sqrt, since mpl expresses
    #       scatterploint marker size in pt^^2, i.e., sqrt of point area;
    #       however I don't really know anything about the science around
    #       perception of size in plots...
    elif transform == 'sqrt':
        vals = np.sqrt(vals)
    vals_0to1 = (vals - np.min(vals))/(np.max(vals) - np.min(vals))
    scaled_vals = (vals_0to1 * (max_marksize - min_marksize)) + min_marksize
    return scaled_vals


#is current woody C density roughly correlated with GDP? HDI?
# NOTE: GDP data from: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
# NOTE: HDI data from: https://hdr.undp.org/en/data


def scale_var(var):
    return (var - np.min(var))/(np.max(var) - np.min(var))


data_for_figs['pct_below'] = np.clip(((data_for_figs['agrofor_feascum'] -
                                     data_for_figs['total_biomass']/2)/(
                data_for_figs['agrofor_feascum']))*100, a_min = 0, a_max = None)
#gdp = pd.read_csv('./API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3731268.csv', skiprows=4)
#gdp_pcap = pd.read_csv('./API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3731360.csv', skiprows=4)
hdi = pd.read_csv('Human Development Index (HDI)_w_ISO3_2000.csv')
data_w_hdi = pd.merge(data_for_figs, hdi.loc[:, ['Country Code', '2000']],
                      left_on='ISO_A3', right_on='Country Code')
data_w_hdi['hdi'] = data_w_hdi['2000']
fig_2 = plt.figure(figsize=(10,8.5))
ax = fig_2.add_subplot(111)
#data_w_gdp['log_gdp'] = np.log(data_w_gdp['2000'])
data_w_hdi['log_dens'] = np.log(data_w_hdi['wt_avg_density'])
sizes = scale_markersizes(data_w_hdi['agrofor_feascum'],
                          min_marksize=min_scattersize,
                          max_marksize=max_scattersize,
                          transform='sqrt')
data_w_hdi['sizes'] = sizes
sns.scatterplot(x='hdi', y='log_dens', hue='cont', data=data_w_hdi, ax=ax,
                size='sizes', sizes=(min_scattersize, max_scattersize),
                style='NDC', style_order=('yes', 'no'),
                palette=cont_palette[:len(data_for_figs['cont'].unique())],
                edgecolor='black', linewidth=0.5, alpha=0.8, legend=False)
sns.regplot(x='hdi', y='log_dens', data=data_w_hdi, ax=ax, scatter=False)
data_for_test = data_w_hdi.loc[:, ['hdi', 'log_dens']].dropna(how='any')
r, p = stats.pearsonr(data_for_test['hdi'], data_for_test['log_dens'])
ax.text(0.26, -5.9, '$R^2=%0.2f$' % r**2, color='red', fontstyle='italic')
ax.text(0.26, -6.1, '$p=%0.4f$' % p, color='red', fontstyle='italic')
for i, row in data_w_hdi.iterrows():
    country = row['NAME_EN']
    # manually shorten some countries
    if country == 'United States of America':
        country = 'USA'
    elif country == "People's Republic of China":
        country = 'China'
    if (row['log_dens'] < np.nanpercentile(data_w_hdi['log_dens'], 10) and
        pd.notnull(row['agrofor_feascum'])):
        ax.text(row['hdi'], row['log_dens'], country,
                color='gray', rotation=20, size=9)
    if row['agrofor_feascum'] > np.nanpercentile(data_w_hdi['agrofor_feascum'],
                                                 90):
        ax.text(row['hdi'], row['log_dens'], country,
                color='black', rotation=20, size=9)
ax.set_xlabel('human development index (HDI; yr. 2000)',
              fontdict={'fontsize': 14})
ax.set_ylabel('log average ag. woody C density (yr. 2000)',
              fontdict={'fontsize': 14})
ax.set_xlim(0.25, 0.96)
ax.set_ylim(-6.2, 3)
ax.tick_params(labelsize=12)
# custom legend at the side
feascum_data = data_w_hdi['agrofor_feascum']
vals = np.linspace(np.quantile(np.sqrt(feascum_data), 0.05),
                   np.quantile(np.sqrt(feascum_data), 0.95), 5)**2
vals = [round(val, -(len(str(int(val)))-2)) for val in vals]
sizes = scale_markersizes(vals)
legend_elements = []
for val, size in zip(vals, sizes):
    label = '$%s.%s\ ×\ 10^{%i}$' % (str(val)[0],
                                     str(val)[1],
                                     len(str(val))-1)
    # add circle and X markers
    element_x = Line2D([0], [0],
                     marker='X',
                     color='none',
                     markeredgecolor='black',
                     label='',
                     markerfacecolor='none',
                     markersize=np.sqrt(size))
    element_o = Line2D([0], [0],
                     marker='o',
                     color='none',
                     markeredgecolor='black',
                     label=label,
                     markerfacecolor='none',
                     markersize=np.sqrt(size))
    legend_elements.append(element_x)
    legend_elements.append(element_o)
    # add spacing element
    if val < vals[-1]:
        element_blank = Line2D([0], [0],
                               marker='.',
                               color='none',
                               markeredgecolor='none',
                               label='',
                               alpha=0,
                               markerfacecolor='none',
                               markersize=np.sqrt(size))
        legend_elements.append(element_blank)
lgd_title = 'total mitigation\npotential by\n2050 ($Mg\ C$)'
lgd = ax.legend(handles=legend_elements,
                bbox_to_anchor=(1.01, 0.9),
                prop={'size': 12},
                title=lgd_title,
                title_fontsize=14,
                fancybox=True,
               )
fig_2.subplots_adjust(left=0.08,
                      bottom=0.08,
                      right=0.81,
                      top=0.98,
                     )

fig_2.show()
if save_it:
    fig_2.savefig('woody_C_vs_HDI_scatter.png', dpi=700)
