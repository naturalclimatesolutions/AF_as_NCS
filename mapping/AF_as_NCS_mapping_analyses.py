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

# load the Chapman SI datasets, then merge onto countries to make spatial
# (NOTE: I found no good metadata doc for her SI data,
#        but I compared to my own datasets and confirmed that units are Mg
#        (biomass and C) and ha)
chapman_potential = pd.read_csv(('./chapman_potential_data/'
                                  'summary_potential_standing.csv'))
chapman_potential = pd.merge(countries, chapman_potential,
                                 left_on='iso_a3', right_on='ISO_A3',
                                 how='left')

# convert densities to Mg C (from Mg biomass)
for c in ['density_crop', 'density_pasture']:
    chapman_potential[c] = chapman_potential[c]/2


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
for i, row in chapman_potential.iterrows():
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
chapman_potential['NDC'] = ndcs
chapman_potential['NC'] = ncs
chapman_potential['NAMA'] = namas


# add country-aggregated ag land area col to country-aggregated ag woody C,
# then get area-normalized woody C
chapman_ag_area_agg['ha'] = chapman_ag_area_agg['sum']/10_000
chapman_C_agg['ha_ag_land'] = chapman_ag_area_agg['ha']
chapman_C_agg['Mg_C_per_ha'] = chapman_C_agg['sum']/chapman_C_agg['ha_ag_land']


##### ANALYSIS OF DENSITY IN NDC AND NON-NDC COUNTRIES
# prep data
data_for_figs = chapman_potential.loc[:, ['area_crop',
                                      'area_pasture',
                                      'total_area',
                                      'density_crop',
                                      'density_pasture',
                                      'potential_crop',
                                      'potential_pasture',
                                      'total_potential',
                                      'total_biomass',
                                      'NDC',
                                      'NAME_EN',
                                      'cont',
                                      'geometry']]

data_for_figs = data_for_figs[np.invert(pd.isnull(data_for_figs['NDC']))]
data_for_figs = data_for_figs[data_for_figs['total_area']>0]
data_for_figs['NDC_num'] = data_for_figs['NDC']
data_for_figs['NDC'] = data_for_figs['NDC'].map(lambda x: {1:'yes', 0:'no'}[x])



# plot dists of mean woody C/m^2 land area for countries w/ and w/out AF in NDCs
fig_1 = plt.figure()
#fig_1 = plt.figure(dpi=dpi,
#                   figsize=(fig1_width, fig1_height))
#gs = fig.add_gridspec(nrows=9, ncols=5,
#                      height_ratios=[1,1,1,1,1.5,1,1,1,1])
fig_1.suptitle(('average crop and pasture woody C density in\n'
                'countries that do and do not mention AF in NDCs'))
for i, col in enumerate(['density_crop', 'density_pasture']):
    ax = fig_1.add_subplot(1,2,i+1)
    ax.set_title(col, fontdict={'fontsize': 15})
    sns.set_theme(style="whitegrid")
    ax = sns.boxenplot(ax=ax,
                        x='NDC',
                        y=col,
                        data=data_for_figs,
                        #inner='box',
                        #orient='h',
                        palette=['#d1caa7', '#b4edb5'],
                       )
    cont_lookup = dict(zip(data_for_figs.cont.unique(),
                           range(len(data_for_figs.cont.unique()))))
    rev_cont_lookup = dict(zip(range(len(data_for_figs.cont.unique())),
                           data_for_figs.cont.unique()))
    x_offset_unit = 0.06
    scat = ax.scatter(data_for_figs['NDC_num'] + (x_offset_unit *
                                       (2+data_for_figs['cont'].map(lambda cont:
                                                            cont_lookup[cont]))),
               data_for_figs[col],
               c = data_for_figs['cont'].map(lambda cont: cont_lookup[cont]),
               s = 60,
               cmap='Set3',
               alpha=0.75,
               edgecolor='black',
               linewidth=0.5,
               label=data_for_figs.cont,
              )
    #leg = ax.legend(*scat.legend_elements(), title='Continent')
    ax.set_xlabel('mention agroforestry in NDC?')
    ax.set_ylabel('mean %s woody C density\n($Mg C / ha$)' % col.split('_')[1])

    for cont in data_for_figs.cont.unique():
        for NDC_status in range(2):
            x = NDC_status + (x_offset_unit * (2+cont_lookup[cont]))
            y = -2
            ax.text(x-0.005, y, cont, fontdict={'fontsize': 7,
                                           'weight': 'bold',
                                           'rotation': 'vertical'}, alpha=0.9)
            if len(data_for_figs.loc[(data_for_figs.cont == cont) &
                                 (data_for_figs.NDC_num == NDC_status)]) > 0:
                ax.plot([x, x],
                        [0, np.max(data_for_figs.loc[(data_for_figs.cont == cont) &
                                             (data_for_figs.NDC_num == NDC_status),
                                             col])],
                        color='black', linewidth=0.2, alpha=1)
            else:
                ax.plot([x,x], [0, 0.01], color='black', linewidth=0.2, alpha=1)


    # t-test of significant diff between NDC and non-NDC groups
    res = ttest_ind(data_for_figs[data_for_figs.NDC_num==1][col],
                    data_for_figs[data_for_figs.NDC_num==0][col],
                   nan_policy='omit')
    print(('\n\nt-test of sig. diff. between woody C ag-land density in NDC and '
           'non-NDC countries:\n\tt-stat: %0.3f\n\tp-value: '
           '%0.3f') % (res.statistic, res.pvalue))

    ax.text(-0.45, 0.95*data_for_figs[col].max(),
            't-stat: %0.3f\np-value: %0.3f' % (res.statistic, res.pvalue),
            fontdict={'fontsize': 10, 'fontstyle': 'italic'})
    ax.set_ylim([-2.5, 1.05*data_for_figs[col].max()])

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


#### MAPS OF CURRENT CARBON DENSITY
# plot AF sites on top of area-normalized ag woody C density
# and on top of percentage of potential C realized,
# with separate colormaps for NDC and non-NDC countries

# set colormaps
cmaplist_NDC = []
for val, color in zip([0,1], ['#c9e4ff', '#003873']):
    cmaplist_NDC.append((val, color))
cmap_NDC = LinearSegmentedColormap.from_list("custom", cmaplist_NDC)
cmaplist_nonNDC = []
for val, color in zip([0,1], ['#ffd9e6', '#730027']):
    cmaplist_nonNDC.append((val, color))
cmap_nonNDC = LinearSegmentedColormap.from_list("custom", cmaplist_nonNDC)
cmaps = {0: cmap_nonNDC,
         #1: 'Greens',
         1: cmap_NDC,
        }
# get the max value to display on the choropleth colorbars
# (max val in the data, rounded up to nearest multiple of 5)
max_cbar_val = max(int(data_for_figs.density_crop.max()),
                   int(data_for_figs.density_pasture.max()))
max_cbar_val = max_cbar_val + (max_cbar_val % 5)

# plot NDC and non-NDC countries' woody C density separately
cbar_title_lookup = {0: 'AF not in NDC', 1: 'AF in NDC'}


for i, col in enumerate(['density_crop', 'density_pasture']):
    fig2 = plt.figure()
    fig2.suptitle(('average woody C density in %s, with '
                   'known agroforestry locations') % col.split('_')[1])


    # plot it
    ax = fig2.add_subplot(111)
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("bottom", size="5%", pad=0.1)
    rcax = divider.append_axes("right", size="5%", pad=0.1)
    lcax = divider.append_axes("left", size="5%", pad=0.1)
    cax_dict = {0: lcax, 1:rcax}


    for NDC_status in range(2):
        subdf = data_for_figs[data_for_figs.NDC_num == NDC_status]
        if NDC_status == 1:
            edgecolor = 'black'
            linewidth = 0.25
        else:
            edgecolor='black'
            linewidth=0.25
        map = subdf.plot(col,
                         ax=ax,
                         vmin=0,
                         vmax=max_cbar_val,
                         cmap=cmaps[NDC_status],
                         edgecolor=edgecolor,
                         linewidth=linewidth,
                         legend=True,
                         legend_kwds={'label': 'Mg woody C/ha',
                                      'orientation': "vertical"},
                         cax=cax_dict[NDC_status])

    rcax.set_title(cbar_title_lookup[1])
    lcax.set_title(cbar_title_lookup[0])
    lcax.yaxis.set_ticks_position('left')
    lcax.yaxis.set_label_position('left')

    # outline missing country boundaries
    chapman_potential[pd.isnull(chapman_potential.NDC)].plot(edgecolor='black',
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
        fig2.savefig(('%s_woody_C_density_and_known_AF_locs'
                      '.png') % col.split('_')[1],
                     dpi=dpi, orientation='landscape')



#### MAP OF DEFICIT

# recast potential as % deficit,
# TODO: FIGURE OUT IF FOR SOME REASON TOTAL POTENTIAL IS EXPRESSED IN Mg C
#       INSTEAD OF Mg BIOMASS LIKE EVERYTHING ELSE, BECAUSE IF SO I NEED TO
#       CORRECT FOR THAT!
# and cap at 0% deficit
# (the countries that have a potential number lower than
#  the current number are just the really high-density ones in Africa
#  and small Caribbean Islands (as well as Kosovo, Turkmenistan, and Suriname)
#  so that makes sense because all the data were used together to set 'potential'
data_for_figs['deficit'] = np.clip(((data_for_figs['total_potential'] -
                                     data_for_figs['total_biomass'])/(
                data_for_figs['total_potential']))*100, a_min = 0, a_max = None)

max_cbar_val = 100
min_cbar_val = 0

# plot NDC and non-NDC countries' woody C density separately
cbar_title_lookup = {0: 'AF not in NDC', 1: 'AF in NDC'}


fig3 = plt.figure()
fig3.suptitle('deficit in potential ag woody C density in current stocks (%%)')

# plot it
ax = fig3.add_subplot(111)
divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.1)
rcax = divider.append_axes("right", size="5%", pad=0.1)
lcax = divider.append_axes("left", size="5%", pad=0.1)
cax_dict = {0: lcax, 1:rcax}


for NDC_status in range(2):
    subdf = data_for_figs[data_for_figs.NDC_num == NDC_status]
    if NDC_status == 1:
        edgecolor = 'black'
        linewidth = 0.25
    else:
        edgecolor='black'
        linewidth=0.25
    map = subdf.plot('deficit',
                     ax=ax,
                     vmin=0,
                     vmax=max_cbar_val,
                     cmap=cmaps[NDC_status],
                     edgecolor=edgecolor,
                     linewidth=linewidth,
                     legend=True,
                     legend_kwds={'label': 'percent deficit',
                                  'orientation': "vertical"},
                     cax=cax_dict[NDC_status])

rcax.set_title(cbar_title_lookup[1])
lcax.set_title(cbar_title_lookup[0])
lcax.yaxis.set_ticks_position('left')
lcax.yaxis.set_label_position('left')

# TODO: FIGURE OUT WHY POTENTIAL_CROP OR POTENTIAL_PASTURE IS NAN
#       FOR SOME COUNTRIES (it makes total_potential nan too);
#       I could some the potential value that she does have with the current
#       for the other value to get a low estimate of total potential,
#       but that's probably more misleading than just omitting those countries,
#       as I do now...
# outline countries with missing potential values
data_for_figs[pd.isnull(data_for_figs.total_potential)].plot(edgecolor='black',
                                                             facecolor='white',
                                                             linewidth=0.25,
                                                             ax=ax)

# outline missing country boundaries
chapman_potential[pd.isnull(chapman_potential.NDC)].plot(edgecolor='black',
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

fig3.show()


if save_it:
    fig3.savefig('woody_C_deficit.png',
                  dpi=dpi, orientation='landscape')

