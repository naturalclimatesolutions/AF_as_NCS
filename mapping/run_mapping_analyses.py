import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from shapely.geometry import Polygon
from scipy.stats import ttest_ind
from scipy import stats
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
    countries.loc[countries.name == country, 'continent'] = 'South America'

# dissolve to continents
continents = countries.dissolve('continent')

# shorten continent names
continents.index = ['Af.', 'Antarctica', 'Asia',
                    'Eur.', 'N. Am.', 'Oceania', 'Seven seas', 'C./S. Am.']

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
                        palette=['#877f78', '#3cb55e'],
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
            ax.plot([x, x],
                    [0, 25],
                    color='black', linewidth=0.2, alpha=1)


    # t-test of significant diff between NDC and non-NDC groups
    res = ttest_ind(data_for_figs[data_for_figs.NDC_num==1][col],
                    data_for_figs[data_for_figs.NDC_num==0][col],
                   nan_policy='omit')
    print(('\n\nt-test of sig. diff. between woody C ag-land density in NDC and '
           'non-NDC countries:\n\tt-stat: %0.3f\n\tp-value: '
           '%0.5f') % (res.statistic, res.pvalue))

    ax.text(-0.45, 0.95*data_for_figs[col].max(),
            't-stat: %0.3f\np-value: %0.5f' % (res.statistic, res.pvalue),
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


def format_map_axes(ax, tcax, bcax, map_type, max_tickval=None):
    """
    Function to custom format all the map images
    """
    assert map_type in ['current', 'pct_below', 'pct_NDC']
    # bound the longitude (cuts off Hawaii, but no data there, and otherwise
    # makes plot nicer; also cuts out Antarctica)
    ax.set_xlim((-13700000, 16500000))
    ax.set_ylim((-7000000, 8550000))
    # get rid of ticks and ticklabels on map axes
    #ax.set_xlabel('$^{\circ} lon.')
    #ax.set_ylabel('$^{\circ} lat.')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # format colorbar axes
    tcax.xaxis.set_ticks_position('top')
    bcax.xaxis.set_ticks_position('bottom')
    tcax.xaxis.set_label_position('top')
    tcax.xaxis.set_label_text(tcax.xaxis.get_label_text(), fontdict={'fontsize': 18})
    bcax.xaxis.set_label_text(tcax.xaxis.get_label_text(), fontdict={'fontsize': 18})
    tcax.tick_params(labelsize=14)
    bcax.tick_params(labelsize=14)
    #tcax.set_xticks(tcax.get_xticks())
    #tcax.set_xticklabels(tcax.get_xticklabels(), fontdict={'fontsize': 12})
    #tcax.set_title(cbar_title_lookup[1], loc='left',
        #               fontdict={'fontsize':14, 'fontweight': 'bold'})
    #bcax.set_title(cbar_title_lookup[0], loc='left',
        #               fontdict={'fontsize':14, 'fontweight': 'bold'})
    # fix ticklabels on colorbars
    for cax in [tcax, bcax]:
        ticklocs = cax.xaxis.get_ticklocs()
        curr_ticklabs = cax.xaxis.get_ticklabels()
        new_ticklabs = []
        for n, loc in enumerate(ticklocs):
            if map_type == 'pct_NDC':
                new_ticklabs.append(('$≥$ '* (loc >= max_tickval)) + '%i%%' % int(loc))
            elif map_type == 'pct_below':
                new_ticklabs.append('%i%%' % int(loc))
            elif map_type == 'current':
                if n%2 == 0:
                    new_ticklabs.append('%i' % loc)
        if map_type == 'current':
            ticklocs = [tl for n, tl in enumerate(ticklocs) if n%2 == 0]
        cax.set_xticks(ticklocs, new_ticklabs)



#### MAPS OF CURRENT CARBON DENSITY
# plot AF sites on top of area-normalized ag woody C density
# and on top of percentage of potential C realized,
# with separate colormaps for NDC and non-NDC countries

# set colormaps
cmaplist_NDC = []
for val, color in zip([0,1], ['#e4ffd9', '#1f6e00']):
    cmaplist_NDC.append((val, color))
cmap_NDC = LinearSegmentedColormap.from_list("custom", cmaplist_NDC)
cmaplist_nonNDC = []
for val, color in zip([0,1], ['#ebeae6', '#5e534b']):
    cmaplist_nonNDC.append((val, color))
cmap_nonNDC = LinearSegmentedColormap.from_list("custom", cmaplist_nonNDC)
cmaps = {0: cmap_nonNDC,
         #1: 'Greens',
         1: cmap_NDC,
        }
# get the max value to display on the choropleth colorbars
# (max val in the data, rounded up to nearest multiple of 5)
max_cbar_val = int(data_for_figs.wt_avg_density.max())
max_cbar_val = max_cbar_val + (5*((max_cbar_val // 5)+1) - max_cbar_val)
# plot NDC and non-NDC countries' woody C density separately
cbar_title_lookup = {0: 'AF not in NDC', 1: 'AF in NDC'}
for i, col in enumerate(['wt_avg_density']):
    fig2 = plt.figure()
    #fig2.suptitle(('average woody C density in ag lands, with '
    #               'known agroforestry locations'))
    # plot it
    ax = fig2.add_subplot(111)

    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("bottom", size="5%", pad=0.1)
    tcax = divider.append_axes("top", size="7%", pad=0.1)
    bcax = divider.append_axes("bottom", size="7%", pad=0.1)
    cax_dict = {0: bcax, 1:tcax}
    for NDC_status in range(2):
        subdf = data_for_figs[data_for_figs.NDC_num == NDC_status]
        if NDC_status == 1:
            edgecolor = 'black'
            linewidth = 0.75
        else:
            edgecolor='black'
            linewidth=0.75
        map = subdf.to_crs(8857).plot(col,
                         ax=ax,
                         vmin=0,
                         vmax=max_cbar_val,
                         cmap=cmaps[NDC_status],
                         edgecolor=edgecolor,
                         linewidth=linewidth,
                         legend=True,
                         legend_kwds={'label': 'Mg woody C/ha',
                                      'orientation': "horizontal"},
                         cax=cax_dict[NDC_status])
    # outline missing country boundaries
    potential[pd.isnull(potential.NDC)].to_crs(8857).plot(edgecolor='black',
                                            facecolor='white',
                                            linewidth=0.25,
                                            ax=ax)
    # add locations
    ax.scatter(af_locs.to_crs(8857).centroid.x,
               af_locs.to_crs(8857).centroid.y,
               c='black',
               s=9,
               alpha=1)
    format_map_axes(ax, tcax, bcax, 'current')
    fig2.show()


    if save_it:
        fig2.savefig('woody_C_density_and_known_AF_locs.png',
                     dpi=dpi, orientation='landscape')



#### MAP PCT BELOW POTENTIAL STOCKING

# recast potential as % below potential,
# and cap at 0% below potential
# (NOTE: combines Roe feasible potential numbers and Chapman current numbers)
# (the countries that have a potential number lower than
#  the current number are just the really high-density ones in Africa
#  and small Caribbean Islands (as well as Kosovo, Turkmenistan, and Suriname)
#  so that makes sense because all the data were used together to set 'potential'
data_for_figs['abs_below'] = np.clip(((data_for_figs['agrofor_feascum'] -
                                     data_for_figs['total_biomass']/2)),
                                       a_min = 0, a_max = None)
#data_for_figs['abs_below'] = np.clip(((data_for_figs['total_potential'] -
                                     #data_for_figs['total_biomass'])),
                                       #a_min = 0, a_max = None)
#data_for_figs['pct_below'] = np.clip(((data_for_figs['total_potential'] -
#                                     data_for_figs['total_biomass'])/(
                #data_for_figs['total_potential']))*100, a_min = 0, a_max = None)
data_for_figs['pct_below'] = np.clip(((data_for_figs['agrofor_feascum'] -
                                     data_for_figs['total_biomass']/2)/(
                data_for_figs['agrofor_feascum']))*100, a_min = 0, a_max = None)

max_cbar_val = 100
min_cbar_val = 0

# plot NDC and non-NDC countries' woody C density separately
cbar_title_lookup = {0: 'AF not in NDC', 1: 'AF in NDC'}


fig3 = plt.figure()
fig3.suptitle('current percent below feasible ag woody C density')

# plot it
ax = fig3.add_subplot(111)
divider = make_axes_locatable(ax)
#cax = divider.append_axes("bottom", size="5%", pad=0.1)
tcax = divider.append_axes("top", size="5%", pad=0.1)
bcax = divider.append_axes("bottom", size="5%", pad=0.1)
cax_dict = {0: bcax, 1:tcax}


for NDC_status in range(2):
    subdf = data_for_figs[data_for_figs.NDC_num == NDC_status]
    if NDC_status == 1:
        edgecolor = 'black'
        linewidth = 0.75
    else:
        edgecolor='black'
        linewidth=0.75
    map = subdf.to_crs(8857).plot('pct_below',
                     ax=ax,
                     vmin=0,
                     vmax=max_cbar_val,
                     cmap=cmaps[NDC_status],
                     edgecolor=edgecolor,
                     linewidth=linewidth,
                     legend=True,
                     legend_kwds={'label': 'percent below potential',
                                  'orientation': "horizontal"},
                     cax=cax_dict[NDC_status])

# TODO: FIGURE OUT WHY POTENTIAL_CROP OR POTENTIAL_PASTURE IS NAN
#       FOR SOME COUNTRIES (it makes total_potential nan too);
#       I could sum the potential value that she does have with the current
#       for the other value to get a low estimate of total potential,
#       but that's probably more misleading than just omitting those countries,
#       as I do now...
# outline countries with missing potential values
data_for_figs[pd.isnull(data_for_figs.agrofor_feascum)].to_crs(8857).plot(
                                                    edgecolor='black',
                                                    facecolor='white',
                                                    linewidth=0.25,
                                                    ax=ax)

# outline missing country boundaries
potential[pd.isnull(potential.NDC)].to_crs(8857).plot(edgecolor='black',
                                                      facecolor='white',
                                                      linewidth=0.25,
                                                      ax=ax)
format_map_axes(ax, tcax, bcax, 'pct_below', max_tickval=100)
fig3.show()


if save_it:
    fig3.savefig('woody_C_pct_below.png',
                  dpi=dpi, orientation='landscape')





# SCATTER TOTAL POTENTIAL VS PCT BELOW

# get count of points in each country poly
dfsjoin = gpd.sjoin(data_for_figs, af_locs)
dfsjoin['count'] = 1
counts = dfsjoin.groupby(['NAME_EN']).sum()['count']

# add a point-density col in data_for_figs
data_for_figs['count'] = 0
for cntry in counts.index:
    data_for_figs.loc[data_for_figs['NAME_EN'] == cntry, 'count'] = counts[cntry]
data_for_figs['pt_density'] = data_for_figs['count']/data_for_figs['total_area']

# determine breakpoints between low, mod, and high avg_density
lo, md, hi = np.nanpercentile(data_for_figs['wt_avg_density'], [10, 50, 90])

colors = {'AF not in NDC':mpl.colors.hex2color('#1f6e00'),
          'AF in NDC': mpl.colors.hex2color('#5e534b')
         }

# rank the pt counts
data_for_figs['count_rank'] = data_for_figs['count'].rank(method='dense')
fig5, ax = plt.subplots(1,1)
sns.scatterplot(x=data_for_figs['pct_below'],
                # TODO: DECIDE IF I'M CORRECT THAT I NEED TO /2 HERE TO GET Mg C
                y=data_for_figs['agrofor_feascum']/2,
                style=['o' if c>0 else 'x' for c in data_for_figs['count']],
                hue=data_for_figs['NDC_num'],
                palette=['#5e534b', '#1f6e00'],
                size=data_for_figs['count_rank'],
                sizes=(50, 300),
                alpha=0.4,
                ax=ax)
legend_elements = [Line2D([0], [0],
                          marker='o',
                          color=c,
                          label=l,
                          linewidth=0,
                          markersize=6,
                          alpha=0.4) for l, c in colors.items()]
legend_elements = legend_elements + [Line2D([0], [0], color='black', label='')]
legend_elements = legend_elements + [Line2D([0], [0],
                                            marker=m,
                                            color='black',
                                            alpha=0.5,
                                            label=l,
                                            linewidth=0,
                                            markersize=6,
                                           ) for l, m in [('no known studies', 'x'),
                                                    ('known studies', 'o')]]
legend_elements = legend_elements + [Line2D([0], [0], color='black', label='')]
legend_elements = legend_elements + [Line2D([0], [0],
                                            marker='o',
                                            color='black',
                                            alpha=0.5,
                                            label=l,
                                            linewidth=0,
                                            markersize=s
                                           ) for l, s in [('few studies', 6),
                                                        ('many studies', 15)]]
ax.legend(handles=legend_elements, loc='upper left')
qtile = np.nanpercentile(data_for_figs['agrofor_feascum'], 95)
for i, row in data_for_figs.iterrows():
    if row['agrofor_feascum']>qtile:
        ax.text(row['pct_below'],
                (row['agrofor_feascum']/2) + 0.005e8,
                 row['NAME_EN'],
                 fontdict={'fontsize':8},
                )
ax.set_xlabel('% below total potential woody C',
              fontdict={'fontsize': 20})
ax.set_ylabel('total potential woody C (Mg)',
              fontdict={'fontsize': 20})
fig5.show()



############################
# NDC contributions analysis

# prep NDC contributions data
df_hist_tia = ndc_contrib.loc[:, ['geo', 'pct_tia', 'pct_tia_ce']]
df_hist_tia.columns = ['country', 'max_potential', 'cost_effective']
df_hist_tia = pd.melt(df_hist_tia, id_vars=['country'],
                      value_vars=['max_potential', 'cost_effective'])
df_hist_tia['NCS'] = 'agroforestry'
df_hist_refor = ndc_contrib.loc[:, ['geo', 'pct_refor', 'pct_refor_ce']]
df_hist_refor.columns = ['country', 'max_potential', 'cost_effective']
df_hist_refor = pd.melt(df_hist_refor, id_vars=['country'],
                        value_vars=['max_potential', 'cost_effective'])
df_hist_refor['NCS'] = 'reforestation'
df_hist = pd.concat((df_hist_tia, df_hist_refor), axis=0)
df_hist.columns = ['country', 'estimate_type', 'percent_NDC_target', 'NCS']


# merge onto countries
ndc_contrib_map = pd.merge(countries, ndc_contrib,
                           left_on='iso_a3', right_on='iso3', how='outer')
ndc_contrib_map = pd.merge(ndc_contrib_map, data_for_figs.loc[:, ['ISO_A3', 'NDC_num']],
         left_on='iso_a3', right_on='ISO_A3', how='outer')

fig_map, ax = plt.subplots(1, 1)
fig_map.suptitle('percent of NDC targets achievable by agroforestry (Baruch-Mordo et al. 2018)')
divider = make_axes_locatable(ax)
tcax = divider.append_axes("top", size="5%", pad=0.1)
bcax = divider.append_axes("bottom", size="5%", pad=0.1)
cax_dict = {0: bcax, 1:tcax}

countries.to_crs(8857).plot(facecolor='none',
           edgecolor='black',
           linewidth=0.25,
           ax=ax)
max_pct=30
for NDC_status in range(2):
    subdf = ndc_contrib_map[ndc_contrib_map.NDC_num == NDC_status]
    map = subdf.to_crs(8857).plot('pct_tia_ce',
                     ax=ax,
                     vmin=0,
                     vmax=max_pct,
                     cmap=cmaps[NDC_status],
                     edgecolor=edgecolor,
                     linewidth=linewidth,
                     legend=True,
                     legend_kwds={'label': 'percent NDC target achievable',
                                  'orientation': "horizontal"},
                     cax=cax_dict[NDC_status])
format_map_axes(ax, tcax, bcax, 'pct_NDC', max_tickval=max_pct)
ax.set_title('cost-effective mitigation')
# fix cbar ticklabels
fig_map.show()

if save_it:
    fig_map.savefig('NDC_contributions_map.png',
                    dpi=dpi, orientation='portrait')


# PERCENT OF NDC VS TOTAL NDC SIZE
# rank the pt counts
fig6, ax = plt.subplots(1,1)
ndc_contrib_for_scat = pd.merge(ndc_contrib, data_for_figs,
                                left_on='iso3', right_on='ISO_A3',
                                how='outer')
sns.scatterplot(x='pct_tia_ce',
                y='targ',
                #style=['o' if c>0 else 'x' for c in data_for_figs['count']],
                hue='NDC_num',
                palette=['#5e534b', '#1f6e00'],
                size='red_pct',
                sizes=(50, 300),
                alpha=0.4,
                data=ndc_contrib_for_scat,
                ax=ax)
legend_elements = [Line2D([0], [0],
                          marker='o',
                          color=c,
                          label=l,
                          linewidth=0,
                          markersize=6,
                          alpha=0.4) for l, c in colors.items()]
legend_elements = legend_elements + [Line2D([0], [0], color='black', label='')]
legend_elements = legend_elements + [Line2D([0], [0],
                                            marker='o',
                                            color='black',
                                            alpha=0.5,
                                            label=l,
                                            linewidth=0,
                                            markersize=s
                                           ) for l, s in [('NDC target low', 6),
                                                        ('NDC target ambitious', 15)]]
ax.legend(handles=legend_elements, loc='upper right')
qtile = np.nanpercentile(ndc_contrib_for_scat['targ'], 95)
for i, row in ndc_contrib_for_scat.iterrows():
    if row['targ']>qtile:
        ax.text(row['pct_tia_ce'],
                row['targ'],
                row['NAME_EN'],
                fontdict={'fontsize': 10},
               )
qtile = np.nanpercentile(ndc_contrib_for_scat['pct_tia_ce'], 95)
for i, row in ndc_contrib_for_scat.iterrows():
    if row['pct_tia_ce']>qtile:
        # NOTE: adjusting Chad for readability
        ax.text(row['pct_tia_ce']-(1*(row['NAME_EN']=='Chad')),
                row['targ'],
                 row['NAME_EN'],
                fontdict={'rotation': '45',
                          'fontsize': 10},
               )
ax.set_xlabel('contribution of cost-effective\nAF to NDC target (%)',
              fontdict={'fontsize': 20})
ax.set_ylabel('NDC target ($Tg\ CO_2e\ yr^{-1}$)',
              fontdict={'fontsize': 20})
fig6.show()


#is current woody C density roughly correlated with GDP?
# NOTE: data from: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
def scale_var(var):
    return (var - np.min(var))/(np.max(var) - np.min(var))
gdp = pd.read_csv('./API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3731268.csv', skiprows=4)
gdp_pcap = pd.read_csv('./API_NY.GDP.PCAP.CD_DS2_en_csv_v2_3731360.csv', skiprows=4)
data_w_gdp = pd.merge(data_for_figs, gdp_pcap.loc[:, ['Country Code', '2000']],
                      left_on='ISO_A3', right_on='Country Code')
fig, ax = plt.subplots(1,1)
data_w_gdp['log_gdp'] = np.log(data_w_gdp['2000'])
data_w_gdp['log_dens'] = np.log(data_w_gdp['wt_avg_density'])
sizes = 50 + scale_var(data_w_gdp['pct_below'])
sns.scatterplot(x='log_gdp', y='log_dens', hue='cont', data=data_w_gdp, ax=ax,
                size='total_potential', sizes=(50, 250),
                style='NDC', style_order=('yes', 'no'),
                edgecolor='black', linewidth=0.5, alpha=0.8, legend='brief')
sns.regplot(x='log_gdp', y='log_dens', data=data_w_gdp, ax=ax, scatter=False)
data_for_test = data_w_gdp.loc[:, ['log_gdp', 'log_dens']].dropna(how='any')
r, p = stats.pearsonr(data_for_test['log_gdp'], data_for_test['log_dens'])
ax.text(4.9, -5.8, '$R^2=%0.2f$' % r, color='red', fontstyle='italic')
ax.text(4.9, -6.1, '$p=%0.4f$' % p, color='red', fontstyle='italic')
for i, row in data_w_gdp.iterrows():
    if row['log_dens'] < np.nanpercentile(data_w_gdp['log_dens'], 10):
        ax.text(row['log_gdp'], row['log_dens'], row['NAME_EN'], color='gray')
    if row['total_potential'] > np.nanpercentile(data_w_gdp['total_potential'], 90):
        ax.text(row['log_gdp'], row['log_dens'], row['NAME_EN'], color='black')
ax.set_xlabel('log GDP per capita (yr. 2000)', fontdict={'fontsize': 18})
ax.set_ylabel('log average ag. woody C density (yr. 2000)', fontdict={'fontsize': 18})
ax.set_xlim(4.823, 11.5)
plt.legend(loc='upper right')
fig.show()

# Damien's suggestion
fig, ax = plt.subplots(1,1)
#sns.scatterplot(x=np.log(ndc_contrib_for_scat['wt_avg_density']), y='count',
sns.scatterplot(x='wt_avg_density', y='count',
                hue='cont',
                size='pct_tia_ce', data=ndc_contrib_for_scat,
                sizes=(30, 300))
#sns.regplot(x='wt_avg_density', y='count', order=1,
#                data=ndc_contrib_for_scat, scatter=False)
for i, row in ndc_contrib_for_scat.iterrows():
    if row['count'] > np.nanpercentile(ndc_contrib_for_scat['count'], 90):
        ax.text(row['wt_avg_density'], row['count'], row['NAME_EN'], color='gray')
    if row['pct_tia_ce'] > np.nanpercentile(ndc_contrib_for_scat['pct_tia_ce'], 90):
        ax.text(row['wt_avg_density'], row['count'], row['NAME_EN'], color='black')


