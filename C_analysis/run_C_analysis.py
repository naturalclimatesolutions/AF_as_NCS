import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
from latlon_utils import get_climate
import re, os


# use Cardinael only, or all data?
card_only = True

###############################################
# load data
###############################################
if card_only:
    # read in Cardinael data
    agb = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_Biomass.xlsx')
    soc = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_SOC_DETH_EDIT.xlsx')

else:
    # read in data merged from all meta-analyses
    p = pd.read_excel('./Agroforestry Data Oct 2021_MERGED_METANALYSES.xlsx',
                    sheet_name='S1 literature')
    s = pd.read_excel('./Agroforestry Data Oct 2021_MERGED_METANALYSES.xlsx',
                    sheet_name='S2 sites')
    m = pd.read_excel('./Agroforestry Data Oct 2021_MERGED_METANALYSES.xlsx',
                    sheet_name='S3 measurements')


###############################################
# pre-processing
###############################################

if card_only:
    ############################
    # pre-process Cardinael data

    agb = agb.loc[:, ['Description',
                      'Technologies/Practices',
                      'Value',
                      'Unit',
                      'Value in common units',
                      'Common Unit',
                      'Full Technical Reference',
                      'Latitud(DD)',
                      'Logitud(DD)',
                      'IPCC Climate zones ',
                      'Country ',
                      'Age of the AFS (years) ',
                      'Time span for biomass production (years) ',
                      'Total tree density',
                      'BGB(Mg/ha) ',
                      'AGB Sequestration rate ',
                      'BGB Sequestration rate ',
                     ]]
    agb.columns = ['meas_type',
                   'practice',
                   'stock',
                   'unit',
                   'stock_comm',
                   'unit_comm',
                   'pub',
                   'lat',
                   'lon',
                   'clim',
                   'cnt',
                   'age_sys',
                   'age_bm',
                   'dens',
                   'bgb_stock',
                   'rate',
                   'bgb_rate',
                  ]
    # handle fact that stock measurements straddle stock and comm-unit stock cols
    corrected_stock = []
    for i, row in agb.iterrows():
        if (pd.isnull(row['stock_comm']) and pd.isnull(row['unit_comm'])):
            try:
                assert row['unit'] == 'tonnes C/ha (274)' and pd.notnull(row['stock'])
                corrected_stock.append(row['stock'])
            except Exception as e:
                print('STOCK DATA NOT AVAILABLE FOR THIS ROW (unit: %s)' % row['unit'])
                corrected_stock.append(np.nan)
        else:
            assert row['unit'] != 'tonnes C/ha (274)'
            corrected_stock.append(row['stock_comm'])
    agb['stock'] = pd.Series(corrected_stock).astype('float')
    agb = agb.loc[:, ((agb.columns != 'stock_comm') &
                      (agb.columns != 'unit') &
                      (agb.columns != 'unit_comm'))]
    # add cols in soc data missing from here
    agb['valid'] = 1
    agb['soil'] = np.nan
    agb['depth'] = np.nan
    # make sure that no 'stock-less' rows remain after all of the 'tonnes C/ha/yr'
    # rows dropped out in the for loop above, then get rid of the 'meas_type' col
    assert np.sum([d.endswith('/yr') for d in agb.meas_type]) == 0
    agb = agb.loc[:, agb.columns != 'meas_type']

    # create separate agb and bgb tables with identical sets of cols
    bgb = agb.loc[:, (agb.columns != 'stock') & (agb.columns != 'rate')]
    agb = agb.loc[:, (agb.columns != 'bgb_stock') & (agb.columns != 'bgb_rate')]
    bgb.rename(columns={'bgb_stock': 'stock', 'bgb_rate': 'rate'}, inplace=True)

    # prep soc data 
    soc = soc.loc[:, ['USED_BY_REMI',
                      'Reference',
                      'Longitude',
                      'Latitude',
                      'Country',
                      'Mean_annual_rainfall ',
                      'Mean_annual_temperature',
                      'IPCC_Climate',
                      'Soil type',
                      'Agroforestry_classification',
                      'Total_tree_density',
                      'Age (yrs)',
                      'Depth (cm)',
                      'AFS_Stock_t_ha',
                      'SOC_Storage_rate_t_ha_yr',
                     ]]
    soc.columns = ['valid',
                   'pub',
                   'lon',
                   'lat',
                   'cnt',
                   'map',
                   'mat',
                   'clim',
                   'soil',
                   'practice',
                   'dens',
                   'age_sys',
                   'depth',
                   'stock',
                   'rate',
                  ]
    # add col in agb and bgb data missing from here
    soc['age_bm'] = np.nan

    # line up columns in same order
    bgb = bgb.loc[:, agb.columns]
    soc = soc.loc[:, agb.columns]

    # add var columns
    agb['var'] = 'agb'
    bgb['var'] = 'bgb'
    soc['var'] = 'soc'

    # concatenate all
    all = pd.concat((agb, bgb, soc))

    # remap practice names
    practice_key = {'Parkland': 'park',
                    'Silvoarable': 'silvoar',
                    'Intercropping': 'intercrop',
                    'Fallow': 'fallow',
                    'Silvopasture': 'silvopas',
                    'Multistrata': 'multistrat',
                    'Shaded_perennial': 'shade',
                    'Shaded_perennial ': 'shade',
                    'Hedgerow': 'hedge',
                    'Alley_cropping': 'alley',
                   }
    practice = [practice_key[p] for p in all['practice']]
    all['practice'] = practice

    # extract MAP and MAT values for all sites missing them?
    print('\n\nDownloading climate data. This will take a few...\n\n')
    new_mat = []
    new_map = []
    for i, row in all.iterrows():
        lat = row['lat']
        lon = row['lon']
        clim = get_climate(lat=lat,
                           lon=lon,
                           res='5m',
                           radius=3,
                          )
        mat = clim.tavg.mean()
        map = clim.prec.mean()
        new_mat.append(mat)
        new_map.append(map)

    all['new_mat'] = new_mat
    all['new_map'] = new_map


else:
    #############################
    # pre-process all merged data

    m['site.id'] = m['site.ID']
    m['density'] = [float(d) if (pd.notnull(d) and  re.search('\d+\.?\d*',
                                        str(d))) else np.nan for d in m['density']]
    m['density'] = [float(d) for d in m['density']]
    dens_pctiles = np.nanpercentile(m['density'], [20, 40, 60, 80,])
    age_pctiles = np.nanpercentile(m['stand.age'], [20, 40, 60, 80,])
    cat_strs = {0: 'X < 20th',
                1: '20th < X < 40th',
                2: '40th < X < 60th' ,
                3: '60th < X < 80th',
                4: 'X > 80th',
               }
    dens_cats = []
    age_cats = []
    means = []
    for i, row in m.iterrows():
        if pd.isnull(row['density']):
            dens_cats.append(np.nan)
        else:
            dens_cat_num = sum(row['density']>dens_pctiles)
            dens_cats.append(cat_strs[dens_cat_num])
        if pd.isnull(row['stand.age']):
            age_cats.append(np.nan)
        else:
            age_cat_num = sum(row['stand.age']>age_pctiles)
            age_cats.append(cat_strs[age_cat_num])
        if pd.isnull(row['mean']):
            means.append(np.nan)
        elif isinstance(row['mean'], str):
            try:
                if re.search('\-\+', row['mean']):
                    mean_val = float(row['mean'].split('-+')[0])
                elif re.search('\+\-', row['mean']):
                    mean_val = float(row['mean'].split('+-')[0])
                elif re.search('\d to \d', row['mean']):
                    mean_val = float(row['mean'].split(' to ')[0])
                elif row['mean'] == 'n.a.':
                    mean_val = np.nan
                else:
                    mean_val = np.mean([float(s) for s in row['mean'].split('-')])
                means.append(mean_val)
            except Exception as e:
                print(row['mean'])
                print(e)
                means.append(np.nan)
        else:
            means.append(row['mean'])
    m['density_cat'] = pd.Series(dens_cats).astype('category')
    m['age_cat'] = pd.Series(age_cats).astype('category')
    m['mean'] = pd.Series(means).astype('float')

    # subset to cols of interest
    p = p.loc[:, ['study.id',
                  'citations.author',
                  'citations.year',
                  'Cardinael 2018',
                 ]]
    s = s.loc[:, ['site.id',
                  'study.id',
                  'site.country',
                  'lat',
                  'lon',
                  'masl',
                  'map',
                  'mat',
                 ]]
    m = m.loc[:, ['site.id',
                  'plot.id',
                  'measurement.ID',
                  'prior',
                  'refor.type',
                  'system.age',
                  'stand.age',
                  'variable.name',
                  'mean',
                  'lower95CI',
                  'upper95CI',
                  'se',
                  'sd',
                  'density',
                  'density_cat',
                  'age_cat',
                 ]]

    # reconcile into single table for analysis
    all = pd.merge(pd.merge(m, s, on='site.id', how='left'),
                      p, on='study.id', how='left')

    # subset for only Cardinael et al. 2018 data (highest-quality meta-analysis)
    all = all[[str(i).startswith('c') for i in all['study.id']]]


# TODO: drop duplicates?

# list of the categorical columns to use for the boxplots
cat_cols = {'age_sys': 'scat',
            'dens': 'scat',
            'practice': 'box',
            ('new_mat', 'new_map'): 'heat',
           }

# make the boxplot fig
fig1, axs = plt.subplots(2,2)
axs = axs.flatten()
for col_fn, ax in zip(cat_cols.items(), axs):
    col, plot_fn = col_fn
    if isinstance(col, str):
        suball = all[pd.notnull(all[col])]
        if plot_fn == 'scat':
            sns.scatterplot(x=col,
                        y='stock',
                        hue='var',
                        data=suball,
                        alpha=.7,
                        ax=ax)
        elif plot_fn == 'box':
            sns.boxenplot(x=col,
                      y='stock',
                      hue='var',
                      data=suball,
                      ax=ax)
    elif isinstance(col, tuple):
        whittaker = pd.read_csv('whittaker_biomes.csv', sep=';')
        whittaker['temp_c'] = whittaker['temp_c'].apply(lambda x:
                                                    float(x.replace(',', '.')))
        whittaker['precp_cm'] = whittaker['precp_cm'].apply(lambda x:
                                                    float(x.replace(',', '.')))
        biomes = []
        centroids = []
        patches = []

        for biome in whittaker['biome'].unique():
            subwhit = whittaker[whittaker.biome == biome].loc[:, ['temp_c', 'precp_cm']].values
            centroids.append(np.mean(subwhit, axis=0))
            poly = Polygon(subwhit, True)
            patches.append(poly)
            biomes.append(re.sub('/', '/\n', biome))

        p = PatchCollection(patches, alpha=0.4, edgecolor='k', cmap='Pastel1')
        p.set_array(['white']*len(biomes))
        ax.add_collection(p)

        for centroid, biome in zip(centroids, biomes):
            ax.text(*centroid, biome, fontdict={'fontsize': 9})

        col1, col2 = col
        suball = all[(pd.notnull(all[col1])) & (pd.notnull(all[col2]))]
        sns.kdeplot(x='new_mat',
                    y='new_map',
                    fill='stock',
                    cmap='plasma',
                    alpha = 0.5,
                    data=all,
                    ax=ax)
        ax.set_xlabel('MAT ($^{○}C$)')
        ax.set_ylabel('MAP ($cm$)')


fig1.show()
