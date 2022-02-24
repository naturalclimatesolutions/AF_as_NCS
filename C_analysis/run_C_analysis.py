import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from latlon_utils import get_climate
import re, os


###############################################
# load data
###############################################
# read in Cardinael data
agb = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_Biomass.xlsx')
soc = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_SOC_DETH_EDIT.xlsx')

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
agb['mat'] = np.nan
agb['map'] = np.nan

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



# get lat,lon for all measurements from meta-analyses
# (these will be used to extract woodyC stock estimates from Chapman
#  for comparison to published estimates)
all_agb = all[all['var'] == 'agb']
agb_pts = all_agb.loc[:, ['lat', 'lon', 'stock', 'practice', 'new_mat',
                          'new_map', 'clim', 'age_sys', 'dens']]
# add a unique ID, to be able to match back up to shuffled data from GEE
np.random.seed(2)
agb_pts['ID'] = np.abs(np.int64(np.random.normal(size=len(agb_pts))*100000))
assert len(set(agb_pts['ID'])) == len(agb_pts)
agb_pts = gpd.GeoDataFrame(agb_pts, geometry=gpd.points_from_xy(agb_pts.lon,
                                                                agb_pts.lat))
# export to file
#agb_pts.to_file('agb_pts.shp')
#agb_pts = gpd.read_file('agb_pts.shp')

# read in the points after Chapman data has been merged onto them,
# for AGB estimate comparison
agb_comp_chap = gpd.read_file('./agb_pts_from_cardinael_2018_chapman_extract.shp')
agb_comp_chap = agb_comp_chap.rename({'mean': 'chap_stock'}, axis=1)
assert len(agb_comp_chap) == len(agb_pts)

agb_comp_whrc = gpd.read_file('./agb_pts_from_cardinael_2018_whrc_extract.shp')
agb_comp_whrc = agb_comp_whrc.rename({'mean': 'whrc_stock'}, axis=1)
assert len(agb_comp_whrc) == len(agb_pts)
agb_comp_whrc_sub = agb_comp_whrc.loc[:, ['ID', 'whrc_stock']]

agb_comp_sant = gpd.read_file('./agb_pts_from_cardinael_2018_santoro_extract.shp')
agb_comp_sant = agb_comp_sant.rename({'mean': 'sant_stock'}, axis=1)
assert len(agb_comp_sant) == len(agb_pts)
agb_comp_sant_sub = agb_comp_sant.loc[:, ['ID', 'sant_stock']]

agb_comp = pd.merge(agb_comp_chap, agb_comp_sant_sub, on='ID', how='left')
agb_comp = pd.merge(agb_comp, agb_comp_whrc_sub, on='ID', how='left')

assert len(agb_comp) == len(agb_comp_chap) == len(agb_comp_whrc_sub) == len(agb_comp_sant_sub)
agb_comp = pd.merge(agb_pts, agb_comp.loc[:,['ID', 'chap_stock', 'whrc_stock', 'sant_stock']],
                    on='ID', how='inner')
assert len(agb_comp) == len(agb_pts)

#rename cols
agb_comp.columns = [c if c!= 'stock' else 'card_stock' for c in agb_comp.columns]
# keep only rows where Chapman extraction was successful
# NOTE: only about 1/4 of rows!
agb_comp = agb_comp[(pd.notnull(agb_comp.chap_stock)) |
                    (pd.notnull(agb_comp.sant_stock)) |
                    (pd.notnull(agb_comp.whrc_stock)) ]
# get a stock-diff col
agb_comp['stock_diff_chap'] = agb_comp['card_stock'] - agb_comp['chap_stock']
agb_comp['stock_diff_whrc'] = agb_comp['card_stock'] - agb_comp['whrc_stock']
agb_comp['stock_diff_sant'] = agb_comp['card_stock'] - agb_comp['sant_stock']


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
              'DeStefano 2018',
              'Feliciano 2018',
              'Kim 2016',
              'Shi 2018',
              'Ma 2020',
              'Drexler 2021',
              'Hubner 2021',
              'Mayer 2021',
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
db = pd.merge(pd.merge(m, s, on='site.id', how='left'),
                  p, on='study.id', how='left')

# add single column indicating if each measurement comes from a meta-anal study
db['in_meta'] = np.sum(db.iloc[:, 25:], axis=1)>1

# subset for only Cardinael et al. 2018 data (highest-quality meta-analysis)
db = db[[str(i).startswith('c') for i in db['study.id']]]




##########################################
# make plots
##########################################



# TODO: drop duplicates?

# list of the categorical columns to use for the boxplots
cat_cols = {'age_sys': 'scat',
            'dens': 'scat',
            'practice': 'box',
            ('new_mat', 'new_map'): 'heat',
           }

def normalize(vals, min_out=0, max_out=1):
    if max_out <= min_out:
        max_out = min_out+1
    norm = min_out + ((max_out-min_out)*(vals-np.min(vals)))/(np.max(vals)-np.min(vals))
    return norm

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
        p = PatchCollection(patches, alpha=0.4, edgecolor='k', facecolors='white')
        ax.add_collection(p)
        for centroid, biome in zip(centroids, biomes):
            ax.text(*centroid, biome, fontdict={'fontsize': 9})
        col1, col2 = col
        suball = all[(pd.notnull(all[col1])) & (pd.notnull(all[col2]))]
        sns.scatterplot(x='new_mat',
                        y='new_map',
                        hue='stock',
                        size=normalize(all['stock'], 200, 500),
                        style='practice',
                        #palette='plasma',
                        edgecolor='black',
                        alpha = 0.5,
                        data=all,
                        ax=ax)
        ax.set_xlabel('MAT ($^{○}C$)')
        ax.set_ylabel('MAP ($cm$)')

fig1.show()

# plot diffs between Cardinael and Chapman, Cardinael and Santoro
fig2, axs2 = plt.subplots(3,2)
axs2 = axs2.flatten()
for i, col in enumerate(['chap_stock', 'whrc_stock', 'sant_stock']):
    stock_diff_col = {'chap_stock': 'stock_diff_chap',
                      'whrc_stock': 'stock_diff_whrc',
                      'sant_stock': 'stock_diff_sant'}[col]
    dataset_label = {'chap_stock': 'Chapman et al. 2020',
                     'whrc_stock': 'WHRC et al. unpub. 2022',
                     'sant_stock': 'Santoro et al. 2021'}[col]
    axs2[0+(2*i)].plot([0,70], [0,70], ':k')
    sns.scatterplot(x='card_stock',
                    y=col,
                    hue='practice',
                    data=agb_comp,
                    ax=axs2[0+(2*i)])
    axs2[0+(2*i)].set_xlim([0, 1.05*agb_comp[col].max()])
    axs2[0+(2*i)].set_xlabel(('published woody C density estimate\n($Mg\ C\ ha^{-1}$);'
                       ' Cardinael et al. 2018'))
    axs2[0+(2*i)].set_ylabel(('remotely sensed woody C density estimate\n($Mg\ C\ ha^{-1}$);'
                       ' %s') % dataset_label)
    divider = make_axes_locatable(axs2[1+(2*i)])
    cax = divider.append_axes("bottom", size="5%", pad=0.25)
    axs2[1+(2*i)].set_xticks(())
    axs2[1+(2*i)].set_xticklabels(())
    axs2[1+(2*i)].set_yticks(())
    axs2[1+(2*i)].set_yticklabels(())
    countries = gpd.read_file('../mapping/country_bounds/NewWorldFile_2020.shp')
    countries = countries.to_crs(4326)
    countries.plot(facecolor='none',
                   edgecolor='black',
                   linewidth=0.25,
                   ax=axs2[1+(2*i)])
    scat=axs2[1+(2*i)].scatter(x=agb_comp.lon,
                         y=agb_comp.lat,
                         c=agb_comp[stock_diff_col],
                         cmap='RdBu',
                         s=100,
                         edgecolors='black',
                         linewidth=0.75,
                         alpha=0.7,
                         vmin=-225,
                         vmax=225,
                        )
    plt.colorbar(scat, orientation="horizontal", cax=cax,
                 label=('published estimate (Cardinael et al.'
                        ' 2018) - remote sensing estimate'
                        ' (%s) ($Mg\ C\ ha^{-1}$)') % dataset_label)
    cax.set_title('discrepancy')
fig2.show()

# show discrepancy by dataset and system type
df_discrep_by_dataset_prac = agb_comp.melt(id_vars=['practice'],
                                           value_vars=['stock_diff_chap',
                                                       'stock_diff_whrc',
                                                       'stock_diff_sant'])
fig3, ax = plt.subplots(1,1)
sns.violinplot(x='practice', y='value', hue='variable',
              data=df_discrep_by_dataset_prac)
ax.plot(ax.get_xlim(), [0,0], '--k', linewidth=2, alpha=0.25)
fig3.show()

