import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
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


###############
# prep agb data
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

# NOTE: get rid of Adesina data b/c it's from a model, not from primary lit
agb = agb[[not ref.startswith('Adesina') for ref in agb['Full Technical Reference']]]


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


###############
# prep soc data 

# only use stands that represent AF as NCS (i.e., not converted
# from forest, etc), in order to avoid confounding of 'legacy C'
soc = soc[soc.Previous_land_use == 'Agriculture']

# subtract AF SOC from control AF to get additional C
soc['add_stock'] = soc['AFS_Stock_t_ha'] - soc['Control_Stock_t_ha']

# rename cols
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
                  'add_stock',
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

# add col that's in agb and bgb data missing from here
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
practice_key = {'Parkland': 'silvoar_and_park',
                'Silvoarable': 'silvoar_and_park',
                'Intercropping': 'intercropping',
                'Fallow': 'fallow',
                'Silvopasture': 'silvopasture',
                'Multistrata': 'multistrata',
                'Shaded_perennial': 'multistrata',
                'Shaded_perennial ': 'multistrata',
                'Hedgerow': 'hedgerow',
                'Alley_cropping': 'intercropping',
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

# add a column indicating locations that are in WHRC data but not in Chapman
# data
agb_comp['in_chap'] = ((pd.notnull(agb_comp['whrc_stock'])) &
                       (pd.notnull(agb_comp['chap_stock'])))


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
                        hue='practice',
                        style='var',
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
        # add practice-specific contours
        practices = all.practice.unique()
        prac_colors = dict(zip(practices,
                             plt.cm.Accent(np.linspace(0, 1, len(practices)))))
        prac_polys = []
        for practice in practices:
            prac_df = all.loc[all.practice == practice, [col1, col2]]
            hull = ConvexHull(prac_df.values)
            prac_poly = Polygon(prac_df.values[hull.vertices, :])
            #prac_poly.set_color(prac_colors[practice])
            prac_polys.append(prac_poly)
        prac_polys = PatchCollection(prac_polys, alpha=0.25, edgecolor='k',
                                     facecolors=[*prac_colors.values()])
        ax.add_collection(prac_polys)
        sns.scatterplot(x=col1,
                        y=col2,
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




# ridgeline plot:
# (code adapted from: https://www.python-graph-gallery.com/ridgeline-graph-seaborn)
# first, generate a color palette with Seaborn.color_palette(),
# then repeat first color twice, so that dummy plot to be removed doesn't cause
# colors to be misaligned with scatterplot colors
pracs = agb_comp.practice.unique()
pal = sns.color_palette(n_colors=len(agb_comp.practice.unique()))
pal = np.vstack((pal[0], pal))
# set shared x-axis lims
ax_lims = [-3.1, 3.1]
# add fake extra practice that will occupy first facetgrid row (to then be
# deleted and replaced with scatterplot)
extra_rows = agb_comp.iloc[:2,]
extra_rows['practice'] = 'AAAAAAAA'
agb_comp_extra_rows = pd.concat((agb_comp, extra_rows))

# add column with logged cardinael stock values
agb_comp['card_stock_log'] = np.log10(agb_comp['card_stock'])

# get the SOC carbon to create comparison ridgeline plot from
soc_comp = all[all['var'] == 'soc']
soc_comp['card_stock_log'] = np.log10(soc_comp['stock'])
# TODO: WHAT TO DO ABOUT NEG VALUES THAT GET DROPPED WHEN LOGGED?

# TODO: ENSURE THAT PRACTICES ARE ALL THE SAME AND ORDERED THE SAME

# TODO: PLOTTING AGB STOCK BUT SOC STOCK CHANGE?? HOW TO RECONCILE?

# create a sns.FacetGrid class, and set hue to practice 
# (and make top axes much taller than rest, to accomodate scatterplot
g_agb = sns.FacetGrid(agb_comp, row='practice', hue='practice',
                  aspect=15, height=0.75, palette=pal,
                  gridspec_kws={'height_ratios': [1]+([0.1]*(len(pracs)))})


# then we add the densities kdeplots for each month
g_agb.map(sns.kdeplot, 'card_stock_log', log_scale=False,
          bw_adjust=1, clip_on=False, fill=True, alpha=0.5, linewidth=1.5)
# here we add a white line that represents the contour of each kdeplot
g_agb.map(sns.kdeplot, 'card_stock_log', common_norm=True, log_scale=False,
          bw_adjust=1, clip_on=False, color="w", lw=2)


# same thing for SOC
g_soc =  sns.FacetGrid(soc_comp, row='practice', hue='practice',
                  aspect=15, height=0.75*((0.1*len(pracs))/(1+0.1*len(pracs))),
                       palette=pal[1:])
g_soc.map(sns.kdeplot, 'card_stock_log', log_scale=False,
          bw_adjust=1, clip_on=False, fill=True, alpha=0.5, linewidth=1.5)
g_soc.map(sns.kdeplot, 'card_stock_log', common_norm=True, log_scale=False,
          bw_adjust=1, clip_on=False, color="w", lw=2)
# invert the y axes
for ax in g_soc.axes[0]:
    ax.invert_yaxis()


# here we add a horizontal line for each plot
g_agb.map(plt.axhline, y=0, lw=2, clip_on=False, alpha=0.4)
g_soc.map(plt.axhline, y=0, lw=2, clip_on=False, alpha=0.4)
# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as
# text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's
# color in each matplotlib.Axes
for i, ridge_ax in enumerate(g_agb.axes.flat[1:]):
        ridge_ax.text(1e-1, 0.1, pracs[i],
                fontweight='bold', fontsize=15, color=ridge_ax.lines[-1].get_color())
        ridge_ax.set_ylim((-0.2, 0.8))
        ridge_ax.set_xlim(ax_lims)
        if i < (len(g_agb.axes.flat)-1):
            ridge_ax.set_xticks([])
        else:
            ridge_ax.set_xticks(tick_locs, tick_labs)

for i, ridge_ax in enumerate(g_soc.axes.flat):
        ridge_ax.text(1e-1, 0.1, pracs[i],
                fontweight='bold', fontsize=15, color=ridge_ax.lines[-1].get_color())
        ridge_ax.set_ylim((-0.2, 0.8))
        ridge_ax.set_xlim(ax_lims)
        if i < (len(g_soc.axes.flat)-1):
            ridge_ax.set_xticks([])
        else:
            ridge_ax.set_xticks(tick_locs, tick_labs)

# use matplotlib.Figure.subplots_adjust() function to get the
# subplots to overlap
for g in [g_agb, g_soc]:
    g.fig.subplots_adjust(hspace=-0.4)
    # remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    #plt.setp(tick_labs, fontsize=15, fontweight='bold')

g_agb.axes[0][-1].set_xlabel('AGB stock change ($Mg\ ha^{-1}$)', fontweight='bold', fontsize=15)
g_soc.axes[0][-1].set_xlabel('SOC stock change ($Mg\ ha^{-1}$)', fontweight='bold', fontsize=15)

# take top axes object of the AGB ridgeline plot for the scatterplot
ax_scat = g_agb.axes[0,0]
# clear them
ax_scat.cla()
    
    # plot diffs between Cardinael and Chapman, Cardinael and WHRC, Cardinael and Santoro
# replace true zeros with 0.01, then manually relabel the axes, to be able to
# use log-log scale but still reflect true 0s
agb_comp['whrc_stock_false0'] = agb_comp.whrc_stock.apply(
                                        lambda x: (x*(x>0)) + (0.001*(x==0)))
agb_comp['whrc_stock_false0_log'] = np.log10(agb_comp['whrc_stock_false0'])
agb_comp['card_stock_log'] = np.log10(agb_comp['card_stock'])
col = 'whrc_stock_false0_log'
dataset_label = 'WHRC et al. unpub. 2022'
ax_scat.plot([-100, 100], [-100, 100], ':k', alpha=0.3)

# median marks (Chapman sites and all sites)
med_log = np.nanmedian(np.log10(agb_comp.card_stock))
chap_med_log = np.nanmedian(np.log10(agb_comp[agb_comp.in_chap].card_stock))
med_abs = np.nanmedian(agb_comp.card_stock)
chap_med_abs = np.nanmedian(agb_comp[agb_comp.in_chap].card_stock)
ax_scat.plot([med_log], [med_log], 'ok', alpha=1, markersize=15)
ax_scat.plot([chap_med_log], [chap_med_log], 'Xk', alpha=1, markersize=15)

sns.scatterplot(x='card_stock_log',
                y=col,
                hue='practice',
                style='in_chap',
                markers={True: 'X', False: 'o'},
                palette=pal[1:], # drop the dummy first palette color
                s=100,
                alpha=0.5,
                data=agb_comp,
                legend=False,
                ax=ax_scat)

ax_scat.set_xlim(ax_lims)
ax_scat.set_ylim(ax_lims)
tick_locs = [-3, -2, -1, 0, 1, 2, 3]
tick_labs = ['0  \\\\',
             '$10^{-2}$',
             '$10^{-1}$',
             '$10^{0}$',
             '$10^{1}$',
             '$10^{2}$',
             '$10^{3}$']
ax_scat.set_xticks(tick_locs, tick_labs)
ax_scat.set_yticks(tick_locs, tick_labs)

# label x axis at bottom of FacetGrid
g_agb.axes[-1,0].set_xlabel(('$log_{10}$ aboveground biomass density ($log_{10}\ Mg\ ha^{-1}$)\n'
               'published in primary studies (analyzed by Cardinael et al. '
               '2018'), fontdict={'fontsize': 14})
g_soc.axes[-1,0].set_xlabel(('$log_{10}$ SOC stock change after AF adoption ($log_{10}\ Mg\ ha^{-1}$)\n'
               'published in primary studies (analyzed by Cardinael et al. '
               '2018'), fontdict={'fontsize': 14})
