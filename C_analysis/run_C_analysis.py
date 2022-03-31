import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
from scipy import stats
import seaborn as sns
from latlon_utils import get_climate
import re, os


# TODO:
    # okay to use log axes even though zeros and neg values lost??



USE_ALL_DATA = False


###############################################
# load data
###############################################
# read in Cardinael data
agb = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_Biomass.xlsx')
soc = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_SOC_DETH_EDIT.xlsx')

if USE_ALL_DATA:
    # read in data merged from all meta-analyses
    p = pd.read_excel('./Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx',
                    sheet_name='S1 literature')
    s = pd.read_excel('./Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx',
                    sheet_name='S2 sites')
    m = pd.read_excel('./Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx',
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

# get count of rows dropped, to be used in assert statements later
rows_b4 = len(agb)
# NOTE: get rid of Adesina data b/c it's from a model, not from primary lit
agb = agb[[not ref.startswith('Adesina') for ref in agb['Full Technical Reference']]]
# NOTE: get rid of Bright and Diels data b/c they only measured pruning biomass
#       and litter inputs, respectively (neither of which are AGB)
agb = agb[[not ref.startswith('Bright') for ref in agb['Full Technical Reference']]]
agb = agb[[not ref.startswith('Diels') for ref in agb['Full Technical Reference']]]
rows_af = len(agb)
rows_dropped = rows_b4 - rows_af


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
practice_key = {'Parkland': 'silvoarable',
                'Silvoarable': 'silvoarable',
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


# get lat,lon for all measurements from meta-analyses
# (these will be used to extract woodyC stock estimates from Chapman
#  for comparison to published estimates)
all_agb = all[all['var'] == 'agb']
agb_pts = all_agb.loc[:, ['lat', 'lon', 'stock', 'practice',
                          'clim', 'age_sys', 'dens']]
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
assert len(agb_comp_chap) == (len(agb_pts) + rows_dropped)

agb_comp_whrc = gpd.read_file('./agb_pts_from_cardinael_2018_whrc_extract.shp')
agb_comp_whrc = agb_comp_whrc.rename({'mean': 'whrc_stock'}, axis=1)
assert len(agb_comp_whrc) == (len(agb_pts) + rows_dropped)
agb_comp_whrc_sub = agb_comp_whrc.loc[:, ['ID', 'whrc_stock']]

agb_comp = pd.merge(agb_comp_chap, agb_comp_whrc_sub, on='ID', how='left')

assert len(agb_comp) == len(agb_comp_chap) == len(agb_comp_whrc_sub)
agb_comp = pd.merge(agb_pts, agb_comp.loc[:,['ID', 'chap_stock', 'whrc_stock']],
                    on='ID', how='inner')
assert len(agb_comp) == len(agb_pts)

#rename cols
agb_comp.columns = [c if c!= 'stock' else 'card_stock' for c in agb_comp.columns]
# keep only rows where Chapman or WHRC extraction were successful
# NOTE: only about 1/4 of rows!
agb_comp = agb_comp[(pd.notnull(agb_comp.chap_stock)) |
                    (pd.notnull(agb_comp.whrc_stock)) ]
# get a stock-diff col
agb_comp['stock_diff_chap'] = agb_comp['card_stock'] - agb_comp['chap_stock']
agb_comp['stock_diff_whrc'] = agb_comp['card_stock'] - agb_comp['whrc_stock']

# add a column indicating locations that are in WHRC data but not in Chapman
# data
agb_comp['in_chap'] = ((pd.notnull(agb_comp['whrc_stock'])) &
                       (pd.notnull(agb_comp['chap_stock'])))


#############################
# pre-process all merged data
if USE_ALL_DATA:
    m['site.id'] = m['site.ID']
    m['density'] = [float(d) if (pd.notnull(d) and  re.search('^\d+\.?\d*$',
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

# get array of all the practices
pracs = agb_comp.practice.unique()
# dict of practice colors; Bright 6 color palette from http://tsitsul.in/blog/coloropt/
palette = [
           '#e935a1', # pink -> silvoarable and parkland
           '#537eff', # neon blue -> intercropping
           '#00e3ff', # light blue -> fallow
           '#efe645', # yellow -> silvopasture
           '#00cb85', # green -> multistrata
           '#e15623', # carrot -> hedgerow
          ]
prac_colors = dict(zip(pracs, palette))

# get the SOC carbon to create comparison ridgeline plot from
soc_comp = all[all['var'] == 'soc']
soc_comp['card_stock_log'] = np.log10(soc_comp['stock'])

# make the figure
plt.close('all')
fig = plt.figure(figsize=(5.5, 8))
# NOTE: add top scatter axes, empty axes for spacing, then all KDE axes pairs
gs = fig.add_gridspec(2+len(pracs), 1,
#gs = fig.add_gridspec(2+len(pracs)*2, 1,
                      height_ratios=[1]+[0.25]+([0.2]*(len(pracs))))
# replace true zeros with 0.001, then manually relabel the axes, to be able to
# use log-log scale but still reflect true 0s
agb_comp['whrc_stock_false0'] = agb_comp.whrc_stock.apply(
                                        lambda x: (x*(x>0)) + (0.001*(x==0)))
agb_comp['whrc_stock_false0_log'] = np.log10(agb_comp['whrc_stock_false0'])
agb_comp['card_stock_log'] = np.log10(agb_comp['card_stock'])
col = 'whrc_stock_false0_log'
dataset_label = 'WHRC et al. unpub. 2022'
# make the scatterplot at top
ax_scat = fig.add_subplot(gs[0,0])
ax_scat.plot([-100, 100], [-100, 100], ':k', alpha=0.3)
# median marks (Chapman sites and all sites)
med_log = np.nanmedian(np.log10(agb_comp.card_stock))
chap_med_log = np.nanmedian(np.log10(agb_comp[agb_comp.in_chap].card_stock))
med_abs = np.nanmedian(agb_comp.card_stock)
chap_med_abs = np.nanmedian(agb_comp[agb_comp.in_chap].card_stock)
ax_scat.scatter([med_log], [med_log], marker='*', s=70,
                alpha=0.9, facecolor='None', edgecolor='k')
ax_scat.scatter([chap_med_log], [chap_med_log], marker='o', s=70,
                alpha=0.9, facecolor='None', edgecolor='k')
# label axes
ax_scat.set_xlabel('$log_{10}$ published AGB density ($Mg\ C\ ha^{-1}$)',
                   fontdict={'fontsize':12})
ax_scat.set_ylabel('$log_{10}$ remotely sensed AGB density ($Mg\ C\ ha^{-1}$)',
                   fontdict={'fontsize': 12})
# put scatterplot xaxis ticks and labels at top, and format
ax_scat.xaxis.tick_top()
ax_scat.xaxis.set_label_position('top')
ax_scat.tick_params(labelsize=9)
for prac in pracs:
    sub_agb_comp = agb_comp[agb_comp['practice'] == prac]
    sns.scatterplot(x='card_stock_log',
                    y=col,
                    hue='practice',
                    style='in_chap',
                    markers={True: '*', False: 'o'},
                    palette=[prac_colors[prac]],
                    edgecolor='black',
                    s=25,
                    alpha=0.5,
                    data=sub_agb_comp,
                    legend=False,
                    ax=ax_scat)
# set tick locations and labels, axis limits
tick_locs = [-3, -2, -1, 0, 1, 2, 3]
tick_labs = ['0  \\\\',
             '$10^{-2}$',
             '$10^{-1}$',
             '$10^{0}$',
             '$10^{1}$',
             '$10^{2}$',
             '$10^{3}$']
x_tick_locs = tick_locs[2:]
x_tick_labs = tick_labs[2:]
ax_lims = (-3.1, 2.5)
x_ax_lims = (-1.1, 2.5)
ax_scat.set_xticks(x_tick_locs, x_tick_labs)
ax_scat.set_yticks(tick_locs, tick_labs)
ax_scat.set_xlim(x_ax_lims)
ax_scat.set_ylim(ax_lims)
# get practice medians
agb_meds = agb_comp.groupby('practice').median().loc[:,['card_stock_log']]
#agb_meds_rs = agb_comp.groupby('practice').median().loc[:,['whrc_stock_false0_log']]
soc_meds = soc_comp.groupby('practice').median().loc[:,['card_stock_log']]
sorted_pracs = agb_meds.sort_values('card_stock_log').index.values
# plot each of the AGB and SOC KDEs
for prac_i, prac in enumerate(sorted_pracs):
    #ax_agb = fig.add_subplot(gs[2+(prac_i*2),0])
    #ax_soc = fig.add_subplot(gs[3+(prac_i*2),0])
    ax_kde = fig.add_subplot(gs[2+prac_i, 0])
    sub_agb_comp = agb_comp.loc[agb_comp['practice'] == prac, ['practice',
                                                           'card_stock_log']]
    sub_soc_comp = soc_comp.loc[soc_comp['practice'] == prac, ['practice',
                                                           'card_stock_log']]
    sub_agb_comp['pool'] = 'AGB'
    sub_soc_comp['pool'] = 'SOC'
    sub_comp = pd.concat([sub_agb_comp, sub_soc_comp])
    sns.violinplot(y="practice",
                   x="card_stock_log",
                   hue="pool",
                   data=sub_comp,
                   palette=[prac_colors[prac]]*2,
                   split=True,
                   inner=None,
                   #inner='stick',
                   legend=False,
                   ax=ax_kde,
                   clip_on=False)
    # make violins transparent
    for n, violin in enumerate(ax_kde.collections):
            violin.set_alpha(0.75 - (n*0.4))
    ax_kde.legend().remove()
    # add medians
    agb_med = agb_meds.loc[prac, 'card_stock_log']
    #agb_med_rs = agb_meds_rs.loc[prac, 'whrc_stock_false0_log']
    soc_med = soc_meds.loc[prac, 'card_stock_log']
    ax_kde.plot([agb_med]*2, [0, -0.1], color='black', linewidth=1.5, alpha=0.75)
    #ax_kde.plot([agb_med_rs]*2, [0, -0.1], '--', color='black', linewidth=1.5, alpha=0.75)
    ax_kde.plot([soc_med]*2, [0, 0.1], color='black', linewidth=1.5, alpha=0.35)
    # add left arrow
    """
    sns.kdeplot('card_stock_log',
                hue='practice',
                data=sub_agb_comp,
                bw_adjust=0.5,
                clip_on=False,
                zorder=10,
                fill=True,
                palette=[prac_colors[prac]],
                alpha=0.5,
                linewidth=1.5,
                legend=False,
                ax=ax_agb)
    sns.kdeplot('card_stock_log',
                hue='practice',
                data=sub_soc_comp,
                bw_adjust=0.5,
                clip_on=False,
                zorder=10,
                fill=True,
                palette=[prac_colors[prac]],
                alpha=0.5,
                linewidth=1.5,
                legend=False,
                ax=ax_soc)
    """
    # label AF practice
    ax_kde.text(1.5*x_ax_lims[0], -0.15, prac, fontweight='bold', fontsize=12,
                color=prac_colors[prac], clip_on=False)
    # add horiz lines for each plot
    ax_kde.axhline(y=0, lw=2, xmin=x_ax_lims[0], xmax=x_ax_lims[1],
               color=prac_colors[prac], clip_on=True, alpha=0.4)
    ax_kde.set_ylim((0.8, -0.8)) #NOTE: lims inverted bc violinplot inverts y ax
    ax_kde.set_xlim(x_ax_lims)
    ax_kde.set_xticks(())
    ax_kde.set_yticks(())
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['bottom'].set_visible(False)
    ax_kde.spines['left'].set_visible(False)
    # add black horizontal line dividing AGB and SOC
    ax_kde.axhline(y=0, lw=1, xmin=x_ax_lims[0], xmax=x_ax_lims[1],
                   color='black', clip_on=True, alpha=1)
    # make axis box transparent (so that plots can overlap one another)
    ax_kde.set(facecolor='none')
fig.subplots_adjust(left=0.15, right=0.97, bottom=0, top=0.93,
                    wspace=0, hspace=-0.35)
fig.savefig('C_density_pub_rs_comp_plot.png', dpi=700)
fig.show()



# assess variance in divergence from 1:1 line as fn of geo coord precision
def calc_coord_precision(coord):
    ct = 0
    # return 0 if no decimals
    if '.' not in str(coord):
        return ct
    str_dec = str(coord).split('.')[1]
    for i, dig in enumerate(str_dec):
        ct += 1
        if i+1 == len(str_dec):
            return ct
        elif i+2 == len(str_dec):
            if str_dec[i+1] == dig:
                return ct
            else:
                return ct+1
        else:
            if str_dec[i+2] == str_dec[i+1] == dig:
                return ct

def calc_coord_precision_column(df):
    lat_prec = [calc_coord_precision(lat) for lat in df.lat]
    lon_prec = [calc_coord_precision(lon) for lon in df.lon]
    prec_col = (np.array(lat_prec) + np.array(lon_prec))/2
    # reclass the numeric coord_prec
    class_bin_lefts= [1, 2, 4]
    class_bin_rights = [2, 4, 10000]
    class_bins = ['low', 'mod', 'high']
    prec_bin_col = []
    for val in prec_col:
        for bin, l, r in zip(class_bins, class_bin_lefts, class_bin_rights):
            if l<= val < r:
                prec_bin_col.append(bin)
    assert len(prec_bin_col) == len(prec_col)
    return prec_col, prec_bin_col

prec_col, prec_bin_col = calc_coord_precision_column(agb_comp)
agb_comp['coord_prec'] = prec_col
agb_comp['coord_prec_bin'] = prec_bin_col

# print variances for low, mod, and high-precision coordinates,
# and print results of Bartlett's test for equal variances
# (for all normally distributed samples, since histograms look close enough)
print(('\n\nStandard deviations of differences between WHRC '
       'remotely sensed estimates and\npublished AGB '
       'for samples with different coordinate precisions:\n'))
bins = ['low', 'mod', 'high']
for prec_bin in bins:
    std = agb_comp[agb_comp.coord_prec_bin==prec_bin]['stock_diff_whrc'].std()
    print('\n\tprecision: %s\n\n\tstd: %0.4f\n\n' % (prec_bin, std))

print(("\n\nResults of Bartlett's test of equal variances (for "
       "normally-distributed samples:\n\n"))
samples = [agb_comp[agb_comp['coord_prec_bin']==b][
                                    'stock_diff_whrc'].values for b in bins]
samples = [samp[np.invert(np.isnan(samp))] for samp in samples]
bart = stats.bartlett(*samples)
print('\n\tstat: %0.4f\n' % bart.statistic)
print('\n\tp-value: %0.4f\n' % bart.pvalue)

# TODO:
# assess correlation between divergence from 1:1 line and divergence of 2000
# RS year from published measurement year
