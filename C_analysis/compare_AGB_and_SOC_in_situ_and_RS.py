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
from statsmodels.regression.linear_model import OLS
import seaborn as sns
from latlon_utils import get_climate
import re, os


###############################################
# plot params
###############################################

dpi = 700


###############################################
# load data
###############################################
# read in Cardinael data
agb = pd.read_excel(('./Cardinael_et_al_2018_ERL_Database_AFS_Biomass'
                     '_DETH_MEAS_YR_ADDED.xlsx'))
soc = pd.read_excel('./Cardinael_et_al_2018_ERL_Database_AFS_SOC_DETH_EDIT.xlsx')

###############################################
# pre-processing
###############################################

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
                  'MEAS_YR',
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
               'meas_yr',
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
# duplicate as a 'stock_change' column (since they're effectively the same
# idea for AGB and BGB in AF being established on untreed lands)
agb['stock_change'] = agb['stock']
agb['bgb_stock_change'] = agb['bgb_stock']
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
bgb = agb.loc[:, (agb.columns != 'stock') &
                 (agb.columns != 'rate') &
                 (agb.columns != 'stock_change')]
agb = agb.loc[:, (agb.columns != 'bgb_stock') &
                 (agb.columns != 'bgb_rate') &
                 (agb.columns != 'bgb_stock_change')]
bgb.rename(columns={'bgb_stock': 'stock',
                    'bgb_rate': 'rate',
                    'bgb_stock_change': 'stock_change'}, inplace=True)


###############
# prep soc data 

# only use stands that represent AF as NCS (i.e., not converted
# from forest, etc), in order to avoid confounding of 'legacy C'
soc = soc[soc.Previous_land_use == 'Agriculture']

# subtract AF SOC from control AF to get additional C
soc['add_stock'] = soc['AFS_Stock_t_ha'] - soc['Control_Stock_t_ha']
# and keep absolute stock data, too
soc['stock'] = soc['AFS_Stock_t_ha']

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
                  'stock',
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
               'stock_change',
               'stock',
               'rate',
              ]

# add cols that are in agb and bgb data missing from here
soc['age_bm'] = np.nan
soc['meas_yr'] = np.nan

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
agb_pts = all_agb.loc[:, ['lat', 'lon', 'stock_change', 'stock', 'practice',
                          'clim', 'age_sys', 'dens', 'meas_yr']]
# add a unique ID, to be able to match back up to shuffled data from GEE
np.random.seed(2)
agb_pts['ID'] = np.abs(np.int64(np.random.normal(size=len(agb_pts))*100000))
assert len(set(agb_pts['ID'])) == len(agb_pts)
agb_pts = gpd.GeoDataFrame(agb_pts, geometry=gpd.points_from_xy(agb_pts.lon,
                                                                agb_pts.lat))
# export to file
#agb_pts.to_file('agb_pts_from_cardinael_2018.shp')
#agb_pts = gpd.read_file('agb_pts_from_cardinael_2018.shp')

# read in the points after Chapman data has been merged onto them,
# for AGB estimate comparison
agb_comp_chap = gpd.read_file('./agb_pts_from_cardinael_2018_chapman_extract.shp')
agb_comp_chap = agb_comp_chap.rename({'mean': 'chap_stock'}, axis=1)
# NOTE: CORRECT FOR FACT THAT I ESTIMATED AGC FROM AGB USING 0.5 C/BIOMASS
# RATIO (BEFORE EXPORTING FROM GEE), BUT CARDINAEL ET AL. 2018 USED 0.47
agb_comp_chap['chap_stock'] = agb_comp_chap['chap_stock']*2*0.47

agb_comp_whrc = gpd.read_file('./agb_pts_from_cardinael_2018_whrc_extract.shp')
agb_comp_whrc = agb_comp_whrc.rename({'mean': 'whrc_stock'}, axis=1)
# NOTE: CORRECT FOR FACT THAT I ESTIMATED AGC FROM AGB USING 0.5 C/BIOMASS
# RATIO (BEFORE EXPORTING FROM GEE), BUT CARDINAEL ET AL. 2018 USED 0.47
agb_comp_whrc['whrc_stock'] = agb_comp_whrc['whrc_stock']*2*0.47
agb_comp_whrc_sub = agb_comp_whrc.loc[:, ['ID', 'whrc_stock']]

# merge Chapman and WHRC extracte datasets together
agb_comp = pd.merge(agb_comp_chap, agb_comp_whrc_sub, on='ID', how='left')
assert len(agb_comp) == len(agb_comp_chap) == len(agb_comp_whrc_sub)

# merge just the extracted values onto the originally output points
# (that were fed into GEE for extraction)
agb_comp = pd.merge(agb_pts, agb_comp.loc[:,['ID', 'chap_stock', 'whrc_stock']],
                    on='ID', how='inner')
print('LENGTH agb_comp AF pts MERGE: %i' % len(agb_comp))
assert len(agb_comp) == len(agb_pts)

#rename cols
agb_comp.columns = [c if c not in ['stock', 'stock_change'] else 'card_'+c for c in agb_comp.columns]

# keep only rows where Chapman or WHRC extraction were successful
agb_comp = agb_comp[(pd.notnull(agb_comp.chap_stock)) |
                    (pd.notnull(agb_comp.whrc_stock)) ]

# get a stock-diff col
agb_comp['stock_diff_chap'] = agb_comp['card_stock_change'] - agb_comp['chap_stock']
agb_comp['stock_diff_whrc'] = agb_comp['card_stock_change'] - agb_comp['whrc_stock']

# add a column indicating locations that are in WHRC data but not in Chapman
# data
agb_comp['in_chap'] = ((pd.notnull(agb_comp['whrc_stock'])) &
                       (pd.notnull(agb_comp['chap_stock'])))


###############
# make figure 2
###############

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
soc_comp['card_stock_change_log'] = np.log10(soc_comp['stock_change'])

# make the figure
plt.close('all')
fig = plt.figure(figsize=(6.75, 10))
# NOTE: add all KDE axes, an empty set of axes (to help w/ spacing), then
#       scatter axes
gs = fig.add_gridspec(2+len(pracs), 1,
                      height_ratios=([0.25]*(len(pracs)))+[0.22]+[1])
# replace true zeros with 0.001, then manually relabel the axes, to be able to
# use log-log scale but still reflect true 0s
agb_comp['whrc_stock_false0'] = agb_comp.whrc_stock.apply(
                                        lambda x: (x*(x>0)) + (0.01*(x==0)))
agb_comp['whrc_stock_false0_log'] = np.log10(agb_comp['whrc_stock_false0'])
agb_comp['card_stock_change_log'] = np.log10(agb_comp['card_stock_change'])
col = 'whrc_stock_false0_log'
dataset_label = 'WHRC et al. unpub. 2022'
# make the scatterplot at top
ax_scat = fig.add_subplot(gs[-1,:])
ax_scat.plot([-100, 100], [-100, 100], ':k', alpha=0.3)
# choose centrality metric
cent_met = 'mean' # 'median'
cent_met_fn_dict = {'median': np.nanmedian, 'mean': np.nanmean}
cent_fn = cent_met_fn_dict[cent_met]
# centrality marks (Chapman sites and all sites)
cent_log_card = np.log10(cent_fn(agb_comp.card_stock_change))
chap_cent_log_card = np.log10(cent_fn(agb_comp[agb_comp.in_chap].card_stock_change))
cent_abs_card = cent_fn(agb_comp.card_stock_change)
chap_cent_abs_card = cent_fn(agb_comp[agb_comp.in_chap].card_stock_change)
cent_log_whrc = np.log10(cent_fn(agb_comp.whrc_stock))
chap_cent_log_whrc = np.log10(cent_fn(agb_comp[agb_comp.in_chap].whrc_stock))
cent_abs_whrc = cent_fn(agb_comp.whrc_stock)
chap_cent_abs_whrc = cent_fn(agb_comp[agb_comp.in_chap].whrc_stock)

ax_scat.scatter([cent_log_card], [cent_log_whrc], marker='*', s=90,
                alpha=0.9, facecolor='None', edgecolor='k', linewidth=2)
ax_scat.scatter([chap_cent_log_card], [chap_cent_log_whrc], marker='o', s=90,
                alpha=0.9, facecolor='None', edgecolor='k', linewidth=2)
# print centrality metrics
print('\n\n%s in situ: only Chapman-covered points: %0.2f\n\n' % (cent_met,
                                                                  chap_cent_abs_card))
print('\n\n%s in situ: all points: %0.2f\n\n' % (cent_met, cent_abs_card))
print('\n\nPercent decrease using all points: %0.2f%%\n\n' % (
                    100 * ((cent_abs_card - chap_cent_abs_card)/cent_abs_card)))
print('\n\n%s remote sensing: only Chapman-covered points: %0.2f\n\n' % (cent_met, chap_cent_abs_whrc))
print('\n\n%s remote sensing: all points: %0.2f\n\n' % (cent_met, cent_abs_whrc))
print('\n\nPercent decrease using all points: %0.2f%%\n\n' % (
                    100 * ((cent_abs_whrc - chap_cent_abs_whrc)/cent_abs_whrc)))


# label axes
ax_scat.set_xlabel('published C density\n($log_{10}\ Mg\ C\ ha^{-1}$)',
                   fontdict={'fontsize':14})
ax_scat.set_ylabel('remotely sensed AGC density\n($log_{10}\ Mg\ C\ ha^{-1}$)',
                   fontdict={'fontsize': 14})
# put scatterplot xaxis ticks and labels at top, and format
#ax_scat.xaxis.tick_top()
#ax_scat.xaxis.set_label_position('top')
ax_scat.tick_params(labelsize=10)
total_n_pts = 0
for prac in pracs:
    sub_agb_comp = agb_comp[agb_comp['practice'] == prac]
    sns.scatterplot(x='card_stock_change_log',
                    y=col,
                    hue='practice',
                    style='in_chap',
                    markers={True: '*', False: 'o'},
                    palette=[prac_colors[prac]],
                    edgecolor='black',
                    s=40,
                    alpha=0.5,
                    data=sub_agb_comp,
                    legend=False,
                    ax=ax_scat)
    total_n_pts += len(sub_agb_comp)
print('\n\n%i POINTS SCATTERED IN TOTAL\n\n' % total_n_pts)
# set tick locations and labels, axis limits
tick_locs = [-2, -1, 0, 1, 2, 3]
tick_labs = ['∅',
             '-1',
             '0',
             '1',
             '2',
             '3',
            ]
x_tick_locs = tick_locs[1:]
x_tick_labs = tick_labs[1:]
ax_lims = (-2.1, 2.5)
x_ax_lims = (-1.3, 2.5)
ax_scat.set_xticks(x_tick_locs, x_tick_labs)
ax_scat.set_yticks(tick_locs, tick_labs)
ax_scat.set_xlim(x_ax_lims)
ax_scat.set_ylim(ax_lims)
# add label for figure part 'B'
ax_scat.text(1.6*ax_scat.get_xlim()[0], 0.85*ax_scat.get_ylim()[1], 'B.',
             size=24, weight='bold', color='black', clip_on=False)

# add broken-stick indicator
tick_locs_spacing = np.diff(tick_locs[:2])[0]
broken_stick_y_centers = [np.mean(tick_locs[:2])+val for val in
                          [tick_locs_spacing * n for n in [-0.05, 0.05]]]
ax_scat.plot([ax_scat.get_xlim()[0]]*2, broken_stick_y_centers,
             color='white', linewidth=1, clip_on=False, zorder=1000)
for y_loc in broken_stick_y_centers:
    x_locs = [factor * ax_scat.get_xlim()[0] for factor in [0.95, 1, 1.05]]
    y_locs = [y_loc * factor for factor in [0.98, 1, 1.02]]
    ax_scat.plot(x_locs, y_locs, '-k', linewidth=1, clip_on=False, zorder=1001)

# add region delineating artifical y-axis zeros
zero_poly = Polygon([*zip(([*ax_scat.get_xlim()] +
                           [*ax_scat.get_xlim()][::-1] +
                           [ax_scat.get_xlim()[0]]),
                          ([ax_scat.get_ylim()[0]]*2 +
                           [broken_stick_y_centers[0]]*2 +
                           [ax_scat.get_ylim()[0]]))])
zero_poly.set_color('#ababab')
zero_poly.set_alpha(0.2)
zero_poly.set_zorder(0)
ax_scat.add_patch(zero_poly)
#ax_scat.plot(ax_scat.get_xlim(), [tick_locs[0]]*2,
#             ':r', linewidth=0.75, alpha=0.85, zorder=0)

# add horizontal line at top, to visually separate parts A. and B.
ax_scat.plot([-2.1, ax_scat.get_xlim()[1]],
             [ax_scat.get_ylim()[1]]*2,
             '-k', linewidth=2.5,
             clip_on=False)

# add gridlines
ax_scat.grid(zorder=0, linestyle=':', color='gray', alpha=0.75, linewidth=0.5)
# and extend gridlines vertically under KDEs
for x in ax_scat.get_xticks():
    ax_scat.plot([x,x], [ax_scat.get_ylim()[0], 6.565],
                 linestyle=':', color='gray', linewidth=0.5, alpha=0.75,
                 zorder=0, clip_on=False)

# get practice medians
agb_meds = np.log10(agb_comp.groupby('practice').median().loc[:,['card_stock_change']])
#agb_meds_rs = agb_comp.groupby('practice').median().loc[:,['whrc_stock_false0_log']]
soc_meds = np.log10(soc_comp[soc_comp['stock_change']>0].groupby('practice').median().loc[:,['stock_change']])
sorted_pracs = agb_meds.sort_values('card_stock_change').index.values
# plot each of the AGB and SOC KDEs
for prac_i, prac in enumerate(sorted_pracs):
    ax_kde = fig.add_subplot(gs[prac_i, :])
    sub_agb_comp = agb_comp.loc[agb_comp['practice'] == prac, ['practice',
                                                           'card_stock_change_log']]
    sub_soc_comp = soc_comp.loc[soc_comp['practice'] == prac, ['practice',
                                                    'card_stock_change_log']]
    sub_agb_comp['pool'] = 'AGC'
    sub_soc_comp['pool'] = 'SOC'
    sub_comp = pd.concat([sub_agb_comp, sub_soc_comp])
    sns.violinplot(y="practice",
                   x="card_stock_change_log",
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
        if n == 0:
            violin.set_alpha(0.75)
        else:
            violin.set_alpha(0.25)

    # drop legend
    ax_kde.legend().remove()

    # add medians
    agb_med = agb_meds.loc[prac, 'card_stock_change']
    #agb_med_rs = agb_meds_rs.loc[prac, 'whrc_stock_false0_log']
    soc_med = soc_meds.loc[prac, 'stock_change']
    ax_kde.plot([agb_med]*2, [0, -0.1], color='black', linewidth=1.5, alpha=0.75)
    ax_kde.plot([soc_med]*2, [0, 0.1], color='black', linewidth=1.5, alpha=0.35)
    # add SOC median including negative values, as a red tick
    soc_med_neg = np.log10(np.nanmedian(soc_comp[soc_comp['practice'] ==
                                                 prac]['stock_change']))
    ax_kde.plot([soc_med_neg]*2, [0, 0.1], color='red', linewidth=1.5, alpha=0.35)

    # label AF practice
    ax_kde.text(0.98*x_ax_lims[0], -0.28, prac, fontweight='bold', fontsize=16,
                color=prac_colors[prac], clip_on=False)
    ax_kde.set_ylim((0.8, -0.8)) #NOTE: lims inverted bc violinplot inverts y ax
    ax_kde.set_xlim(x_ax_lims)
    # add vertical lines to visually line up kde axes with scatter axes,
    # and a horizontal line on the top axes, to close off the 'box'
    ax_kde.plot([x_ax_lims[0]]*2, [-0.8,2], color='black', linewidth=0.75,
                clip_on=False)
    ax_kde.plot([x_ax_lims[1]]*2, [-0.8,2], color='black', linewidth=0.75,
                clip_on=False)
    if prac_i == 0:
        ax_kde.plot(x_ax_lims, [-0.8]*2, color='black', linewidth=0.75,
                    clip_on=False)

    # add counts of positive AGB and SOC values
    # (and count of negative SOC values)
    # NOTE: I CHECKED AND THERE ARE NO AGB OR SOC STOCK-CHANGE VALUES == 0
    # NOTE: this automatically drops NAs too (I checked)
    agb_ct = np.sum(agb_comp[agb_comp['practice'] == prac]['card_stock_change']>0)
    soc_ct = np.sum(soc_comp[soc_comp['practice'] == prac]['stock_change']>0)
    soc_ct_neg = np.sum(soc_comp[soc_comp['practice'] == prac]['stock_change']<0)
    ax_kde.text(0.985*ax_kde.get_xlim()[0], -0.09, '%i' % agb_ct,
                fontdict={'fontsize':9,})
    ax_kde.text(0.985*ax_kde.get_xlim()[0], 0.2, '%i' % soc_ct,
                fontdict={'fontsize':9,})
    ax_kde.text((1.06 + (0.04*(soc_ct_neg>10)))*ax_kde.get_xlim()[0], 0.2,
                '%i' % soc_ct_neg, fontdict={'fontsize':9, 'color':'red'})

    # get rid of ticks and spines
    ax_kde.set_xticks(())
    ax_kde.set_yticks(())
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('')
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['bottom'].set_visible(False)
    ax_kde.spines['left'].set_visible(False)
    # add horiz lines for each plot
    ax_kde.axhline(y=0, lw=3, xmin=x_ax_lims[0], xmax=x_ax_lims[1],
               color=prac_colors[prac], clip_on=True, alpha=0.4)
    # add black horizontal line dividing AGC and SOC
    ax_kde.plot([1.08*x_ax_lims[0], x_ax_lims[1]], [0,0],
                color='black', linewidth=1, clip_on=False, alpha=1)
    # make axis box transparent (so that plots can overlap one another)
    ax_kde.set(facecolor='none')
    # add label for figure part 'A' (if this is the top KDE axis)
    if prac_i == 0:
        ax_kde.text(1.6*ax_kde.get_xlim()[0], -0.25, 'A.',
                    size=24, weight='bold', color='black', clip_on=False)
fig.subplots_adjust(left=0.18, right=0.97, bottom=0.09, top=0.99,
                    wspace=0, hspace=-0.35)
fig.savefig('FIG2_C_density_pub_rs_comp_plot.png', dpi=dpi)
fig.show()




################
# make figure s2
################

#stock_var = 'stock_change'
stock_var = 'stock'
save_it = True
var_dict = {
            'dens': 'stem density',
           }
var_axlabel_dict = {'dens': 'density ($stems\  ha^{-1}$)'}
for var in var_dict.keys():
    fig, axs = plt.subplots(3,1, figsize=(6.5,9.75))
    for i, pool in enumerate(['agb', 'bgb', 'soc']):
        ax = axs[i]
        sns.scatterplot(var,
                        stock_var,
                        hue='practice',
                        hue_order=pracs,
                        s=30,
                        alpha=0.8,
                        data=all[all['var'] == pool],
                        ax=ax,
                        legend=i==2,
                        palette=prac_colors,
                        edgecolor='black',
                       )
        ax.set_title(pool.upper(), fontdict={'fontsize': 20})
        ax.set_xlabel(var_axlabel_dict[var], fontdict={'fontsize': 16})
        ax.set_ylabel('stock %s($Mg\ C\ ha^{-1}$)' % ('change ' * (stock_var ==
                                                                  'stock_change')),
                      fontdict={'fontsize': 16})
        ax.tick_params(labelsize=12)
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.13, right=0.96, hspace=0.6)
    fig.show()
    if save_it:
        fig.savefig('FIGS2_C_vs_%s_scatters.png' % var, dpi=700)



################
# make figure s3
################

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
    prec_col = np.max(np.vstack((np.array(lat_prec), np.array(lon_prec))), axis=0)
    # reclass the numeric coord_prec
    class_bin_lefts= [1, 2, 3]
    class_bin_rights = [2, 3, 10000]
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
for _ in range(3):
    print('-'*80)
print('\n\n')
print(('\n\nStandard deviations of differences between WHRC '
       'remotely sensed estimates and\npublished AGC '
       'for samples with different coordinate precisions:\n'))
bins = ['low', 'mod', 'high']
for prec_bin in bins:
    std = agb_comp[agb_comp.coord_prec_bin==prec_bin]['stock_diff_whrc'].std()
    print('\n\tprecision: %s\n\n\tstd: %0.4f\n\n' % (prec_bin, std))

print(("\n\nResults of Levene's test of equal variances (for "
       "non-normally-distributed samples):\n\n"))
samples = [agb_comp[agb_comp['coord_prec_bin']==b][
                                    'stock_diff_whrc'].values for b in bins]
samples = [samp[np.invert(np.isnan(samp))] for samp in samples]
levene = stats.levene(*samples)
print('\n\tstat: %0.4f\n' % levene.statistic)
print('\n\tp-value: %0.4f\n' % levene.pvalue)

# assess correlation between divergence from 1:1 line and divergence of 2000
# RS year from published measurement year
for _ in range(3):
    print('-'*80)
print('\n\n')
# subtracting WHRC measurement year (2000) from Cardinael data's measurement
# years (collected from pubs), in the same order as the stock-diff calculation,
# so that a positive correlation would be expected (i.e., negative values have
# larger stocks and later measurement dates in WHRC, and positive values are
# the opposite)
agb_comp['meas_yr_diff'] = np.float64(agb_comp['meas_yr'] - 2000)
agb_comp = agb_comp[agb_comp['meas_yr_diff'] > -10]
fig_yr_diff, ax_yr_diff = plt.subplots(1, 1, figsize=(6.5,6.5))
for prac in pracs:
    sub_agb_comp = agb_comp[agb_comp['practice'] == prac]
    sns.scatterplot(x='meas_yr_diff',
                    y='stock_diff_whrc',
                    hue='practice',
                    palette=[prac_colors[prac]],
                    edgecolor='black',
                    s=30,
                    alpha=0.5,
                    data=sub_agb_comp,
                    legend=True,
                    ax=ax_yr_diff)
sns.regplot(x='meas_yr_diff', y='stock_diff_whrc', data=agb_comp, scatter=False)
ax_yr_diff.set_xlabel(('measurement year discrepancy\n(Cardinael data '
                       'year - WHRC data year (2000))'),
                       fontdict={'fontsize':12})
ax_yr_diff.set_ylabel(('stock estimate difference ($Mg\ C\ ha^{-1}$)\n'
                       '(Cardinael data stock estmate - '
                       'WHRC stock estimate'),
                       fontdict={'fontsize':12})
ax_yr_diff.set_xlim(-10, 15)
ax_yr_diff.set_ylim(-225, 165)
ax_yr_diff.tick_params(labelsize=8)
agb_comp_reg = agb_comp.loc[:, ['stock_diff_whrc', 'meas_yr_diff']].dropna()
agb_comp_reg = agb_comp_reg[agb_comp_reg['meas_yr_diff'] > -10]
print(('\n\nResults of regression of stock-estimate difference on measurement-'
       'year difference:\n\n'))
mod = OLS(agb_comp_reg['stock_diff_whrc'],
          np.vstack((np.ones(len(agb_comp_reg)),
                     agb_comp_reg['meas_yr_diff'])).T).fit()
print('\n\tintercept: %0.4f $Mg\ C\ ha^{-1} (p=%0.2e)' % (mod.params['const'],
                                          mod.pvalues['const']))
print('\n\tslope: %0.4f $Mg\ C\ ha^{-1} yr^{-1} (p=%0.2e)' % (mod.params['x1'],
                                      mod.pvalues['x1']))
print('\n\tR-squared: %0.4f' % mod.rsquared)
fig_yr_diff.subplots_adjust(left=0.2, right=0.97, bottom=0.15, top=0.93,
                            wspace=0, hspace=0)
fig_yr_diff.savefig('FIGS3_regression_WHRC_Cardinael_stock_diff_vs_meas_yr_diff.png',
               dpi=dpi)
fig_yr_diff.show()



