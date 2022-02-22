import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# read in and prep countries and continents
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



# read in and prep Chris' table
# NOTE: got rid of 1st row with units, but units are TgCO2e yr^-1
df_raw = pd.read_csv('./pathway_mitigation_potential_and_NDC_targets_with_ISO3.csv')

# subset and rename cols
df = df_raw.loc[:, ['iso3',
                'CountryGeography',
                'Trees in Agriculture Lands [Chapman]',
                'Cost-effective Trees in Agriculture Lands [Chapman]',
                'Reforestation (GROA)',
                'Cost-effective Reforestation (GROA)',
                '(Sharon et al) Emissions Reduction Target',
                '(Sharon et al) NDC Reduction Percent',
                '(Sharon et al) Reference Year Emissions Rate',
                '(Sharon et al) New Annual Emisions after target reached',
                'NDC Summary',
               ]]
df.columns = ['iso3',
              'geo',
              'tia',
              'tia_ce',
              'refor',
              'refor_ce',
              'targ',
              'red_pct',
              'ref_yr',
              'new_emis',
              'ndc_summ'
             ]

# calculate proportion of NDC goals that could be met by AF
df['pct_tia'] = df['tia']/df['targ'] * 100
df['pct_tia_ce'] = df['tia_ce']/df['targ'] * 100
df['pct_refor'] = df['refor']/df['targ'] * 100
df['pct_refor_ce'] = df['refor_ce']/df['targ'] * 100

# plot hists for comparison
df_hist_tia = df.loc[:, ['geo', 'pct_tia', 'pct_tia_ce']]
df_hist_tia.columns = ['country', 'max_potential', 'cost_effective']
df_hist_tia = pd.melt(df_hist_tia, id_vars=['country'],
                      value_vars=['max_potential', 'cost_effective'])
df_hist_tia['NCS'] = 'agroforestry'
df_hist_refor = df.loc[:, ['geo', 'pct_refor', 'pct_refor_ce']]
df_hist_refor.columns = ['country', 'max_potential', 'cost_effective']
df_hist_refor = pd.melt(df_hist_refor, id_vars=['country'],
                        value_vars=['max_potential', 'cost_effective'])
df_hist_refor['NCS'] = 'reforestation'
df_hist = pd.concat((df_hist_tia, df_hist_refor), axis=0)
df_hist.columns = ['country', 'estimate_type', 'percent_NDC_target', 'NCS']


fig, axs = plt.subplots(2,1)
sns.histplot(x="percent_NDC_target",
             hue='NCS',
             alpha=0.5,
             binwidth=2.5,
             binrange=(0,110),
             legend=True,
             ax=axs[0],
             data=df_hist[df_hist['estimate_type']=='max_potential'],
            )
sns.histplot(x="percent_NDC_target",
             hue='NCS',
             alpha=0.5,
             binwidth=2.5,
             binrange=(0,110),
             legend=True,
             ax=axs[1],
             data=df_hist[df_hist['estimate_type']=='cost_effective'],
            )
for i, ax in enumerate(axs):
    ax.set_ylabel('count')
    if i == 0:
        ax.set_title('Max potential')
    if i == 1:
        ax.set_title('Cost-effective')
        ax.set_xlabel('achievable percent of NDC target')

# merge onto countries
df_map = pd.merge(countries, df, left_on='iso_a3', right_on='iso3', how='outer')

fig_map, axs = plt.subplots(2, 1)
fig.suptitle('percent of NDC targets achievable by agroforestry (Baruch-Mordo et al. 2018)')
for ax in axs:
    countries.plot(facecolor='none',
               edgecolor='black',
               linewidth=0.25,
               ax=ax)
    ax.set_xticks(())
    ax.set_xticklabels(())
    ax.set_yticks(())
    ax.set_yticklabels(())
map = df_map.plot('pct_tia',
               ax=axs[0],
                cmap='rainbow',
                vmin=0,
                vmax=100,
                legend=True,
                legend_kwds={'label': 'percent NDC target achievable'})
map = df_map.plot('pct_tia_ce',
               ax=axs[1],
                  cmap='rainbow',
                  vmin=0,
                  vmax=100,
                  legend=True,
                  legend_kwds={'label': 'percent_NDC_target_achievable'})
axs[0].set_ylabel('max potential mitigation')
axs[1].set_ylabel('cost-effective mitigation')

plt.show()


