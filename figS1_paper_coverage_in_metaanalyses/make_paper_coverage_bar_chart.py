import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# plotting params
ax_fontdict={'fontsize': 18}
ticklabel_size = 15
save_it = True

# read data
df = pd.read_excel(('../fig3_maps_fig5_curr_vs_poten/AF_locations_from_papers/'
                    'Agroforestry Data Dec 2021_MERGED_METANALYSES.xlsx'))

# get meta-analysis cols
with open('./meta-analysis_cols.txt', 'r') as f:
    cols = [col.strip() for col in f.readlines()]
assert(len(cols) == 21)

# get study coverage sums
df['coverage'] = df.loc[:, cols].sum(axis=1)
df = df[df['coverage']>=1]

# print total study count
print('\n\nTOTAL NUMBER OF STUDIES COVERED BY >=1 META-ANALYSIS: %i \n\n' % len(df))

# make df for plotting
df_barplot = df.loc[:, ['study.id', 'coverage']]
df_barplot['count'] = 1
df_barplot = df_barplot.groupby(by='coverage').sum('count').reset_index()
df_barplot['coverage'] = np.int64(df_barplot['coverage'])

# make plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.bar(x=df_barplot['coverage'], height=df_barplot['count'])
ax.set_xlabel('coverage across meta-analyses', fontdict=ax_fontdict)
ax.set_ylabel('number of primary studies', fontdict=ax_fontdict)
ax.tick_params(labelsize=ticklabel_size)
fig.subplots_adjust(bottom=0.12, left=0.12, top=0.97, right=0.97)
if save_it:
    fig.savefig('FIGS1_primary_study_coverage_across_meta-analyses.png', dpi=700)


# print % studies covered only once
print('\n\n%0.2f%% OF STUDIES COVERED ONLY ONCE\n\n' % (
      df_barplot[df_barplot['coverage']==1]['count']/df_barplot['count'].sum()*100))

# print max coverage for any primary study
print('\n\n%i STUDIES HAVE THE MAX OBSERVED COVERAGE OF %i \n\n' % (
      df_barplot[df_barplot['coverage'] == df_barplot['coverage'].max()]['count'],
      df_barplot['coverage'].max()))


