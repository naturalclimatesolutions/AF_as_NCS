import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from venn import venn, pseudovenn

# read data
df = pd.read_excel('./Paper coverage across reviews.xlsx')

# get meta-analysis cols
metas = [*df.columns[8:-1]]

# get study coverage values
subdf = df.loc[:, metas]

# convert to dict
venn_data = {}
for meta in metas:
    venn_data[meta] = set(subdf[meta].dropna().index)

# plot Venn (can only do 6 sets at a time!
plt.close('all')
fig, axs = plt.subplots(1,2)
ax1, ax2 = axs
ax_1_metas = metas[:6]
ax_2_metas = metas[:3] + metas[6:]
pseudovenn({k:v for k,v in venn_data.items() if k in ax_1_metas},
           cmap='plasma', ax=ax1)
pseudovenn({k:v for k,v in venn_data.items() if k in ax_2_metas},
           cmap='plasma', ax=ax2)
plt.show()


# get number of metas containing each study
subdf[pd.isnull(subdf)] = 0
counts = np.sum(subdf, axis=1)
# print percents of studies covered by 1, 2, ..., all meta-analyses
for i in range(2, len(metas)+1):
    print('---------------------\n')
    print('%0.1f%% of studies contained in at least %i meta-analyses\n\n' % (
                                                    100*np.mean(counts>=i), i))
