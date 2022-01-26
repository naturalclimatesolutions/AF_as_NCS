import pandas as pd
import numpy as np
import re

for col, sheet in {'mean_and_sd_SOC_rate': 'Cardinael_table_2',
                   'mean_and_sd_AGB_rate': 'Cardinael_table_3',
                   'mean_and_sd_BGB_rate': 'Cardinael_table_3',
                  }.items():
    vals = pd.read_excel('./cardinael_2008_data.xlsx',
                         sheet_name=sheet).loc[:, col]
    means = []
    sds = []
    for val in vals:
        try:
            nums = re.findall('-?\d*\.\d', val)
            if len(nums) == 2:
                means.append(float(nums[0]))
                sds.append(float(nums[1]))
            else:
                pass
                # analysis makes no sense for estimates without SDs b/c N=1
                #means.append(float(nums[0]))
                #sds.append(0)
        except:
            pass
        assert len(means) == len(sds)

    means = np.array(means)
    sds = np.array(sds)
    covs = sds/means
    print('-'*80)
    print(col)
    print('>=50%')
    print('\t:', np.mean(covs >= 0.5))
    print('>=75%')
    print('\t:', np.mean(covs >= 0.75))
    print('>=100%')
    print('\t:', np.mean(covs >= 1))
    print('\n'*4)
