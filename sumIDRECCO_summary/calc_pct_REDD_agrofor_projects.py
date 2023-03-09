import pandas as pd
import numpy as np
import re

df = pd.read_excel('./IDRECCO_Ver4-2_All Table_20220826.xlsx',
                   sheet_name='1. Project')
deets = df['Details for Afforestation/Reforestation activity']
hits = [re.search('agroforestry', d) if pd.notnull(d) else None for d in deets]
pct = np.mean([hit is not None for hit in hits])
print(("\n\n%0.3f%% of IDRECCO REDD+ projects list 'agroforestry' "
       "as an activity detail.\n\n") % (100*pct))
