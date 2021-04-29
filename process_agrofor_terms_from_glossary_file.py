import numpy as np
import pandas as pd
import re

gloss = pd.read_excel('./Glossary of Agroforestry Search Terms.xlsx', 'DETH')

agrofor_terms = gloss.iloc[0,2]
agrofor_terms = re.sub('“', '', agrofor_terms)
agrofor_terms = re.sub('”', '', agrofor_terms)

with open('search_terms.txt', 'w') as f:
    f.write('TS='+agrofor_terms)

print('\ndone')
