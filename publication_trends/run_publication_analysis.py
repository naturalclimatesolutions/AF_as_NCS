import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter as C
import statsmodels.api as sm
import re, os

# file lists
file_patt = '^%s_results_\d+\.txt\.?t?s?v?$'
af_files = [f for f in os.listdir('.') if re.search(file_patt % 'AF', f)]
ncs_files = [f for f in os.listdir('.') if re.search(file_patt % 'NCS', f)]
af_ncs_files = [f for f in os.listdir('.') if re.search(file_patt % 'AF_NCS',f)]


results_file_lists = { 'AF': af_files,
                       'CC': ncs_files,
                       'AF and CC': af_ncs_files}
terms_file_lists = {'AF': './search_terms_agrofor.txt',
                    'CC': './search_terms_ncs.txt',
                    'AF and CC': './search_terms_agrofor_ncs.txt'
                   }

all_yrs = []
dfs = {}
yr_cts = {}
# concat tables
for topic, file_list in results_file_lists.items():
    print('\n\n%s\n------------------------------------------\n\n' % topic)
    dfs_list = []
    for f in file_list:
        print(f)
        subdf = pd.read_csv(f, sep='\\t')
        subdf = subdf.loc[:, ['PT', 'PY']]
        assert np.issubdtype(subdf['PY'].dtype, np.number)
        dfs_list.append(subdf)
    df = pd.concat(dfs_list)
    df = df.loc[np.invert(pd.isnull(df['PY'])),]
    dfs[topic] = df
    cts = C(df['PY'])
    yr_cts[topic] = {}
    tot_ct = np.sum([*cts.values()])
    for yr, ct in cts.items():
        if yr < 2022:
            ct_vect = []
            ct_vect.append(ct)
            ct_vect.append(ct/tot_ct)
            yr_cts[topic][yr] = ct_vect
            all_yrs.append(yr)
        # drop 2022, or combine with 2021?
        else:
            pass

# plot trends
plt.close('all')
fig, axs= plt.subplots(1,2)
ax1, ax2 = axs
ax1.set_title('annual count')
ax2.set_title('annual count normalized\nby total count')
colors = {'AF': '#9999dd',
          'CC': '#dd9999',
          'AF and CC': '#dd55dd',
         }
for topic, cts in yr_cts.items():
    plot_yrs = []
    plot_cts = []
    plot_norm_cts = []
    for yr, ct_vect in cts.items():
        plot_yrs.append(yr)
        plot_cts.append(ct_vect[0])
        plot_norm_cts.append(ct_vect[1])
    for yr in all_yrs:
        if yr not in plot_yrs:
            plot_yrs.append(yr)
            plot_cts.append(0)
            plot_norm_cts.append(0)
    sort_idxs = np.argsort(plot_yrs)
    plot_yrs = np.array(plot_yrs)[sort_idxs]
    plot_cts = np.array(plot_cts)[sort_idxs]
    plot_norm_cts = np.array(plot_norm_cts)[sort_idxs]
    ax1.plot(plot_yrs, plot_cts, '.', label=topic, color=colors[topic])
    ax2.plot(plot_yrs, np.log(plot_norm_cts), '-', label=topic,
             color=colors[topic])
    # fit exponential rate
    y = np.array(np.log(plot_cts))
    data = np.array([*zip(y, plot_yrs)])
    data = data[np.invert(np.isinf(data[:,0]))]
    y = data[:,0]
    x = data[:,1]
    #curve_fit = np.polyfit(x, y, 1)
    #y_fitted = np.exp(curve_fit[0]) * np.exp(curve_fit[1]*x)
    #ax1.plot(x, y_fitted, color=colors[topic])
    #X = data[:,1]
    X = sm.add_constant(x)
    mod = sm.OLS(y, X).fit()
    params = mod.params
    #pred_xs = np.linspace(np.min(all_yrs), np.max(all_yrs))
    pred_xs = sm.add_constant(np.sort(all_yrs))
    pred_ys = np.exp(mod.params[0] + mod.params[1]*pred_xs)
    #pred_ys = mod.predict(pred_xs)
    ax1.plot(pred_xs[:,1], pred_ys, color=colors[topic])
    print('\n\nTOPIC: %s\n\n' % topic)
    print('fitted rate: %0.3f (95%%CI: %0.3f - %0.3f)' % (mod.params[0],
                                                mod.conf_int()[0][0],
                                                mod.conf_int()[0][1]))

ax1.legend()
ax2.legend()
plt.show()


#import requests

#from wos import WosClient
#import wos.utils
#with WosClient('drew.hart', 'TOR0t0uga$') as client:
    #print(wos.utils.query(client, 'TS=natural climate solutions'))


#from scholarly import scholarly
#from scholarly import ProxyGenerator
#pg = ProxyGenerator()
#pg.FreeProxies()
#scholarly.use_proxy(pg)
#
#topic_pub_years = {}
#for topic, terms_file in terms_file_lists.items():
#    with open(terms_file, 'r') as f:
#        terms = f.read().strip()
#    pub_years = []
#    pubs = scholarly.search_pubs(terms)
#    for pub in pubs:
#        try:
#            pub_years.append(pub['bib']['pub_year'])
#        except Exception as e:
#            print('DIDN\'T WORK FOR PUB %s\n\nERROR THROWN:\n\n\t%s' % (
#                                                    pub['bib']['title'], e))
#    topic_pub_years[topic] = pub_years
#

