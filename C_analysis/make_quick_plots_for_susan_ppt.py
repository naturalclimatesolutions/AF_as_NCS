# NOTE: ASSUMES I'VE ALREADY LOADED AND PREPPED DATA USING
#       MAIN AF AS NCS C-ANALYSIS SCRIPT

#stock_var = 'stock_change'
stock_var = 'stock'
save_it = True
var_dict = {
            'age_sys': 'system age',
            'dens': 'stem density',
           }
var_axlabel_dict = {'age_sys': 'system age ($yr$)',
                    'dens': 'density ($stems\  ha^{-1}$)',
                   }
for var in var_dict.keys():
    fig, axs = plt.subplots(3,1, figsize=(8,13))
    for i, pool in enumerate(['agb', 'bgb', 'soc']):
        ax = axs[i]
        sns.scatterplot(var,
                        stock_var,
                        hue='practice',
                        hue_order=sorted(np.unique(all['practice'])),
                        s=45,
                        alpha=0.8,
                        data=all[all['var'] == pool],
                        ax=ax,
                        legend=i==2,
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
        fig.savefig('%s_scatterplots_EUROPE_CONF.png' % var, dpi=700)
