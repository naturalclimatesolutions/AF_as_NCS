# import libraries
import circlify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re

# TODO:
    # decide if using
    # add piecharts inside showing % NDC target?
    # get rid of axis lines, etcA
    # make circle bounds colored instead of full face


def determine_first_n_sorted_vals_in_top_frac(vals, top_frac=0.65):
    return np.where(np.nancumsum(vals)/np.nansum(vals)>=top_frac)[0][0] + 1


def prep_circle_plot_data(df, col, potent_col, top_n=0.65):

    data = [{'id': 'World',
             'datum': np.nansum(df[col]),
             'children':[],
            }]
    for cont in df['cont'].unique():
        cont_child = {'id': cont,
                 'datum': np.nansum(df.loc[df['cont'] == cont, col]),
                 'children': [],
                }
        # get top n countries, by potential
        sorted = df[df.cont==cont].sort_values(by=potent_col, ascending=False)
        if np.issubdtype(type(top_n), np.integer):
            pass
        else:
            top_n_rows = determine_first_n_sorted_vals_in_top_frac(sorted[potent_col],
                                                          top_frac=top_n)
        assert np.issubdtype(type(top_n_rows), np.integer), type(top_n_rows)

        for n, cntry in enumerate(sorted.iloc[:top_n_rows, :]['NAME_EN']):
            cntry_child = {'id': cntry,
                        'datum': sorted.iloc[n,:][col]}
            cont_child['children'].append(cntry_child)
        # add remaining countries as a single entity
        if len(sorted) > top_n_rows:
            etc_cntry_child = {'id': '+%i' % (len(sorted)-top_n_rows),
                               'datum': np.nansum(sorted.iloc[top_n_rows:,:][col]),
                              }
            cont_child['children'].append(etc_cntry_child)
        data[0]['children'].append(cont_child)

    return data


def make_circle_plot(df, curr_col, potent_col, top_n=0.65):

    # set continent colors
    colors = ['#d73027',
              '#fc8d59',
              '#fee090',
              '#ffffff',
              '#bfddff',
              '#0d488c',
              '#4575b4',
             ]
    cont_colors = dict(zip(df.cont.unique(), colors))

    # set NDC colors
    ndc_colors = {'no': '#877f78',
                  'yes': '#3cb55e',
                 }

    # create figure with 2 subplots and a gridspec
    fig, axs = plt.subplots(figsize=(14,28))
    gs = gridspec.GridSpec(100, 200, figure=fig)

    for n, col in enumerate([curr_col, potent_col]):
        # create the circles' data structure
        data = prep_circle_plot_data(df, col, potent_col, top_n=top_n)

        # compute the circle positions thanks to the circlify() function
        circles = circlify.circlify(data,
                                show_enclosure=False,
                                target_enclosure=circlify.Circle(x=0, y=0, r=1))

        if n == 0:
            n_subaxs = int(np.nansum(df[col])/np.nansum(df[potent_col])*100)
            #ax = fig.add_subplot(gs[int(49-(n_subaxs/2)):int(50+(n_subaxs/2)),
            #                        int(49-(n_subaxs/2)):int(50+(n_subaxs/2))])
            ax = fig.add_subplot(gs[:, :100])
        else:
            ax = fig.add_subplot(gs[:, 100:])
        ax.axis('off')

        # Title
        ax.set_title(('Current total ag woody C (left) and feasible potential '
                  '(right) (in Mg)'))

        # Remove axes
        ax.axis('off')

        # Find axis boundaries
        lim = max(
            max(
                abs(circle.x) + circle.r,
                abs(circle.y) + circle.r,
            )
            for circle in circles
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Print circle the highest level (continents):
        for circle in circles:
            x, y, r = circle
            if circle.level == 2:
                color = cont_colors[circle.ex['id']]
            elif circle.level==3:
                if not re.search('\+\d+$', circle.ex['id']):
                    color = ndc_colors[df.loc[df.NAME_EN == circle.ex['id'],
                                              'NDC'].values[0]]
                else:
                    color = '#f2f0e9'
            else:
                color = '#f2f0e9'
            ax.add_patch(plt.Circle((x, y),
                                    r,
                                    alpha=1-(0.5*(circle.level==2)),
                                    linewidth=2,
                                    color=color))

        # Print circle and labels for the highest level:
        for circle in circles:
            if circle.level == 3:
                x, y, r = circle
                label = circle.ex["id"]
                ax.add_patch(plt.Circle((x, y), r, alpha=0.5, linewidth=2,
                                     color="#69b3a2"))
                ax.annotate(label, (x,y ), ha='center', color="black", fontsize=8)

        # Print labels for the continents
        #text_xy_coords = {'N. Am.': (0.4, 0.95),
        #                  'C. Am./Car.': (0.2, 0.95),
        #                  'S. Am.': (0.1, 0.9),
        #                  'Eur.': (0.05, 0.8),
        #                  'Af.': (0.05, 0.3),
        #                  'Oceania': (0.15, 0.05),
        #                  'Asia': (0.4, 0.1),
        #                 }
        for circle in circles:
            if circle.level == 2:
                x, y, r = circle
                label = circle.ex["id"]
                #txy = text_xy_coords[circle.ex['id']]
                #ax.annotate(label, (x,y ) ,
                            #xytext=[(0.5*n)+txy[0], txy[1]],
                            #va='center',
                            #ha='center',
                            #alpha=0.5,
                            #bbox=dict(facecolor='white',
                            #          edgecolor='black',
                            #          boxstyle='round',
                            #          pad=.5)
                           #)
    plt.axis('off')
