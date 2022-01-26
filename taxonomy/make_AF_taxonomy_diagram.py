import plotly.graph_objects as go
from pySankey.sankey import sankey

# TODO:
    # What other taxonomies to incorporate?
    # How to merge others in?
    # Do a second plot like this to demo our prpoposed hierarchical typology
    # for NCS analysis??

# DATA SOURCES:
    # FAO info from: https://www.fao.org/forestry/agroforestry/89998/en/

# create label lists
labels = [
    # FAO level 2
    ['improved fallows',
     'taungya',
     'alley cropping (hedgerow intercropping)',
     'multilayer tree gardens',
     'multipurpose trees on crop lands',
     'plantation crop combinations',
     'home gardens',
     'trees in soil conservation and reclamation',
     'shelterbelts and windbreaks, live hedges',
     'fuelwood production',
     'trees on rangeland or pasture',
     'protein banks',
     'plantation crops with pasture and animals',
     'homegardens involving animals',
     'multipurpose woody hedgerows',
     'apiculture with trees',
     'aquaforestry'
     ],
    # FAO level 1
    ['agrisilviculture',
     'silvopasture',
     'agrosilvopasture'
    ],
]

label_colors = [
    # FAO level 1
    ['gray'] * 17,
    # FAO level 2,
    ['#fcdb03'] + ['#c93b1e'] + ['#3e9426'],
]

# create link lists
sources = [
    # FAO level 1 --> FAO level 2
    [*range(17)],
]
targets = [
    # FAO level 1 --> FAO level 2
    [17]*10 + [18]*3 + [19]*4
]

# create value lists
values = [
    # FAO level 1 --> FAO level 2
    [1]*17
]
# create color lists
link_colors = [
    # FAO level 1 --> FAO level 2
    ['#fcdb03']*10 + ['#c93b1e']*3 + ['#3e9426']*4,
]

# collapse all lists
labels = [list_item for sublist in labels for list_item in sublist]
label_colors = [list_item for sublist in label_colors for list_item in sublist]
sources = [list_item for sublist in sources for list_item in sublist]
targets = [list_item for sublist in targets for list_item in sublist]
values = [list_item for sublist in values for list_item in sublist]
link_colors = [list_item for sublist in link_colors for list_item in sublist]



# make Sankey diagram
fig = go.Figure(data=[go.Sankey(
        node = dict(
                  pad = 15,
                  thickness = 20,
                  line = dict(color = "black", width = 0.5),
                  label = labels,
                  color = label_colors
                ),
        link = dict(
                  source = sources,
                  target = targets,
                  value = values,
                  label=labels,
                  color=link_colors,
              ))])

fig.update_layout(title_text="Example Agroforestry Taxonomies", font_size=10)
fig.show()

fig.write_html("DRAFT_AF_taxonomy_comparison.html")
