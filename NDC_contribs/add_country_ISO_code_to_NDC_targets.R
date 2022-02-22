# load packages
library(readxl)
library(sf)

# load my country-recoding script
source('../../assorted_tools/recode_country_vector.r')

# load the NDC targets file
df = read_excel('./Pathway mitigation potential and NDC targets.xlsx')
# get rid of row 1, with units in it
df = df[2:nrow(df),]
# get rid of Alaska
df = df[df$CountryGeography != 'Alaska (United States)',]

# get new col of the ISO3-recoded countries
iso3 = recode_country_vector(df$CountryGeography, allow_dups=F)
df['iso3'] = iso3

# write out as CSV
write.csv(df,
          'pathway_mitigation_potential_and_NDC_targets_with_ISO3.csv',
          row.names=F)
