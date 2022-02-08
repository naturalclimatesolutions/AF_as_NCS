# load packages
library(readxl)
library(sf)

# load my country-recoding script
source('../../../assorted_tools/recode_country_vector.r')

# load the Rosenstock et al. 2019 ("Making trees count: ...") database
df = read_excel('./MRV_AgroforestryDatabase_FINAL.xlsx',
                sheet='Database',
                skip=3)

# subset to just the desired output columns
cols.2.keep = c('Region',
                'Country',
                "mentioned  explicitly ? (1=Yes, 0=No)",
                "Section mentioned (1=mtigation, 2=adapatation, 3=both adaptation&mitigation)...13",
                "land management activities potentially AF? (1=Yes, 0=No)",
                "Section mentioned (1=mtigation, 2=adapatation, 3=both adaptation&mitigation)...15",
                "NDC_mentions Agroforestry explicitly ? (1=Yes, 0=No)",
                "NDC_section with explicit mention\r\n(1=mitigation, 2=adapatation, 3=adaptation&mitigation)",
                "NDC_potential mentions ? (1=Yes, 0=No)",
                "NDC_section with potential mention\r\n(1=mitigation, 2=adapatation, 3=adaptation&mitigation)",
                "mentions Agroforestry explicitly ? (1=Yes, 0=No)",
                "possible mentions of Agroforestry? (1=Yes, 0=No)")

cols.renamed = c('region',
                 'country',
                 'NCexp',
                 'NCexp_s',
                 'NCimp',
                 'NCimp_s',
                 'NDCexp',
                 'NDCexp_s',
                 'NDCimp',
                 'NDCimp_s',
                 'NAMAexp',
                 'NAMAimp')
subdf = df[cols.2.keep]
colnames(subdf) = cols.renamed

# combine NC and NDC explicit and implicit mentions into a single T/F col
subdf$NCmnt = (((subdf$NCexp) & (!is.na(subdf$NCexp))) *1) | (((subdf$NCimp) & (!is.na(subdf$NCimp))) *1)
subdf$NDCmnt = (((subdf$NDCexp) & (!is.na(subdf$NDCexp))) *1) | (((subdf$NDCimp) & (!is.na(subdf$NDCimp))) *1)
subdf$NAMAmnt = (((subdf$NAMAexp) & (!is.na(subdf$NAMAexp))) *1) | (((subdf$NAMAimp) & (!is.na(subdf$NAMAimp))) *1)
# correct so that rows that are NA for both implicit and explicit mentions
# are NA, rather than T or F
subdf[is.na(subdf$NCexp) & is.na(subdf$NCimp), ]$NCmnt = NA
subdf[is.na(subdf$NDCexp) & is.na(subdf$NDCimp), ]$NDCmnt = NA
subdf[is.na(subdf$NAMAexp) & is.na(subdf$NAMAimp), ]$NAMAmnt = NA



# get new col of the ISO3-recoded countries
subdf.country.ISO3 = recode_country_vector(subdf$country, allow_dups=F)

subdf['country_ISO3'] = subdf.country.ISO3

# load a shapefile of TNC country boundaries
shp = st_read('./NewWorldFile_2020.shp')

# drop (or otherwise handle) problem rows
keep.rows = !shp$CNTRY_NAME %in% c('Wake I.', 'Juan De Nova I.',
                                   'Glorioso Is.', 'Johnston Atoll',
                                   'Midway Is.', 'Virgin Is.', 'Jan Mayen',
                                   'West Bank')
palestine.geometry = st_union(shp[shp$CNTRY_NAME %in% c('West Bank', 'Gaza Strip'),])
shp = shp[keep.rows, ]
shp[shp$CNTRY_NAME == 'Gaza Strip',2:6] = NA
shp[shp$CNTRY_NAME == 'Gaza Strip',8:19] = NA
shp[shp$CNTRY_NAME == 'Gaza Strip',7] = 'Palestine'
shp[shp$CNTRY_NAME == 'Gaza Strip',20] = palestine.geometry

# get a new col of ISO3-recoded countries for the boundaries
# (already exists, but just to be safe)
shp.country.ISO3 = recode_country_vector(shp$CNTRY_NAME, allow_dups=F)

shp['country_ISO3'] = shp.country.ISO3

# merge Rosenstock data onto spatial data
outdf = merge(shp, subdf, on='country_ISO3', all.x=T)

# reproject ot WGS84
outdf = st_transform(outdf, 4326)

# write out
st_write(outdf, './rosenstock_et_al_2019_AF_NDCs_db.shp', append=F)
