# CONTAINS: code and small static data files used for our 'state of the knowledge' lit review on agroforestry as an NCS (Terasaki Hart et al. 2022)

**Analyses include**:

A. analysis of overlap in original studies analyzed in 9 meta-analyses
B. analysis of uncertainty in C estimates (from Cardinael et al. 2018 data) and comparison to remotely sensed estimates
C. maps comparing global AF rasters to known AF point localities, and comparison between current and potential AF use
D. figure demonstrating spatial and temporal resolution and precision issues for AF MRV methodologies
E. estimate of global AF aboveground C based on Lesiv et al. 2022 global AF map


# WORKFLOWS:

## preparatory:

1. Download raw datasets.
2. Download into ./AF_locations_from_papers the latest verison of known AF locations (extracted for GROA-AF carbon database; downloaded 07/19/22).
3. Run ./AF_locations_from_papers/prep_AF_location_data.py to process the raw GROA-AF geocoordinates and generate a cleaned-up, formatted shapefile ('AF_locations_from_meta-analyses.shp').
4. Ingest that shapefile into GEE.
5. Run 'prep_fig2_fig3_data.js' with flags 'export_lesiv', 'export_chapman', and 'extract_agb_at_cardinael' set to true.
6. Download the resulting GeoTIFFs (to external hard drive) and shapefiles, from Google Cloud storage and Google Drive, respectively.
7. Navigate to directory with downloaded GeoTIFFs, then run 'reproject_chapman_lesiv.sh' to project from GEE's output EPSG:4326 to target global equal-area projection (ESRI:54012; world Eckert IV).
8. Manually download all PDFs from which Cardinael et al. 2018 collected data, review each PDF, and record or estimate date of data collection to best precision possible, then save in an altered version of the Cardinael et al. supplemental data.


## analysis A:

1. Run ./make_paper_coverage_bar_chart.py to calculate summary statistics of coverage of primary literature in our 21 identified AF meta-analyses and to produce Fig. S1 ('FIGS1_primary_study_coverage_across_meta-analyses.png').

## analysis B:

1. Run ./compare_AGB_and_SOC_in_situ_and_RS.py to produce comparison between SOC and AGB and between in situ and remotely sensed AGB values ('FIG2_C_density_pub_rs_comp_plot.png'), to analyze divergence between in situ and RS estimates as a function of temporal divergence ('FIGS3_regression_WHRC_Cardinael_stock_diff_vs_meas_yr_diff.png'), and to plot AGC, BGC, and SOC as a function of stem density ('FIGS2_C_vs_stem_density_scatters.png').


## analysis C:

1. Run ./map_and_analyze_current_and_potential_AF.py to produce maps of current AF locations ('FIG3_AF_maps_and_known_AF_locs.png') and analysis of current vs. potential AF mitigation ('FIG5_curr_and_potent_AF_C_boxplots_and_curr_scat.png'). Some statistics for publication are printed to STDOUT.


## analysis D:

1. Manually choose an AF site with high-enough coordinate precision to be visually identified in Google Earth aerial imagery, and with decent-enough temporal coverage in the Google Earth aerial imagery archive to show temporal dynamics
2. Run 'get_fig4_data.js' on GEE to load, prep, and map the aerial basemap and satellite RGB maps used in Fig 4, as well as the purple precision radii and polygons to frame the data up for screenshotting; NOTE: SCRIPT NOT MADE PUBLIC TO MAINTAIN CONFIDENTIALITY OF FARM LOCATION
3. Take screenshots of each map at the same zoom level and framed within the framing polygons produced by the script (for Fig. 4b), and also at a coarser zoom level (for Fig. 4a), using the default Ubuntu screenshotting tool, and naming files by dataset and date.
4. Use Google Earth desktop app to create identical framing polygons and then capture screenshots of historical Google Earth aerial imagery in the identical spatial extent and at identical zoom, naming files by dataset and date. 
5.Run ./make_legend_for_MRV_radii.py to load Cardinael et al. 2018 data and estimate spatial precision of each coordinate, and produce the legend image for Fig. 4a.
6. Open LibreOffice Draw, load all saved images, compose Fig. 4, and save as PNG (18.5x23.93", to ensure high resolution).
7. Run 'adjust_brightness_contrast_fig4.sh' to increase the brightness and contrast of Figure 4 a bit, to make land features and land changes more discernible in publication.


## analysis E:
- ???? Run script to calculate global Lesiv C estimate.
