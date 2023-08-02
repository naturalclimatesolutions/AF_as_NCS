### overview:
This repo contains code and some data files used for our 'state of the knowledge' perspective piece: "Priority Science Can Accelerate Agroforestry as a Natural Climate Solution" (Terasaki Hart et al. 2022).

Code was written by Drew Ellison Terasaki Hart,
and is available to freely distribute and modify, with proper
attribution, under the MIT License.

[![DOI](https://zenodo.org/badge/362674213.svg)](https://zenodo.org/badge/latestdoi/362674213)

# < PUT CITATION HERE >

Analyses were run in two major stages, so directories of content are organized to reflect this:
  1. data prep and analysis on Google Earth Engine (*directories containing this material are prepended with 'GEE_'*)
  2. local analysis and production of results and figures (*directories containing this material have names that indicate any figures their content is involved in creating*)

***NOTE:*** **All GEE code must be run in GEE. All other code is designed to be run within the directory in which it is saved (i.e., with the directory it is saved in set as the current working directory).**


### analysis includes various components:

**A.** analysis of overlap in original studies analyzed in 9 meta-analyses (produces Fig. S1)
**B.** analysis of uncertainty in C estimates (from Cardinael et al. 2018 data) and comparison to remotely sensed estimates (produces Figs. 2, S3, S4, S5)
**C.** maps comparing global AF rasters to known AF point localities (produces Fig. 3), and comparison between current and potential AF use (produces Fig. 5), and estimate of global AF aboveground C based on Lesiv et al. 2022 global AF map
**D.** demonstration of spatial and temporal resolution and precision issues for AF MRV methodologies (produces Fig. 4)


### workflow to reproduce results:

## data prep:

1. Download raw datasets from papers' supplemental materials.
2. Interactively run `./fig3_maps_fig5_curr_vs_poten/rosenstock_data/convert_rosenstock_data_to_shapefile.R` to add ISO3 3-digit country codes to Rosenstock (this requires my interactive `recode_country_vector.r` script, to make sure that none are incorrectly assigned) and then format and produce a shapefile from the Rosenstock data.
3. Download into `./fig3_maps_fig5_curr_vs_poten/AF_locations_from_papers` the latest verison of the draft GROA-AF carbon database (NOTE: version used for analysis was downloaded on 07/21/22).
4. Run `./fig3_maps_fig5_curr_vs_poten/AF_locations_from_papers/prep_AF_location_data.py` to process the raw GROA-AF geocoordinates and generate a cleaned-up, formatted shapefile ('AF_locations_from_meta-analyses.shp').
5. Ingest that shapefile into GEE.
6. Run `./GEE_code/prep_fig2_fig3_data.js` on GEE, with flags 'export_lesiv' and 'export_chapman' set to true and 'extract_agb_at_cardinael' set to false.
7. Download the resulting GeoTIFFs (to `./GEE_tif_output`) and shapefiles (to `./GEE_shp_output`), from Google Cloud storage and Google Drive, respectively.
8. Run `./GEE_tif_output/reproject_chapman_lesiv.sh1` to warp rasters from GEE's output EPSG:4326 to target global equal-area projection (ESRI:54012; world Eckert IV).
9. Manually choose from the current draft GROA-AF database an AF site with high-enough coordinate precision to be visually identified in Google Earth aerial imagery, and with decent-enough temporal coverage in the Google Earth aerial imagery archive to show temporal dynamics
10. Run `./GEE_code/get_fig4_imagery.js` on GEE to load, prep, and map the aerial basemap and satellite RGB maps used in Fig 4, as well as the purple precision radii and polygons to frame the data up for screenshotting; NOTE: THE CODE FROM THIS SCRIPT IS SHARED, BUT HAS BEEN REDACTED TO PRESERVE CONFIDENTIALITY OF THE SITE CHOSEN.
11. Take screenshots of each map at the same zoom level and framed within the framing polygons produced by the script (for Fig. 4b), and also at a coarser zoom level (for Fig. 4a), using the default Ubuntu screenshotting tool, and naming files by dataset and date.
12. Use Google Earth desktop app to create identical framing polygons and then capture screenshots of historical Google Earth aerial imagery in the identical spatial extent and at identical zoom, naming files by dataset and date. 
13. Manually download all PDFs from which Cardinael et al. 2018 collected data, review each PDF, and record or estimate date of data collection as precisely as possible, then save in an altered version of 'Cardinael_et_al_2018_ERL_Database_AFS_Biomass.xlsx', as column 'MEAS_YR' (naming new file 'Cardinael_et_al_2018_ERL_Database_AFS_Biomass_BT_DETH_MEAS_YR_ADDED.xlsx'). (NOTE: intials in filename refer to fact that this work was done by authors Bhuwan Thapa and Drew Terasaki Hart.)

## analysis component A:

1. Run `./figS1_paper_coverage_in_metaanalyses/make_paper_coverage_bar_chart.py` to calculate from the GROA-AF draft database summary statistics of coverage of primary literature in our 21 identified AF meta-analyses and to produce Fig. S1 ('FIGS1_primary_study_coverage_across_meta-analyses.pdf').

## analysis component B:

1. Run `./fig2_figS3_figS4_figS5_pubd_and_RS_C_analyses/compare_AGB_and_SOC_in_pubs_and_RS.py` manually, up to line 230, then manually run line 231 to export the prepped data to shapefile (to be imported into GEE).
2. Import the resulting file ('agb_pts_from_cardinael_2018.shp') import GEE.
3. Run `./GEE_code/prep_fig2_fig3_data.js` on GEE, with flags 'export_lesiv' and 'export_chapman' set to false and 'extract_agb_at_cardinael' set to true.
4. Download the resulting GEE output files from Google Drive ('agb_pts_from_cardinael_2018_\*').
5. Run `./fig2_figS3_figS4_figS5_pubd_and_RS_C_analyses/compare_AGB_and_SOC_in_pubs_and_RS.py` in full, to produce carbon density ridgline plots ('FIG2_C_density_practice_comp_plot.pdf'), comparison between SOC and AGB and between in situ and remotely sensed AGB values ('FIGS4_C_density_pub_rs_comp_plot.pdf'), and to analyze divergence between in situ and RS estimates as a function of temporal divergence ('FIGS5_regression_WHRC_Cardinael_stock_diff_vs_meas_yr_diff.pdf') and spatial precision ('FIGS6_regression_WHRC_Cardinael_stock_diff_vs_coord_precision.pdf').

## analysis component C:

1. Run `./fig3_maps_fig5_curr_vs_poten/map_and_analyze_current_and_potential_AF.py` to produce maps of current AF locations ('FIG3_AF_maps_and_known_AF_locs.pdf') and analysis of current vs. potential AF mitigation ('FIG5_curr_and_potent_AF_C_boxplots_and_curr_scat.pdf'). Some statistics for publication are printed to STDOUT.
2. Run `./fig3_maps_fig5_curr_vs_poten/stimate_global_AF_area_C_from_lesiv.py` to get estimates of global total AF land area and global total AF AGC according to Lesiv et al. 2022 (combined with the year-2000 AGB map from WHRC/GFW, which is a mismatch in year but a better comparator against Chapman's global AF AGC estimates derived from the same map).

## analysis component D:

1.Run `./fig4_spatial_temporal_chars/make_legend_for_MRV_radii.py` to load Cardinael et al. 2018 data and estimate spatial precision of each coordinate, and produce the legend image for Fig. 4a.
2. Open LibreOffice Draw, load all saved images, compose Fig. 4, and save as PDF (18.5x23.93", to ensure high resolution).
