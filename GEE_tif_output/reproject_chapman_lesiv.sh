gdalwarp -t_srs ESRI:54012 -srcnodata -9999 -dstnodata -999 -r cubic chapman_3km_EPSG_4326.tif chapman_3km_ESRI_54012.tif

gdalwarp -t_srs ESRI:54012 -srcnodata -9999 -dstnodata -999 -r nearest lesiv_3km_EPSG_4326.tif lesiv_3km_ESRI_54012.tif
