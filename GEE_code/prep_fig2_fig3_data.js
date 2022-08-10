// BEHAVIORAL PARAMS AND BASIC SETTINGS
var export_lesiv = true;
var export_chapman = true;
var extract_agb_at_cardinael = true;

var palettes = require('users/gena/packages:palettes');

var working_scale = 30; //CHANGE THIS?

var total_n_pts = 10000;

var min_n_pts_per_biome = 100;

// load our standard countries dataset
var countries = ee.FeatureCollection('projects/gee-planet-test-308516/assets/TNC_country_bounds');

// zoom map all the way out, to import global data
Map.setZoom(0);

// create white map background for viz
Map.addLayer(ee.Image(0), {'palette': ['white'], 'opacity':0.95}, 'background');

////////////////////////////////////////////////////////////////////////////////////////////////
// MAP CHAPMAN DATA

var tip6 = ee.Image("users/MillieChapman/tip_all_6"),
    tip5 = ee.Image("users/MillieChapman/tip_all_5"),
    tip4 = ee.Image("users/MillieChapman/tip_all_4"),
    tip3 = ee.Image("users/MillieChapman/tip_all_3"),
    tip2 = ee.Image("users/MillieChapman/tip_all_2"),
    tip1 = ee.Image("users/MillieChapman/tip_all_1"),
    tic6 = ee.Image("users/MillieChapman/tic_all_6"),
    tic5 = ee.Image("users/MillieChapman/tic_all_5"),
    tic4 = ee.Image("users/MillieChapman/tic_all_4"),
    tic3 = ee.Image("users/MillieChapman/tic_all_3"),
    tic2 = ee.Image("users/MillieChapman/tic_all_2"),
    tic1 = ee.Image("users/MillieChapman/tic_all_1");


var chapman_crop = ee.ImageCollection([tic1,tic2,tic3,tic4,tic5,tic6]).max().rename('woody_C'); 

var chapman_pasture = ee.ImageCollection([tip1,tip2,tip3,tip4,tip5,tip6]).max().rename('woody_C');


// to combine both layers: 1.) stack (crop first!), 2.) take the first non-null
// this avoids dropout of pixels where there are crop but no pasture values, which even leads to dropout
// of whole nations, e.g., Philippines, Papua New Guinea (must also happen inversely, too?)
// also handles the data correctly in terms of letting higher-detail crop data take precedence
// (in line with original Chapman methods), but backfills with 

// also multiply by IPCC 0.47 C fraction, to go from Mg biomass/ha to Mg C/ha
var chapman_all_Mg_C_ha = ee.ImageCollection([chapman_crop, chapman_pasture])
  .reduce(ee.Reducer.firstNonNull())
  .selfMask()
  .multiply(ee.Image.constant(0.47))
  .rename('woody_C');
  
// convert each pixel's value (originally expressed in Mg biomass/ha) to Mg C/pix,
// by dividing by cell areas expressed in hectares,
// so that pixels can later be summed
var chapman_all = chapman_all_Mg_C_ha
  .multiply(ee.Image.pixelArea().divide(ee.Image.constant(10000)));


// add country borders
var empty = ee.Image().byte();
var outline = empty.paint({
  featureCollection: countries,
  color: 1,
  width: 0.5,
});
Map.addLayer(outline, {palette: '000000'}, 'country borders');

//var biomass_pal = "75322B,8e562d,b47b3e,da8c19,ef9e0b,ffc011,ffdb2d,ffe215,e3ef46,d5e400,c9d800,becc00,b4c200,B7B95B,B2B659,AFB457,ABB156,A6AE53,A3AB52,A1AA51,9FA950,9EA850,9CA74F,9BA64E,9AA54E,99A44D,95A24C,92A04A,909E49,8C9C48,8B9A47,869745,859745,839544,839543,819443,7E9241,7A8F40,778D3E,758C3E,758B3D,728A3C,71893C,70883B,6F873B,6D863A,6A8438,678237,648036,627E37,607D34,5E7B33,5A7831,577630,53742E,50722D,4F712C,4E702C,4C6F2B,4A6D2A,496D29,486C29,486C29,476B29,466A28,426827,3E6525,3B6323,3A6223,396222,386122,355F21,345E22,315C1F,305B1E,2C591D,2B581C,28561B,27551A,255419,245319,235218,225218,225118,215118,205017,1F4F17,1C4E16,1B4D15,1A4C15,194C14,184A14,164913,154812,124711,114610,114610,114610,114610";  
var biomass_pal = palettes.cmocean.Speed[7]
var chapman_biomass_vis = {"max":400,"min":0, 'palette': biomass_pal};
Map.addLayer(chapman_crop, chapman_biomass_vis, 'crop (Mg C)', 1);
Map.addLayer(chapman_pasture, chapman_biomass_vis, 'pasture (Mg C)', 1);
if (extract_agb_at_cardinael){
  Map.addLayer(chapman_all_Mg_C_ha.updateMask(chapman_all_Mg_C_ha.gt(0)), {"max":40,"min":0, 'palette': biomass_pal}, 'crop and pasture combined (Mg C/ha)', 1);
} else {
  Map.addLayer(chapman_all, {"max":25000,"min":0, 'palette': biomass_pal}, 'crop and pasture combined (Mg C)', 1);
}



////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT AGGREGATED LESIV DATA
if (export_lesiv){
  var lesiv = ee.Image('users/drewhart/lesiv_et_al_2022_forest_management');
  var lesiv = lesiv.eq(53).selfMask();
  Map.addLayer(lesiv, {min:0, max:1}, 'Lesiv et al. 2022');
  var roi = ee.Geometry.Polygon([[-179, 70], [-179,-70], [179,-70], [179,70], [-179,70]]);
   Export.image.toCloudStorage(
    {image: lesiv.unmask(-9999), 
    description: 'lesiv_3km_EPSG_4326',
    bucket: 'agroforestry',
    //region: roi,
    maxPixels: 900000000,
    fileDimensions: 45056, // >expected n pix on wider dim (lon), & divisble by shard size (256)
    crs: 'EPSG:4326',
    scale: 3000,
    fileFormat: 'GeoTIFF',
    });
    
  // also output the global AF land area, as estimated by Lesiv et al. map
  var lesiv_af_area = lesiv
    .multiply(ee.Image.pixelArea())
    .reduceRegions({collection: countries, 
                    reducer: ee.Reducer.sum(),
                    scale: lesiv.projection().nominalScale().getInfo(), 
                    crs: 'EPSG:3857'});
      //.reduceColumns(ee.Reducer.sum(), ['sum'])
      //.get('sum');
  Export.table.toDrive({collection: lesiv_af_area,
                        description: 'Lesiv_et_al_national_AF_areas',
                        fileFormat: 'SHP',
  });
  
  // also output the global AF carbon density, as estimated by combining the
  // Lesiv et al. map with the Woods Hole global C map (so that the result
  // is directly comparable to Chapman et al.)
  // load WHRC biomass data
  var whrc = ee.Image('users/tncwogis/Carbon/global_AGB_2000_30m_Mgha_V4');
  var whrc = whrc
    // convert to C (using 0.47, IPCC C fraction)
    .select('AGB_2000_Mgha').multiply(0.47).rename('woody_C')
    // mask to only Lesiv AF pixels
    .updateMask(lesiv);

  var lesiv_af_C = whrc
    // convert from Mg C/ha to Mg C/pixel
    .multiply(ee.Image.pixelArea().divide(ee.Image.constant(10000)))
    // sum by country
    .reduceRegions({collection: countries, 
                    reducer: ee.Reducer.sum(),
                    scale: whrc.projection().nominalScale().getInfo(), 
                    crs: 'EPSG:3857'});
      //.reduceColumns(ee.Reducer.sum(), ['sum'])
      //.get('sum');
  Export.table.toDrive({collection: lesiv_af_C,
                        description: 'Lesiv_et_al_national_AF_C_stocks',
                        fileFormat: 'SHP',
  });
}




////////////////////////////////////////////////////////////////////////////////////////////////
// EXPORT AGGREGATED CHAPMAN DATA
if (export_chapman){
  var roi = ee.Geometry.Polygon([[-179, 70], [-179,-70], [179,-70], [179,70], [-179,70]]);
  Export.image.toCloudStorage(
    {image: chapman_all_Mg_C_ha.select('woody_C').unmask(-9999), 
    description: 'chapman_3km_EPSG_4326',
    bucket: 'agroforestry',
    //region: roi,
    maxPixels: 900000000,
    fileDimensions: 45056, // >expected n pix on wider dim (lon), & divisble by shard size (256)
    crs: 'EPSG:4326',
    scale: 3000,
    fileFormat: 'GeoTIFF',
    });
}
////////////////////////////////////////////////////////////////////////////////////////////////
// EXTRACT   AGB DENSITY AT CARDINAEL ET AL. 2018 POINTS
if (extract_agb_at_cardinael){
  var card_pts = ee.FeatureCollection('projects/gee-planet-test-308516/assets/agb_pts_from_cardinael_2018');
  Map.addLayer(card_pts, {}, 'points from Cardinael et al. 2018');
    
  // tweaking code from: 
  // https://developers.google.com/earth-engine/tutorials/community/extract-raster-values-for-points
  var  bufferPoints = function(radius, bounds) {
    return function(pt) {
      pt = ee.Feature(pt);
      return bounds ? pt.buffer(radius).bounds() : pt.buffer(radius);
    };
  };

  var calcZonalStats = function(ic, fc) {
    // Map the reduceRegions function over the image collection.
    var results = ic.map(function(img) {
      // Reduce the image by regions.
      return img.reduceRegions({
        collection: fc,
        reducer: ee.Reducer.mean(),
        scale: working_scale,
        crs: 'EPSG:3857',
      });
    }).flatten();
    return results;
  };
  // buffer the points, then extract Chapman values
  // NOTE: BUFFERING TO 5m BECAUSE ANYTHING LESS THAN THAT STARTED RETURNING NO MEAN VALUES
  //       EVEN AT CELLS THAT CLEARLY OVERLAPPED WITH CHAPMAN DATA... GEE IS ODD
  var buffs = card_pts.map(bufferPoints(5, true));
  Map.addLayer(buffs, {}, 'buffered Cardinael et al. points');
  
  // prep each Chapman layer separately, as well as using the combined layer
  // NOTE: using system:time_start, since I have to add it anyhow, to code the separate layers in the output
  var chapman_crop_Mg_C_ha = chapman_crop
    .selfMask()
    .multiply(ee.Image.constant(0.47))
    .rename('woody_C')
    //.neighborhoodToArray(ee.Kernel.square(500, 'meters'))
    //.arrayReduce(ee.Reducer.mean(), [0,1])
    .set('system:time_start', ee.Date('2000-01-01').millis());
  var chapman_pasture_Mg_C_ha = chapman_pasture
    .selfMask()
    .multiply(ee.Image.constant(0.47))
    .rename('woody_C')
    //.neighborhoodToArray(ee.Kernel.square(500, 'meters'))
    //.arrayReduce(ee.Reducer.mean(), [0,1])
    .set('system:time_start', ee.Date('2010-01-01').millis());
  var chapman_all_for_stack = chapman_all_Mg_C_ha
    //.neighborhoodToArray(ee.Kernel.square(500, 'meters'))
    //.arrayReduce(ee.Reducer.mean(), [0,1])
    .set('system:time_start', ee.Date('2020-01-01').millis());
  //var chapman_stack = ee.ImageCollection([chapman_crop_Mg_C_ha,
                                          //chapman_pasture_Mg_C_ha,
                                          //chapman_all_for_stack,]);
  var chapman_stack = ee.ImageCollection([chapman_all_for_stack]);
  print(chapman_stack);
  
  //prep WHRC et al. data for extraction
  // NOTE: this is the dataset that Chapman is based on, prior to Chapman's filtering
  //       out pixels >=25% tree cover, so this will allow us to include known AF sites
  //       under dense-canopy AF systems as well
  var whrc = ee.Image('users/tncwogis/Carbon/global_AGB_2000_30m_Mgha_V4');
  var whrc_stack = ee.ImageCollection([whrc.select('AGB_2000_Mgha').multiply(0.47).rename('woody_C')]);
  Map.addLayer(whrc.updateMask(whrc.gt(0)), {"max":40,"min":0, 'palette': biomass_pal}, 'WHRC (Mg C/ha)', 1);

                                          
  // extract values
  var extract_chapman = calcZonalStats(chapman_stack, buffs);
  print('Chapman extracted', extract_chapman);
  var extract_whrc = calcZonalStats(whrc_stack, buffs);
  print('WHRC extracted', extract_whrc);

 // export it
 Export.table.toDrive({
  collection: extract_chapman,
  description: 'agb_pts_from_cardinael_2018_chapman_extract',
  fileFormat: 'SHP'
  });
 Export.table.toDrive({
  collection: extract_whrc,
  description: 'agb_pts_from_cardinael_2018_whrc_extract',
  fileFormat: 'SHP'
  });
}
