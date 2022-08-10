
var roi = ee.Geometry.Rectangle(
  [0, 100,
   0, 100]);  // NOTE: REDACTED FOR CONFIDENTIALITY
  
var start = '2020-01-01T00:00';
var end = '2022-04-05T00:00';

// include AF locations on map?
var include_af_locs = false;

// stretching function taken from Rodrigo E. Principe: 
// https://gis.stackexchange.com/questions/259072/google-earth-engine-different-stretch-options
var stretch_std = function(i, n_std) {
  var mean = i.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: i.geometry(),
    bestEffort: true})
  var std = i.reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: i.geometry(),
    bestEffort: true})
  var min =     mean.map(function(key, val){
      return ee.Number(val).subtract(ee.Number(std.get(key)).multiply(n_std))
    }).getInfo()
  var max = mean.map(function(key, val){
      return ee.Number(val).add(ee.Number(std.get(key)).multiply(n_std))
    }).getInfo()

  return {vmin: min, vmax: max}
}



var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
  .filterBounds(roi)
  .filterDate(start, end)
  // get least cloudy annual single scene
  .sort('CLOUD_COVER');
// NOTE: explorted cloud-sorted imagery, then found Landsat image nearly aligned with Google's latest
// imagery (Google: 2021-02; Landsat: 2021-03-10)
// and also close to the Sentinel image selected below (2021-02-24)
var landsat = landsat.filterDate('2021-03-09T23:59', '2021-03-11T00:00');
print(landsat);
var s = stretch_std(landsat.first(), 3);
var min = ee.Number(ee.List([s.vmin.B4, s.vmin.B3, s.vmin.B2]).reduce(ee.Reducer.mean()));
var max = ee.Number(ee.List([s.vmax.B4, s.vmax.B3, s.vmax.B2]).reduce(ee.Reducer.mean()));
var viz = {bands:["B4", "B3", "B2"],
           min: 0,//min.getInfo(),
           max: [0.4, 0.4, 0.5],//max.getInfo(),
           gamma: [1.2, 1.2, 1],
};
Map.addLayer(landsat.first(), viz, 'landsat');

var sentinel = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(roi)
  .filterDate(start, end)
  .sort('CLOUDY_PIXEL_PERCENTAGE');

// NOTE: explorted cloud-sorted imagery, then found one aligned with Google's latest
// imagery (Google: 2021-02; Sentinel: 2021-02-24 T16:05)
var sentinel = sentinel.filterDate('2021-02-23T23:59', '2021-02-25T00:00');
print(sentinel);
var s = stretch_std(sentinel.first(), 3);
var min = ee.Number(ee.List([s.vmin.B4, s.vmin.B3, s.vmin.B2]).reduce(ee.Reducer.mean()));
var max = ee.Number(ee.List([s.vmax.B4, s.vmax.B3, s.vmax.B2]).reduce(ee.Reducer.mean()));
var viz = {bands:["B4", "B3", "B2"],
           min: 0,//min.getInfo(),
           max: 3000,//max.getInfo(),
           gamma: 1.2,
};
Map.addLayer(sentinel.first(), viz, 'sentinel');

var planet = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/americas")
  .filterBounds(roi)
  .filterDate(start, end);
print(planet);
var viz = {bands:["R", "G", "B"],
           min: 0,
           max: 2600,
           gamma: 1.2,
};
// NOTE: grabbing the 2021-02 Planet image, to align with Google latest imagery (2021-02)
var planet = planet.filterDate('2021-01-31T00:00', '2021-03-01T00:00');
Map.addLayer(planet.first(), viz, 'planet');

Map.setOptions('SATELLITE');
print(roi.centroid().coordinates().getInfo());
Map.setCenter(roi.centroid().coordinates().getInfo()[0],
              roi.centroid().coordinates().getInfo()[1],
              17);

// add circles of different sizes (to show coordinate uncertainty)
var pt = ee.Geometry.Point([0, 100]); // NOTE: REDACTED FOR CONFIDENTIALITY
var decimal_1 = pt.buffer(1110, 0.001); // precision of decimal degrees with 1 decimal place
var decimal_2 = pt.buffer(111, 0.001); // precision of decimal degrees with 2 decimal places
var decimal_3 = pt.buffer(11.1, 0.001); // precision of decimal degrees with 3 decimal places
var decimal_1_fc = ee.FeatureCollection([decimal_1]);
var decimal_2_fc = ee.FeatureCollection([decimal_2]);
var decimal_3_fc = ee.FeatureCollection([decimal_3]);
var empty = ee.Image().byte();
var decimal_1_outline = empty.paint({
  featureCollection: decimal_1_fc,
  color: 1,
  width: 5
});
var decimal_2_outline = empty.paint({
  featureCollection: decimal_2_fc,
  color: 1,
  width: 5
});
var decimal_3_outline = empty.paint({
  featureCollection: decimal_3_fc,
  color: 1,
  width: 5
});
Map.addLayer(decimal_1_outline, {palette: '#f7ebffaa'}, '1 decimal places');
Map.addLayer(decimal_2_outline, {palette: '#e3baffaa'}, '2 decimal places');
Map.addLayer(decimal_3_outline, {palette: '#b15aedaa'}, '3 decimal places');


// add known AF locations
if (include_af_locs){
  var af_locs = ee.FeatureCollection('projects/gee-planet-test-308516/assets/AF_locations_from_meta-analyses');
  Map.addLayer(af_locs, {}, 'AF locations from studies in meta-analyses');
}


// add rectangles I used to 'frame' the screenshots
// (lined up LL corner of screenshot with LR corner of long, tall rectangle;
//  line up UR corner of screenshot with LL corner of UR, smallest rectangle)
var frame = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      },
      {
        "type": "rectangle"
      },
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.MultiPolygon(
        [[[[0, 100],  // NOTE: REDACTED FOR CONFIDENTIALITY
           [0, 100],
           [0, 100],
           [0, 100]]],
         [[[0, 100],
           [0, 100],
           [0, 100],
           [0, 100]]]], null, false);
Map.addLayer(frame, {}, 'screenshot frame');
