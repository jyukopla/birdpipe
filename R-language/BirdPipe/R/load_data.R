#' load_data
#'
#' Loads BirdPipe geojson data: recordings and species detections
#' @param BirdPipe.folder The name of the BirdPipe folder
#' @param threshold Classification threshold to be used to filter the data
#' @param geojson.folder The name of the geojson folder within the  BirdPipe data folder
#' @return A named list with elements $recordings and $detections
#' @return A BirdPipe data object
#' @examples
#' load_data(BirdPipe.folder = "BirdPipe_demo_data");
#' @importFrom jsonlite fromJSON
#' @importFrom sf read_sf
#' @importFrom sf st_coordinates
#' @export
load_data = function(BirdPipe.folder,
                     threshold = 0,
                     geojson.folder = "geojson"){
  filenames = list.files(paste0(BirdPipe.folder,"/",geojson.folder))
  filenames = paste0(BirdPipe.folder,"/",geojson.folder,"/",filenames)
  all.recordings = c()
  all.detections = c()
  for(filename in filenames){
    recordings = as.data.frame(sf::read_sf(filename))
    rownames(recordings) = recordings$audio_filename
    recordings$audio_filename = NULL
    recordings$species_detected = NULL
    ny = nrow(recordings)
    xy = sf::st_coordinates(recordings$geometry)
    recordings$longitude = xy[,1]
    recordings$latitude = xy[,2]
    recordings$geometry = NULL
    detections = c()
    for(i in 1:ny){
      models = jsonlite::fromJSON(recordings$models[i])
      nm  = nrow(models)
      for(j in 1:nm){
        da = models$detections[[1]]
        if(length(da)>0){
          da$model_name = models$model_name[j]
          da$audio_filename = rownames(recordings)[i]
        }
      }
      all.detections = rbind(all.detections,da)
    }
    recordings$models = NULL
    all.recordings = rbind(all.recordings,recordings)
  }
  all.detections = subset(all.detections, probability>threshold)

  all.recordings$segment.latitude = NA
  all.recordings$segment.longitude = NA
  all.recordings$segment = paste(all.recordings$device, all.recordings$segment)
  all.recordings$segment = as.factor(all.recordings$segment)
  for(segment in levels(all.recordings$segment)){
    sel = which(all.recordings$segment==segment)
    me.lat = mean(all.recordings$latitude[sel],na.rm = T)
    me.lon = mean(all.recordings$longitude[sel],na.rm = T)
    all.recordings$segment.longitude[sel] = me.lon
    all.recordings$segment.latitude[sel] = me.lat
  }
  no.coordinates = which(is.na(all.recordings$segment.latitude))
  if(length(no.coordinates)>0){
    print("No coordinates: ")
    print(as.character(unique(all.recordings[no.coordinates,]$segment)))
  }

  BirdPipe = list()
  BirdPipe$recordings = all.recordings
  BirdPipe$detections = all.detections
  BirdPipe$recordings$device = as.factor(BirdPipe$recordings$device)
  BirdPipe$detections$species_name = as.factor(BirdPipe$detections$species_name)
  BirdPipe$detections$scientific_name = as.factor(BirdPipe$detections$scientific_name)
  return(BirdPipe)
}
