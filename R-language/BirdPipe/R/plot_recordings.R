#' plot_recordings
#'
#' Plots BirdPipe recordings as timeline and on a map
#' @param BirdPipe.data A BirdPipe data object
#' @param show.timeline Boolean variable (TRUE/FALSE) describing whether timeline is to be plotted
#' @param show.map Boolean variable (TRUE/FALSE) describing whether map is to be plotted
#' @param background.map optional terra/SpatRaster object to be plotted as background map
#' @param show.number.of.species Boolean variable (TRUE/FALSE) describing whether number of detected species is shown
#' @param threshold Classification threshold to be used to filter the data (influences number of detected species)
#' @param show.segment.id Boolean variable (TRUE/FALSE) describing whether segment id:s are shown
#' @param number.of.species.color color for plotting number of species
#' @param segment.id.color color for plotting segmend id:s
#' @param jitter the amount of random jitter added to spatial locations to avoid their possible overlap (proportional to data extent)
#' @param spatial.buffer size of margins added to the map (proportional to data extent)
#' @return NULL
#' @examples
#' BirdPipe.data = load_data(BirdPipe.folder = "BirdPipe_demo_data");
#' plot_recordings(BirdPipe.data = BirdPipe.data)
#' @importFrom terra plot
#' @export
plot_recordings = function(BirdPipe.data,
                           show.timeline = TRUE,
                           show.map = TRUE,
                           background.map = NULL,
                           show.number.of.species = FALSE,
                           threshold = 0,
                           show.segment.id = FALSE,
                           number.of.species.color = "white",
                           segment.id.color = "white",
                           jitter = 0,
                           spatial.buffer = 1){

  detections = droplevels(BirdPipe.data$detections[BirdPipe.data$detections$probability>=threshold,])

  if(show.timeline){
    no.coordinates = is.na(BirdPipe.data$recordings$latitude)
    plot(BirdPipe.data$recordings$timestamp,
         BirdPipe.data$recordings$device,
         pch = 16, col=BirdPipe.data$recordings$segment,
         xlab ="",ylab="",yaxt="n",
         main = "Recording timeline")

    points(BirdPipe.data$recordings$timestamp[no.coordinates],
           BirdPipe.data$recordings$device[no.coordinates],
           pch = 16,col="white",cex=0.4)
    axis(2,at=1:length(levels(BirdPipe.data$recordings$device)),
         labels = levels(BirdPipe.data$recordings$device),
         las=2, cex.axis=0.7)
  }

  if(show.map){
    segments = levels(BirdPipe.data$recordings$segment)
    lonlatnm = matrix(NA,nrow=length(segments),ncol=4)
    rownames(lonlatnm) = segments
    colnames(lonlatnm) = c("longitude","latitude","recordings","species")
    for(segment in segments){
      sel = which(BirdPipe.data$recordings$segment==segment)
      lonlatnm[segment,c(1,2)] = as.numeric(BirdPipe.data$recordings[sel[1],c("segment.longitude","segment.latitude")])
      lonlatnm[segment,3] = length(sel)
      sel2 = detections$audio_filename %in% rownames(BirdPipe.data$recordings)[sel]
      lonlatnm[segment,4] = length(unique(detections$species_name[sel2]))
    }
    dlat = 110.574
    dlon = 111.320*cos(2*pi*mean(lonlatnm[,2])/360)
    mi.lon = min(lonlatnm[,1])
    ma.lon = max(lonlatnm[,1])
    mi.lat = min(lonlatnm[,2])
    ma.lat = max(lonlatnm[,2])
    d.lon = (ma.lon-mi.lon)*dlon
    d.lat = (ma.lat-mi.lat)*dlat
    delta = spatial.buffer*max(d.lat,d.lon)
    me.lon = mean(c(mi.lon,ma.lon))
    me.lat = mean(c(mi.lat,ma.lat))
    mi.lon = mi.lon-delta/dlon
    ma.lon = ma.lon+delta/dlon
    mi.lat = mi.lat-delta/dlat
    ma.lat = ma.lat+delta/dlat
    if(jitter>0){
      lonlatnm[,"longitude"] = lonlatnm[,"longitude"]+jitter*max(d.lat,d.lon)*(runif(nrow(lonlatnm))-0.5)/dlon
      lonlatnm[,"latitude"] = lonlatnm[,"latitude"]+jitter*max(d.lat,d.lon)*(runif(nrow(lonlatnm))-0.5)/dlat
    }
    if(is.null(background.map)){
      plot(NULL,xlab="longitude",ylab="latitude",xlim=c(mi.lon,ma.lon),ylim=c(mi.lat,ma.lat),
           main = "Recording locations")
    } else {
      background_map_cropped = terra::crop(background.map,y=c(mi.lon,ma.lon,mi.lat,ma.lat))
      terra::plot(background_map_cropped,xlab="longitude",ylab="latitude",
           main = "Recording locations")
    }
    for(i in 1:nrow(lonlatnm)){
      points(lonlatnm[i,1],lonlatnm[i,2],pch=16,
             col = factor(rownames(lonlatnm),levels = segments)[i],
             cex=1+2*sqrt(lonlatnm[i,3]/max(lonlatnm[,3])))
      if(show.number.of.species) text(lonlatnm[i,1],lonlatnm[i,2],lonlatnm[i,4],col = number.of.species.color)
      if(show.segment.id) text(lonlatnm[i,1],lonlatnm[i,2],rownames(lonlatnm)[i],col = segment.id.color, cex=0.5)
    }
  }
}
