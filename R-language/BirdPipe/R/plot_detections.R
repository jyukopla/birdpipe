#' plot_detections
#'
#' Plots BirdPipe detections as timeline
#' @param BirdPipe.data A BirdPipe data object
#' @param name.to.use Which names to use for the species
#' @param species.order In which order to show the species ("alphabetical" or "number_of_detections")
#' @param threshold Classification threshold to be used to filter the data (influences number of detected species)
#' @param jitter the amount of random jitter added to shows overlapping datapoints
#' @return NULL
#' @examples
#' BirdPipe.data = load_data(BirdPipe.folder = "BirdPipe_demo_data");
#' plot_detections(BirdPipe.data = BirdPipe.data)
#' @export
plot_detections = function(BirdPipe.data,
                           name.to.use = "species_name",
                           species.order = "alphabetical",
                           jitter = 0,
                           threshold = 0){

  detections = droplevels(BirdPipe.data$detections[BirdPipe.data$detections$probability>=threshold,])
  detections$name = detections[,name.to.use]
  plot.order = sort(names(table(detections$name)))
  if(species.order=="number_of_detections"){
    plot.order = names(sort(table(detections$name),decreasing = T))
  }
  detections = detections[order(match(detections$name,plot.order)),]
  id = match(detections$audio_filename,rownames(BirdPipe.data$recordings))
  timestamp = BirdPipe.data$recordings$timestamp[id]
  xx = match(detections$name,plot.order)
  if(jitter>0) xx = xx+jitter*(runif(length(xx))-0.5)
  plot(xx,timestamp,
       xlab = "", ylab = "", xaxt = "n",
       pch = 16, col = detections$name,
       main = paste(BirdPipe.data$devices,collapse = ", "),
       cex.main = 0.7,
       cex.axis = 0.5)
  abline(v=1:length(plot.order),
         col=match(plot.order,levels(detections$name)))
  axis(1,at=1:length(plot.order),
       labels= plot.order,
       las = 2, cex.axis = 0.5)
  print("Numbers of detections:")
  print(table(detections$name)[plot.order])
}
