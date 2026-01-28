#' extract_community_matrix
#'
#' Extracts a community matrix from BirdPipe data
#' @param BirdPipe.data A BirdPipe data object
#' @param threshold Classification threshold to be used to filter the data (influences number of detected species)
#' @param name.to.use Which names to use for the species
#' @return A community matrix, i.e. samples (=segments) times species matrix
#' @examples
#' BirdPipe.data = load_data(BirdPipe.folder = "BirdPipe_demo_data");
#' extract_community_matrix(BirdPipe.data = BirdPipe.data);
#' @export
extract_community_matrix = function(BirdPipe.data = BirdPipe.data,
                            threshold = 0,
                            name.to.use = "species_name"){
  detections = droplevels(BirdPipe.data$detections[BirdPipe.data$detections$probability>=threshold,])
  detections$name = detections[,name.to.use]
  spp = levels(detections$name)
  segments = levels(BirdPipe.data$recordings$segment)
  Y = matrix(nrow=length(segments), ncol=length(spp))
  rownames(Y) = segments
  colnames(Y) = spp
  for(segment in segments){
    sel = detections$audio_filename %in% rownames(BirdPipe.data$recordings)[BirdPipe.data$recordings$segment==segment]
    ta = table(detections$name[sel])
    Y[segment,names(ta)] = ta
  }
  return(Y)
}
