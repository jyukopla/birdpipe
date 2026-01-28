#' highest_classification_per_segment
#'
#' Finds the vocalization with highest classification probability for each species and segment
#' @param BirdPipe.data A BirdPipe data object
#' @param threshold Classification threshold to be used to filter the data
#' @param manual.validation.file Excel file to be written for entering manual validation results
#' @return A named list where the locations of highest classifications are given for each species
#' @examples
#' BirdPipe.data = load_data(BirdPipe.folder = "BirdPipe_demo_data");
#' highest_classification_per_segment(BirdPipe.data = BirdPipe.data);
#' @importFrom xlsx write.xlsx
#' @export
highest_classification_per_segment = function(BirdPipe.data,
                                              threshold = 0.5,
                                              manual.validation.file = NULL){
  detections = droplevels(BirdPipe.data$detections[BirdPipe.data$detections$probability>=threshold,])
  spp = levels(detections$species_name)
  ns = length(spp)
  highest.classifications = list()
  for(sp in spp){
    sel = which(detections$species_name==sp)
    da = detections[sel,c("audio_filename","probability","offset_s")]
    sel = match(da$audio_filename,rownames(BirdPipe.data$recordings))
    da = cbind(da,as.character(BirdPipe.data$recordings[sel,"segment"]))
    colnames(da)[4] = "segment"
    segments = sort(unique(da$segment))
    ny = length(segments)
    res.id = rep(NA,ny)
    res.pr = rep(NA,ny)
    res.os = rep(NA,ny)
    for(i in 1:ny){
      seg = segments[i]
      sel = which(da$segment==seg)
      sel = sel[which.max(da$probability[sel])]
      res.id[i] = da[sel,"audio_filename"]
      res.pr[i] = da[sel,"probability"]
      res.os[i] = da[sel,"offset_s"]
    }
    res = data.frame(segment = segments, audio_filename = res.id, probability = res.pr, offset_s = res.os)
    highest.classifications[[sp]] = res
  }

  if(!is.null(manual.validation.file)){
    tmp = highest.classifications[[1]]
    tmp$manual_validation = "NA"
    xlsx::write.xlsx(tmp,
               file=manual.validation.file,
               sheetName=names(highest.classifications)[1],
               append=FALSE,
               row.names=FALSE)
    for(j in 2:length(highest.classifications)){
      tmp = highest.classifications[[j]]
      tmp$manual_validation = "NA"
      write.xlsx(tmp,
                 file=manual.validation.file,
                 sheetName=names(highest.classifications)[j],
                 append=TRUE, row.names=FALSE)
    }
  }
  return(highest.classifications)
}

