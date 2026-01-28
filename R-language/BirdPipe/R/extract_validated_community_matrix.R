#' extract_validated_community_matrix
#'
#' Extracts a validated community matrix from BirdPipe data and manual validation file
#' @param BirdPipe.data A BirdPipe data object
#' @param manual.validation.file Excel file of manual validation results
#' @param validated Which strings are considered as validated detections
#' @return A community matrix, i.e. samples (=segments) times species matrix
#' @examples
#' BirdPipe.data = load_data(BirdPipe.folder = "BirdPipe_demo_data", tgre);
#' extract_validated_community_matrix(BirdPipe.data = BirdPipe.data, manual.validation.file = "detections_validated.xlsx", validated = c("Y","YM"));
#' @importFrom readxl excel_sheets
#' @importFrom xlsx read.xlsx
#' @export
extract_validated_community_matrix = function(BirdPipe.data = BirdPipe.data,
                              manual.validation.file = NULL,
                              validated = c("Y")){
  spp = readxl::excel_sheets(manual.validation.file)
  segments = levels(BirdPipe.data$recordings$segment)
  Y = matrix(0, nrow=length(segments), ncol=length(spp))
  colnames(Y) = spp
  rownames(Y) = segments
  for(sp in spp){
    res = xlsx::read.xlsx(file = manual.validation.file, sheetIndex = sp)
    sel = which(res$manual_validation %in% validated)
    Y[res$segment[sel],sp] = 1
  }
  return(Y)
}
