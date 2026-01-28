#' play_sound
#'
#' Plays part of audio file (.flac or .wav) e.g. for manual validation purpose
#' @param BirdPipe.folder The BirdPipe data folder
#' @param audio_folder The name of the audio folder within the  BirdPipe data folder
#' @param audio_filename The name of the audio file
#' @param offset The beginning of the 3 second segment from which the vocalization is classified
#' @param classification.interval Length of the classification interval in seconds
#' @param buffer Number of seconds added before and after the classification interval
#' @return NULL
#' @examples
#' play_sound(BirdPipe.folder = "BirdPipe_demo_data", audio_filename = "field-4_2025-12-22T10_05_03.011300Z.wav", offset = 28);
#' #This function call plays a vocalization of Eurasian Bullfinch from the R-package demonstration data.
#' #Note that Mac users may obtain a "Permission denied” error. If this happens, you may need to set the folder where the audio player is located by calling e.g. tuneR::setWavPlayer("/usr/bin/afplay").
#' @importFrom sonicscrewdriver readAudio
#' @importFrom tuneR extractWave
#' @importFrom tuneR play
#' @export
play_sound = function(BirdPipe.folder,
                      audio_folder = NULL,
                      audio_filename,
                      offset,
                      classification.interval = 3,
                      buffer = 1
                      ){
  if(is.null(audio_folder)) audio_folder = "audio"
  audiofiles = list.files(paste0(BirdPipe.folder,"/",audio_folder))
  if(audio_filename%in%audiofiles){
    a = sonicscrewdriver::readAudio(paste0(BirdPipe.folder,"/",audio_folder,"/",audio_filename))
    duration = length(a)/a@samp.rate
    a = tuneR::extractWave(a,from = max(0,offset-buffer),
                           to = min(duration,offset+buffer+classification.interval),
                           xunit = "time")
    tuneR::play(a)
  } else {
    print("audiofile not found")
  }
}
