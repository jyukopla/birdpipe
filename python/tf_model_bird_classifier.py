"""
Bird Audio Classifier

Specialized classifier for bird species identification.
Handles downsampling from ultrasonic frequencies to 48kHz for BirdNET models.
"""

import numpy as np
import librosa
from scipy.signal import butter, lfilter, resample

from tf_model_base_classifier import BaseAudioClassifier
from bird_classifier_utils import pad, split_signal


class BirdClassifier(BaseAudioClassifier):
    """
    Bird species classifier using BirdNET feature extractor and regional models.
    
    Expects 48kHz audio and can downsample from higher sample rates with anti-aliasing.
    """
    
    def __init__(self, region_model_path, offset=0, dur=0, **kwargs):
        """
        Initialize bird classifier.
        
        Args:
            region_model_path: Path to region-specific classifier model
            offset: Audio offset for analysis (seconds)
            dur: Duration limit for analysis (seconds, 0 = full file)
        """
        model_paths = {
            'feature_extractor': 'BN-2.4_tflite/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite',
            'classifier': region_model_path
        }
        
        super().__init__(
            expected_sr=48000,  # BirdNET expects 48kHz
            model_paths=model_paths,
            **kwargs
        )
        
        self.offset = offset
        self.dur = dur
    
    def _init_models(self):
        """Initialize BirdNET feature extractor and regional classifier."""
        # Feature extractor (BirdNET)
        self.feature_interpreter = self._get_tflite_interpreter(self.model_paths['feature_extractor'])
        
        input_details = self.feature_interpreter.get_input_details()
        output_details = self.feature_interpreter.get_output_details()
        self.feature_input_index = input_details[0]["index"]
        self.feature_output_index = output_details[0]["index"] - 1
        
        # Regional classifier
        self.classifier_interpreter = self._get_tflite_interpreter(self.model_paths['classifier'])
        
        input_details2 = self.classifier_interpreter.get_input_details()
        output_details2 = self.classifier_interpreter.get_output_details()
        self.classifier_input_index = input_details2[0]["index"]
        self.classifier_output_index = output_details2[0]["index"]
    
    def preprocess_audio(self, audio_data, input_sr):
        """
        Preprocess audio for bird classification.
        
        Downsamples from ultrasonic frequencies to 48kHz with anti-aliasing filter.
        
        Args:
            audio_data: Raw audio data
            input_sr: Original sample rate
            
        Returns:
            Audio data resampled to 48kHz
        """
        if input_sr != self.expected_sr:
            print(f"Resampling audio from {input_sr}Hz to {self.expected_sr}Hz")
            
            # Log memory usage before resampling
            try:
                import psutil
                import os
                from recorder_config import SHOW_MEMORY_USAGE
                if SHOW_MEMORY_USAGE:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"  Memory before resampling: {memory_mb:.1f}MB")
            except:
                pass
            
            # Apply anti-aliasing lowpass filter before resampling
            audio_data = self._lowpass_filter(audio_data, cutoff=24000, samplerate=input_sr)
            
            # Resample to 48kHz
            num_samples = int(len(audio_data) * self.expected_sr / input_sr)
            audio_data = resample(audio_data, num_samples)
            
            # Log memory usage after resampling
            try:
                if SHOW_MEMORY_USAGE:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"  Memory after resampling: {memory_mb:.1f}MB")
            except:
                pass
        
        return audio_data
    
    def _lowpass_filter(self, data, cutoff, samplerate, order=5):
        """
        Apply anti-aliasing lowpass Butterworth filter.
        
        Args:
            data: Audio data
            cutoff: Cutoff frequency (Hz)
            samplerate: Original sample rate
            order: Filter order
            
        Returns:
            Filtered audio data
        """
        nyquist = 0.5 * samplerate
        # Ensure cutoff is below Nyquist frequency
        actual_cutoff = min(cutoff, nyquist * 0.99)
        normal_cutoff = actual_cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_data = lfilter(b, a, data)
        return filtered_data
    
    def _load_audio_file(self, file_path, input_sr):
        """Load audio file with optional offset and duration limits."""
        if self.dur > 0:
            audio_data, _ = librosa.load(
                file_path, sr=input_sr, mono=True, res_type='kaiser_fast',
                offset=self.offset, duration=self.dur
            )
        else:
            audio_data, _ = librosa.load(
                file_path, sr=input_sr, mono=True, res_type='kaiser_fast'
            )
        return audio_data
    
    def _run_inference(self, processed_audio, overlap=1.0, max_pred=True):
        """
        Run bird species classification on preprocessed audio.
        
        Args:
            processed_audio: Audio at 48kHz
            overlap: Overlap between audio chunks (seconds)
            max_pred: If True, return max prediction per species. If False, return all predictions.
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        print("Analyzing audio...")
        chunks = split_signal(processed_audio, self.expected_sr, self.clip_dur, overlap)
        
        samples = []
        for chunk in chunks:
            samples.append(chunk)
        
        X = np.array(samples, dtype='float32')
        
        # Extract embeddings using BirdNET feature extractor
        embeddings = self._extract_embeddings(X)
        
        # Run regional classifier on embeddings
        predictions = self._classify_embeddings(embeddings)
        
        
        if max_pred:
            # Return maximum prediction for each species
            try:
                if len(predictions) > 0:
                    pred = list(map(max, zip(*predictions)))
                    timestamps = np.argmax(predictions, axis=0) * (self.clip_dur - overlap)
                else:
                    pred = []
                    timestamps = np.array([])
            except Exception as e:
                print(f"Warning: Error processing predictions - {e}")
                pred = []
                timestamps = np.array([])
        else:
            # Return all predictions
            pred = predictions
            timestamps = np.array(range(len(predictions))) * (self.clip_dur - overlap)
        
        return pred, timestamps
    
    def _extract_embeddings(self, audio_samples):
        """Extract feature embeddings using BirdNET."""
        # Resize input tensor for batch processing
        self.feature_interpreter.resize_tensor_input(
            self.feature_input_index, [len(audio_samples), *audio_samples[0].shape]
        )
        self.feature_interpreter.allocate_tensors()
        
        # Run feature extraction
        self.feature_interpreter.set_tensor(self.feature_input_index, audio_samples)
        self.feature_interpreter.invoke()
        
        embeddings = self.feature_interpreter.get_tensor(self.feature_output_index)
        return embeddings
    
    def _classify_embeddings(self, embeddings):
        """Run regional classifier on feature embeddings."""
        # Resize input tensor for embeddings
        self.classifier_interpreter.resize_tensor_input(
            self.classifier_input_index, [len(embeddings), *embeddings[0].shape]
        )
        self.classifier_interpreter.allocate_tensors()
        
        # Run classification
        self.classifier_interpreter.set_tensor(
            self.classifier_input_index, np.array(embeddings, dtype="float32")
        )
        self.classifier_interpreter.invoke()
        
        predictions = self.classifier_interpreter.get_tensor(self.classifier_output_index)
        return predictions
    
    def classify(self, audio_data_or_path, input_sr, overlap=1.0, max_pred=True):
        """
        Classify bird species in audio.
        
        Args:
            audio_data_or_path: Audio data or file path
            input_sr: Sample rate of input audio
            overlap: Overlap between chunks (seconds)
            max_pred: Return max predictions per species
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        # Load audio if path provided
        if isinstance(audio_data_or_path, str):
            audio_data = self._load_audio_file(audio_data_or_path, input_sr)
        elif hasattr(audio_data_or_path, 'read'):
            # Handle BytesIO buffer
            import soundfile as sf
            audio_data_or_path.seek(0)
            audio_data, _ = sf.read(audio_data_or_path)
            # Ensure it's 1D array (mono)
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]  # Take first channel if stereo
        else:
            # Already numpy array
            audio_data = audio_data_or_path
        
        # Ensure audio_data is a 1D numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        
        # Preprocess to 48kHz
        processed_audio = self.preprocess_audio(audio_data, input_sr)
        
        # Run inference
        return self._run_inference(processed_audio, overlap, max_pred)