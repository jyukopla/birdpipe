"""
Base Audio Classifier Class

Provides common interface for different types of audio classifiers.
Each classifier can handle its own sample rate requirements and preprocessing.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAudioClassifier(ABC):
    """
    Base class for audio classifiers.
    
    Each subclass should specify its expected sample rate and implement
    the preprocessing and inference methods.
    """
    
    def __init__(self, expected_sr, model_paths, clip_dur=3.0, tflite_threads=1, **kwargs):
        """
        Initialize base classifier.
        
        Args:
            expected_sr: Target sample rate for this classifier
            model_paths: Dictionary of model file paths
            clip_dur: Duration of audio clips for analysis (seconds)
            tflite_threads: Number of threads for TFLite inference
        """
        self.expected_sr = expected_sr
        self.model_paths = model_paths
        self.clip_dur = clip_dur
        self.tflite_threads = tflite_threads
        
        # Initialize TFLite interpreters
        self._init_models()
    
    @abstractmethod
    def _init_models(self):
        """Initialize TFLite model interpreters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def preprocess_audio(self, audio_data, input_sr):
        """
        Preprocess audio to match classifier requirements.
        
        Args:
            audio_data: Raw audio data (numpy array)
            input_sr: Original sample rate of the audio
            
        Returns:
            Preprocessed audio data at expected_sr
        """
        pass
    
    @abstractmethod
    def _run_inference(self, processed_audio):
        """
        Run the actual ML inference on preprocessed audio.
        
        Args:
            processed_audio: Audio data preprocessed to expected format
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        pass
    
    def classify(self, audio_data_or_path, input_sr, **kwargs):
        """
        Main classification pipeline.
        
        Args:
            audio_data_or_path: Either numpy array of audio data or file path
            input_sr: Sample rate of the input audio
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        # Load audio if path provided
        if isinstance(audio_data_or_path, str):
            audio_data = self._load_audio_file(audio_data_or_path, input_sr)
        else:
            audio_data = audio_data_or_path
        
        # Preprocess to expected sample rate and format
        processed_audio = self.preprocess_audio(audio_data, input_sr)
        
        # Run inference
        return self._run_inference(processed_audio)
    
    def _load_audio_file(self, file_path, expected_sr):
        """Load audio file. Can be overridden by subclasses."""
        import librosa
        audio_data, _ = librosa.load(file_path, sr=expected_sr, mono=True, res_type='kaiser_fast')
        return audio_data
    
    def _get_tflite_interpreter(self, model_path):
        """Helper to create TFLite interpreter with fallback."""
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=self.tflite_threads)
        interpreter.allocate_tensors()
        return interpreter