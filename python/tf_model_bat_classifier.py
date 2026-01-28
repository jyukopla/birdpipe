"""
Bat Audio Classifier

Specialized classifier for European bat species identification using 384kHz ultrasonic recordings.
Implements mel-spectrogram analysis optimized for bat echolocation calls in the 9-150kHz range.

Key Features:
- 384kHz ultrasonic audio processing
- Mel-spectrogram with 512-frame segments (0.5s overlap)
- Batched inference for performance optimization
- European bat species detection (22 classes including background)
- Detailed segment-level detection reporting in test mode

This classifier processes ultrasonic audio to detect European bat species using a pre-trained 
TensorFlow Lite model (bsg_europe_bat_v1.tflite). It segments audio into overlapping windows and analyzes 
frequency content optimized for bat echolocation calls.

Compatible with the multi-model recorder system for simultaneous bird and bat detection.
"""

import numpy as np
import librosa
import tempfile
import os
import io
import soundfile as sf
import time

from tf_model_base_classifier import BaseAudioClassifier
from recorder_config import TEST_MODE, SHOW_SEGMENT_DETECTIONS, MODEL_INFORMATION


class BatClassifier(BaseAudioClassifier):
    """
    Bat species classifier for ultrasonic recordings.
    
    Processes 384kHz audio using mel-spectrograms optimized for bat echolocation calls.
    Detects European bat species including:
    - Western barbastelle (Barbastella barbastellus)
    - Northern bat (Eptesicus nilssonii)
    - Common pipistrelle (Pipistrellus pipistrellus)
    - And 19 other European species
    
    Audio processing:
    - 384kHz sample rate (full ultrasonic resolution)
    - Mel-spectrogram: 9kHz-150kHz frequency range
    - 1-second analysis windows with 50% overlap
    - 512 time frames × 128 frequency bins
    """
    
    def __init__(self, bat_model_path, class_path=None, **kwargs):
        """
        Initialize bat classifier.
        
        Args:
            bat_model_path: Path to bat species classifier model (.tflite)
            class_path: Path to species class mapping CSV file
        """
        model_paths = {
            'classifier': bat_model_path
        }
        
        # Store class path for species mapping
        self.class_path = class_path
        
        super().__init__(
            expected_sr=384000,  # Bat classifier expects 384kHz
            model_paths=model_paths,
            clip_dur=1.0,  # 1 second segments for bat analysis
            **kwargs
        )
        
        # Bat-specific parameters (from classifier_bat_384.py)
        self.ntime = 512  # Time frames in spectrogram
        self.nhop = 250   # Hop size between segments (0.5s overlap)
        self.nfreq = 128  # Frequency bins in mel-spectrogram
        self.fmin = 9000  # Minimum frequency (Hz)
        self.fmax = 150000  # Maximum frequency (Hz)
    
    def _init_models(self):
        """Initialize bat species classifier."""
        self.classifier_interpreter = self._get_tflite_interpreter(self.model_paths['classifier'])
        
        # Get model input/output details
        self.input_details = self.classifier_interpreter.get_input_details()
        self.output_details = self.classifier_interpreter.get_output_details()
    
    def preprocess_audio(self, audio_data, input_sr):
        """
        Preprocess audio for bat analysis.
        
        Args:
            audio_data: Raw audio data (numpy array)
            input_sr: Original sample rate of the audio
            
        Returns:
            Audio data resampled to 384kHz
        """
        if input_sr != self.expected_sr:
            # Log memory usage before resampling
            try:
                import psutil
                import os
                from recorder_config import SHOW_MEMORY_USAGE, MODEL_INFORMATION
                if SHOW_MEMORY_USAGE:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"  Memory before bat resampling: {memory_mb:.1f}MB")
            except:
                pass
            
            # Time the resampling operation
            resample_start = time.time()
            if MODEL_INFORMATION:
                print(f"    Starting resampling: {len(audio_data)} samples @ {input_sr}Hz → {self.expected_sr}Hz")
            
            # Resample to 384kHz using librosa (kaiser_fast for speed)
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=input_sr, 
                target_sr=self.expected_sr, 
                res_type='kaiser_fast'
            )
            
            resample_time = time.time() - resample_start
            if MODEL_INFORMATION:
                print(f"    Resampling completed in {resample_time:.2f}s → {len(audio_data)} samples")
            
            # Log memory usage after resampling
            try:
                if SHOW_MEMORY_USAGE:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"  Memory after bat resampling: {memory_mb:.1f}MB")
            except:
                pass
        
        return audio_data
    
    def _wav2spectrograms(self, audio_data):
        """
        Convert audio to mel-spectrograms for bat analysis.
        Based on wav2spectrograms from classifier_bat_384.py.
        
        Args:
            audio_data: Audio signal at 384kHz
            
        Returns:
            Numpy array of spectrograms (n_segments, ntime, nfreq)
        """
        spectrogram_start = time.time()
        
        # Log audio preprocessing details
        try:
            from recorder_config import MODEL_INFORMATION
            if MODEL_INFORMATION:
                print(f"    BAT SPECTROGRAM ANALYSIS:")
                print(f"      • Audio length: {len(audio_data)} samples ({len(audio_data)/self.expected_sr:.2f}s)")
                print(f"      • Frequency range: {self.fmin}-{self.fmax} Hz")
                print(f"      • Mel-spectrogram params: n_fft=1024, hop_length=768, n_mels={self.nfreq}")
        except:
            pass
        
        # Time the mel-spectrogram computation
        mel_start = time.time()
        
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.expected_sr, 
            n_fft=1024, 
            hop_length=768, 
            n_mels=self.nfreq, 
            fmin=self.fmin, 
            fmax=self.fmax
        ).T
        
        mel_time = time.time() - mel_start
        
        # Log spectrogram details
        try:
            if MODEL_INFORMATION:
                print(f"      Mel-spectrogram computed in {mel_time:.2f}s")
                print(f"      • Raw spectrogram shape: {S.shape} (time × freq)")
                print(f"      • Spectrogram energy range: {np.min(S):.2e} to {np.max(S):.2e}")
        except:
            pass
        
        # Split into segments
        n = int(np.max((np.ceil((len(S) - self.ntime) / self.nhop), 1)))
        data = np.ndarray((n, self.ntime, self.nfreq), dtype='float32')
        
        # Log segmentation details
        try:
            if MODEL_INFORMATION:
                print(f"      • Segmentation: {n} segments of {self.ntime} time frames")
                print(f"      • Hop size: {self.nhop} frames ({self.nhop * 768 / self.expected_sr:.3f}s overlap)")
                if n > 100:
                    print(f"      WARNING: Large number of segments detected - this will be slow!")
        except:
            pass
        
        segment_start = time.time()
        
        if len(S) < self.ntime:
            # Recording shorter than desired segment length, do zero padding
            try:
                if MODEL_INFORMATION:
                    print(f"      • Short audio: padding {len(S)} → {self.ntime} frames")
            except:
                pass
            X = np.zeros((self.ntime, self.nfreq), dtype='float32')
            X[:len(S)] = S
            data[0] = self._normalize_spectrogram(np.log10(X + 1e-6))
        else:
            # Chop into segments every nhop frames
            for i in range(n):
                start_i = i * self.nhop
                if start_i + self.ntime <= len(S):
                    data[i] = self._normalize_spectrogram(np.log10(S[start_i:start_i + self.ntime] + 1e-6))
                else:
                    # Last segment too short, include data from left
                    start_i = len(S) - self.ntime
                    data[i] = self._normalize_spectrogram(np.log10(S[start_i:start_i + self.ntime] + 1e-6))
        
        segment_time = time.time() - segment_start
        total_time = time.time() - spectrogram_start
        
        # Log final spectrogram array details
        try:
            if MODEL_INFORMATION:
                print(f"      Segmentation completed in {segment_time:.2f}s")
                print(f"      Total spectrogram processing: {total_time:.2f}s")
                print(f"      • Final spectrograms: {data.shape} (segments × time × freq)")
                print(f"      • Normalized range: {np.min(data):.2f} to {np.max(data):.2f}")
        except:
            pass
        
        return data
    
    def _normalize_spectrogram(self, data):
        """
        Normalize spectrogram data.
        Based on normalize function from classifier_bat_384.py.
        """
        x = (data - np.mean(data)) / np.std(data)
        return np.clip(x - np.median(x, axis=0), 0.0, 6.0)
    
    def _softmax(self, x):
        """Convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _run_inference(self, processed_audio):
        """
        Run bat species classification on preprocessed audio.
        Matches the reference implementation from bat_analyzer.py.
        
        Args:
            processed_audio: Audio data at 384kHz
            
        Returns:
            Tuple of (predictions, timestamps) where predictions is array of max confidence per class
        """
        inference_start = time.time()
        
        # Log inference start
        try:
            from recorder_config import MODEL_INFORMATION
            if MODEL_INFORMATION:
                print(f"    BAT INFERENCE:")
                print(f"      • Processing {len(processed_audio)} samples at {self.expected_sr}Hz")
        except:
            pass
        
        # Generate spectrograms
        spectrograms = self._wav2spectrograms(processed_audio)
        
        if spectrograms is None or len(spectrograms) == 0:
            try:
                if MODEL_INFORMATION:
                    print(f"      • No spectrograms generated - returning empty results")
            except:
                pass
            return [], []
        
        # Get expected batch size from model and number of classes
        expected_batch_size = self.input_details[0]['shape'][0]
        num_classes = self.output_details[0]['shape'][-1]  # Get number of output classes
        
        # Log model details
        try:
            if MODEL_INFORMATION:
                print(f"      • Model expects batch size: {expected_batch_size}")
                print(f"      • Model outputs {num_classes} classes")
                print(f"      • Input shape: {self.input_details[0]['shape']}")
                print(f"      • Output shape: {self.output_details[0]['shape']}")
        except:
            pass
        
        all_predictions = []
        all_timestamps = []
        segment_details = []
        
        # Time the inference loop
        segment_inference_start = time.time()
        
        # Calculate all timestamps first
        timestamps = [i * self.nhop * 768 / self.expected_sr for i in range(len(spectrograms))]
        
        # Process segments in batches for efficiency
        batch_size = expected_batch_size
        
        try:
            if MODEL_INFORMATION:
                print(f"      • Processing {len(spectrograms)} segments in batches of {batch_size}")
        except:
            pass
        
        for batch_start in range(0, len(spectrograms), batch_size):
            batch_end = min(batch_start + batch_size, len(spectrograms))
            batch_spectrograms = spectrograms[batch_start:batch_end]
            batch_timestamps = timestamps[batch_start:batch_end]
            
            batch_inference_start = time.time()
            
            # Prepare batch input data
            input_data = np.array(batch_spectrograms).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=1)  # Add channel dimension
            
            # Pad batch if necessary
            if input_data.shape[0] < expected_batch_size:
                padding_shape = list(input_data.shape)
                padding_shape[0] = expected_batch_size - input_data.shape[0]
                padding = np.zeros(padding_shape, dtype=np.float32)
                input_data = np.concatenate([input_data, padding], axis=0)
            
            # Run batch inference
            self.classifier_interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.classifier_interpreter.invoke()
            prediction_logits = self.classifier_interpreter.get_tensor(self.output_details[0]['index'])
            
            batch_inference_time = time.time() - batch_inference_start
            
            # Process results for each segment in the batch
            for i, (timestamp, logits) in enumerate(zip(batch_timestamps, prediction_logits[:len(batch_spectrograms)])):
                probs = self._softmax(logits)
                
                # Find best prediction for this segment
                max_confidence = np.max(probs)
                predicted_class = np.argmax(probs)
                
                # Store segment details for logging
                segment_details.append({
                    'timestamp': timestamp,
                    'max_confidence': max_confidence,
                    'predicted_class': predicted_class,
                    'top3_classes': np.argsort(probs)[-3:][::-1],
                    'top3_probs': np.sort(probs)[-3:][::-1],
                    'batch_inference_time': batch_inference_time
                })
                
                # Store predictions and timestamps
                all_predictions.append(probs)
                all_timestamps.append(timestamp)
            
            try:
                if MODEL_INFORMATION:
                    segments_in_batch = len(batch_spectrograms)
                    avg_per_segment = batch_inference_time / segments_in_batch
                    print(f"        Batch {batch_start//batch_size + 1}: "
                          f"{segments_in_batch} segments in {batch_inference_time:.2f}s "
                          f"({avg_per_segment:.3f}s/segment)")
            except:
                pass
        
        total_segment_time = time.time() - segment_inference_start
        
        # Log segment analysis results
        try:
            if MODEL_INFORMATION:
                print(f"      Processed {len(spectrograms)} segments in {total_segment_time:.2f}s")
                print(f"      Average per segment: {total_segment_time/len(spectrograms):.3f}s")
                print(f"      • Analyzed {len(spectrograms)} segments:")
                
                # Show first few segments and any high-confidence detections
                shown_count = 0
                for i, details in enumerate(segment_details):
                    if details['max_confidence'] > 0.1 or shown_count < 3:  # Show first 3 or high confidence
                        print(f"        Segment {i+1}: t={details['timestamp']:.2f}s, "
                              f"class {details['predicted_class']} conf={details['max_confidence']:.3f}")
                        if details['max_confidence'] > 0.1:
                            print(f"          Top 3: {details['top3_classes']} "
                                  f"({details['top3_probs'][0]:.3f}, {details['top3_probs'][1]:.3f}, {details['top3_probs'][2]:.3f})")
                        shown_count += 1
                    if shown_count >= 10:  # Limit output for very long files
                        remaining = len(segment_details) - i - 1
                        if remaining > 0:
                            print(f"        ... and {remaining} more segments")
                        break
                        
                # In test mode, show detailed segment detections above threshold
                if TEST_MODE and SHOW_SEGMENT_DETECTIONS:
                    print(f"      SEGMENT DETECTIONS (threshold 0.5):")
                    detection_count = 0
                    
                    # Load species mapping for display - use existing loaded mapping if available
                    species_mapping = {}
                    if hasattr(self, '_species_mapping') and self._species_mapping:
                        species_mapping = self._species_mapping
                    else:
                        # Load mapping as fallback
                        try:
                            import pandas as pd
                            species_df = pd.read_csv(self.class_path)
                            for _, row in species_df.iterrows():
                                species_mapping[int(row['class'])] = {
                                    'common_name': row['common_name'],
                                    'scientific_name': row['scientific_name']
                                }
                        except Exception as e:
                            # Fallback if CSV loading fails - use known classes
                            species_mapping = {
                                0: {'common_name': 'Western barbastelle', 'scientific_name': 'Barbastella barbastellus'},
                                21: {'common_name': 'Background', 'scientific_name': 'Background'}
                            }
                    
                    for i, details in enumerate(segment_details):
                        if details['max_confidence'] > 0.5:  # Detection threshold
                            start_time = details['timestamp']
                            # Each segment is ~0.5s, calculate end time
                            end_time = start_time + 0.5
                            predicted_class = details['predicted_class']
                            species_info = species_mapping.get(
                                predicted_class, 
                                {'common_name': f'Class {predicted_class}', 'scientific_name': 'Unknown'}
                            )
                            
                            print(f"        Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s")
                            print(f"          {species_info['common_name']} ({species_info['scientific_name']})")
                            print(f"          Class {predicted_class}: {details['max_confidence']:.3f} confidence")
                            print(f"          Top 3: {details['top3_classes']} "
                                  f"({details['top3_probs'][0]:.3f}, {details['top3_probs'][1]:.3f}, {details['top3_probs'][2]:.3f})")
                            detection_count += 1
                    if detection_count == 0:
                        print(f"        No segments above detection threshold (0.5)")
                    else:
                        print(f"        Total: {detection_count} segments with detections")
        except:
            pass
        
        # Convert to max prediction per class (like bird classifier does)
        if len(all_predictions) > 0:
            # Get maximum confidence for each class across all time segments
            max_predictions = np.max(all_predictions, axis=0)
            
            # Get the timestamp where each class achieved its maximum confidence
            max_timestamps = []
            for class_idx in range(len(max_predictions)):
                # Find which segment had the maximum confidence for this class
                segment_confidences = [pred[class_idx] for pred in all_predictions]
                max_segment_idx = np.argmax(segment_confidences)
                max_timestamps.append(all_timestamps[max_segment_idx])
            
            total_time = time.time() - inference_start
            
            # Log final class predictions
            try:
                if MODEL_INFORMATION:
                    print(f"      Total inference time: {total_time:.2f}s")
                    print(f"      • Final class maxima:")
                    top_classes = np.argsort(max_predictions)[-5:][::-1]  # Top 5 classes
                    for cls_idx in top_classes:
                        if max_predictions[cls_idx] > 0.01:  # Show classes with >1% confidence
                            print(f"        Class {cls_idx}: {max_predictions[cls_idx]:.4f} "
                                  f"at {max_timestamps[cls_idx]:.2f}s")
            except:
                pass
            
            return list(max_predictions), max_timestamps
        else:
            return [], []
    
    def classify(self, audio_data_or_path, input_sr, **kwargs):
        """
        Main classification pipeline for bat audio.
        
        Args:
            audio_data_or_path: Either numpy array of audio data or file path
            input_sr: Sample rate of the input audio
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        # Log classification start
        try:
            from recorder_config import MODEL_INFORMATION
            if MODEL_INFORMATION:
                print(f"  BAT CLASSIFIER PIPELINE:")
                if isinstance(audio_data_or_path, str):
                    print(f"    • Loading audio from file: {audio_data_or_path}")
                else:
                    print(f"    • Processing audio array: {len(audio_data_or_path)} samples")
                print(f"    • Input sample rate: {input_sr}Hz → Target: {self.expected_sr}Hz")
        except:
            pass
        
        # Load audio if path provided
        if isinstance(audio_data_or_path, str):
            audio_data = self._load_audio_file(audio_data_or_path, self.expected_sr)
        else:
            audio_data = audio_data_or_path
        
        # Log audio characteristics
        try:
            if MODEL_INFORMATION:
                print(f"    • Audio loaded: {len(audio_data)} samples ({len(audio_data)/input_sr:.2f}s)")
                print(f"    • Audio range: {np.min(audio_data):.4f} to {np.max(audio_data):.4f}")
                print(f"    • Audio RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")
        except:
            pass
        
        # Preprocess to expected sample rate
        processed_audio = self.preprocess_audio(audio_data, input_sr)
        
        # Log post-processing characteristics
        try:
            if MODEL_INFORMATION:
                if input_sr != self.expected_sr:
                    print(f"    • Resampled to: {len(processed_audio)} samples "
                          f"({len(processed_audio)/self.expected_sr:.2f}s)")
                    print(f"    • Resampled RMS: {np.sqrt(np.mean(processed_audio**2)):.4f}")
        except:
            pass
        
        # Run inference
        predictions, timestamps = self._run_inference(processed_audio)
        
        # Log final results
        try:
            if MODEL_INFORMATION:
                print(f"    • Final results: {len(predictions)} class predictions, {len(timestamps)} timestamps")
                if len(predictions) > 0:
                    max_conf = np.max(predictions)
                    best_class = np.argmax(predictions)
                    print(f"    • Best detection: Class {best_class} with confidence {max_conf:.4f}")
        except:
            pass
        
        return predictions, timestamps