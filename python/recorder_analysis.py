"""
Multi-model audio analysis module for BirdPipe recorder.

Handles coordination of multiple AI models (birds, bats, etc.) on the same audio data.
"""

import time
import pandas as pd
import recorder_config as config
from recorder_config import (
    MODELS_TO_RUN,
    MODEL_INFORMATION,
    SHOW_MODEL_TIMING,
)


def log_model_info(message, show_memory=False):
    """Log model processing information with optional memory usage."""
    if MODEL_INFORMATION:
        import psutil
        import os
        from recorder_config import SHOW_MEMORY_USAGE
        
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        if show_memory and SHOW_MEMORY_USAGE:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"[{timestamp}] {message} (Memory: {memory_mb:.1f}MB)")
        else:
            print(f"[{timestamp}] {message}")


def run_single_model(audio_data, recorded_samplerate, model_config, latitude=None, longitude=None, day_of_year=None):
    """
    Run a single model on the audio data.
    
    Args:
        audio_data: Audio buffer or file path
        recorded_samplerate: Sample rate of the audio
        model_config: Dictionary with model configuration
        latitude: GPS latitude for migration adjustment (optional)
        longitude: GPS longitude for migration adjustment (optional)
        day_of_year: Day of year for migration adjustment (optional)
        
    Returns:
        Dictionary with model results
    """
    model_region = model_config["region"]
    model_analysis_type = model_config["analysis_type"]
    model_name = model_config["name"]
    
    # Get the model file path from the config
    model_file = model_config["model_path"]
    
    print(f"Running {model_name} ({model_config['description']})...")
    log_model_info(f"STARTING {model_name} classifier", show_memory=True)
    
    # Check sample rate compatibility for bat models
    if model_analysis_type == "bats" and recorded_samplerate < 192000:
        print(f"  {model_name}: Skipping bat analysis - sample rate {recorded_samplerate}Hz too low for ultrasonic detection (requires ≥192kHz)")
        log_model_info(f"Skipped {model_name} - insufficient sample rate")
        # Return proper format with empty detections
        return {
            "model_name": model_name,
            "model_region": model_region,
            "model_analysis_type": model_analysis_type,
            "model_description": model_config["description"],
            "model_file": model_file,
            "model_version": model_config.get("version", "unknown"),
            "threshold": model_config.get("threshold", 0.3),
            "input_sample_rate_hz": recorded_samplerate,
            "processed_sample_rate_hz": None,  # No processing occurred
            "analysis_duration_seconds": 0.0,  # No time spent on analysis
            "detections": [],
            "skipped_reason": f"Sample rate {recorded_samplerate}Hz insufficient for ultrasonic analysis"
        }
    
    # Initialize classifier based on analysis type
    if model_analysis_type == "birds":
        log_model_info(f"Loading BirdNet feature extractor and {model_region} classifier")
        from tf_model_bird_classifier import BirdClassifier
        classifier = BirdClassifier(model_file)
        log_model_info(f"Bird classifier ready - target sample rate: {classifier.expected_sr}Hz")
    elif model_analysis_type == "bats":
        log_model_info(f"Loading bat classifier for ultrasonic analysis")
        from tf_model_bat_classifier import BatClassifier
        class_file = model_config["class_path"]
        classifier = BatClassifier(model_file, class_path=class_file)
        log_model_info(f"Bat classifier ready - target sample rate: {classifier.expected_sr}Hz")
    else:
        raise ValueError(f"Unknown analysis type: {model_analysis_type}")
    
    # Log audio preprocessing info and notify user about resampling
    if recorded_samplerate != classifier.expected_sr:
        print(f"  Resampling audio from {recorded_samplerate}Hz to {classifier.expected_sr}Hz")
        log_model_info(f"Resampling audio from {recorded_samplerate}Hz to {classifier.expected_sr}Hz")
    else:
        log_model_info(f"Audio sample rate matches classifier: {recorded_samplerate}Hz")
    
    # Run classification with timing
    log_model_info("Starting classification analysis...", show_memory=True)
    start_time = time.time()
    predictions, timestamps = classifier.classify(audio_data, recorded_samplerate)
    end_time = time.time()
    analysis_duration = end_time - start_time
    
    if SHOW_MODEL_TIMING:
        log_model_info(f"Analysis completed in {analysis_duration:.2f} seconds")
    
    print(f"  {model_name}: Analysis completed in {analysis_duration:.2f} seconds")
    
    # Apply calibration and migration adjustment for bird models (if available)
    if model_analysis_type == "birds" and model_region != "madagascar":
        try:
            import os
            import numpy as np
            
            # Try to load calibration parameters
            cal_path = "Pred_adjustment/calibration_params.npy"
            if os.path.exists(cal_path):
                log_model_info("Applying calibration adjustment")
                from functions import calibrate
                cal_table = np.load(cal_path)
                predictions = calibrate(predictions, cal_table=cal_table)
            
            # Try to apply migration adjustment if location and day_of_year provided
            migr_path = "Pred_adjustment/migration_params.npy"
            distribution_map_path = "Pred_adjustment/distribution_maps"
            if (latitude is not None and longitude is not None and day_of_year is not None 
                and os.path.exists(migr_path) and os.path.isdir(distribution_map_path)):
                log_model_info(f"Applying migration adjustment (lat={latitude:.4f}, lon={longitude:.4f}, day={day_of_year})")
                from functions import adjust
                migr_table = np.load(migr_path)
                
                # Migration adjustment only for classes > 1 (skip background/human)
                # Create arrays for adjustment
                preds_list = list(predictions)
                classes_list = list(range(len(predictions)))
                
                # Filter valid classes (> 1)
                valid_indices = [i for i in range(len(classes_list)) if classes_list[i] > 1]
                valid_classes = [classes_list[i] for i in valid_indices]
                valid_preds = [preds_list[i] for i in valid_indices]
                
                if valid_classes:
                    preds_adjusted = adjust(valid_preds, valid_classes, migr_table, latitude, longitude, day_of_year)
                    
                    # Replace adjusted predictions back
                    for i, idx in enumerate(valid_indices):
                        preds_list[idx] = preds_adjusted[i]
                    
                    predictions = preds_list
                    log_model_info(f"Migration adjustment applied to {len(valid_classes)} classes")
        except Exception as e:
            log_model_info(f"Warning: Could not apply prediction adjustments: {e}")
            print(f"  Warning: Prediction adjustments skipped: {e}")
    
    # Convert classifier output to detected species format
    log_model_info("Processing predictions and mapping to species names")
    detected_species = []
    if predictions and timestamps is not None:
        # Load species mapping for this model
        class_file = model_config["class_path"]
        log_model_info(f"Loading species mapping from {class_file}")
        sp_list = pd.read_csv(class_file)
        
        log_model_info(f"Model returned {len(predictions)} class predictions")
        
        # Get model-specific threshold
        model_threshold = model_config.get("threshold", 0.3)
        log_model_info(f"Applying detection threshold: {model_threshold}")
        
        # Count predictions above threshold
        detections_count = 0
        
        # Process predictions
        for i, (pred, timestamp) in enumerate(zip(predictions, timestamps)):
            # For bird models, skip background/human classes (0,1)
            # For bat models, include all classes including 0 (which is a valid bat species)
            skip_class = False
            if model_analysis_type == "birds" and i <= 1:
                skip_class = True
            elif model_analysis_type == "bats" and i == 21:
                # For bat models, class 21 is "Background" - we can optionally skip it
                skip_class = False  # Keep background for now to match reference behavior
            
            if not skip_class and pred > model_threshold:
                detections_count += 1
                species_class = int(i)  # Convert to Python int
                probability = float(pred)  # Convert numpy float32 to Python float
                timestamp_offset = float(timestamp)  # Convert numpy float to Python float
                
                # Find species name
                row = sp_list.loc[sp_list["class"] == species_class]
                if not row.empty:
                    species_name = row.iloc[0]["common_name"]
                    scientific_name = row.iloc[0]["scientific_name"]
                else:
                    species_name = f"Class {species_class}"
                    scientific_name = "Unknown"
                
                detected_species.append({
                    "species_class": species_class,
                    "probability": probability,
                    "timestamp_offset": timestamp_offset,
                    "species_name": species_name,
                    "scientific_name": scientific_name
                })
        
        log_model_info(f"Found {detections_count} species above threshold from {len(predictions)} classes")
    else:
        log_model_info("No predictions returned from classifier")
    
    log_model_info(f"COMPLETED {model_name} with {len(detected_species)} detections", show_memory=True)
    
    # Return model results
    return {
        "model_name": model_name,
        "model_region": model_region,
        "model_analysis_type": model_analysis_type,
        "model_description": model_config["description"],
        "model_file": model_file,
        "model_version": model_config.get("version", "unknown"),
        "threshold": model_threshold,
        "input_sample_rate_hz": recorded_samplerate,
        "processed_sample_rate_hz": classifier.expected_sr,
        "analysis_duration_seconds": analysis_duration,
        "detections": detected_species
    }


def run_multi_model_analysis(audio_data, recorded_samplerate, latitude=None, longitude=None, day_of_year=None):
    """
    Run multiple models on the same audio data.
    
    Args:
        audio_data: Audio buffer or file path
        recorded_samplerate: Sample rate of the audio
        latitude: GPS latitude for migration adjustment (optional)
        longitude: GPS longitude for migration adjustment (optional)
        day_of_year: Day of year for migration adjustment (optional)
    
    Returns:
        List of model results
    """
    model_results = []
    
    # Filter models based on run attribute
    enabled_models = [model for model in MODELS_TO_RUN if model.get("run", True)]
    print(f"Running analysis with {len(enabled_models)} enabled model(s) (out of {len(MODELS_TO_RUN)} total)...")
    
    for model_config in enabled_models:
            try:
                result = run_single_model(audio_data, recorded_samplerate, model_config, latitude, longitude, day_of_year)
                model_results.append(result)
                
                # Print summary for this model
                detection_count = len(result["detections"])
                if detection_count > 0:
                    print(f"  {result['model_name']}: {detection_count} species detected")
                    for detection in result["detections"]:
                        print(f"    - {detection['species_name']} ({detection['scientific_name']}) "
                              f"[Class {detection['species_class']}]: "
                              f"Probability {detection['probability']:.2f}, "
                              f"Offset {detection['timestamp_offset']:.2f}s")
                else:
                    print(f"  {result['model_name']}: No species detected")
            except Exception as e:
                print(f"  Error running {model_config['name']}: {e}")
    
    return model_results


def print_analysis_performance(model_results):
    """
    Print performance summary for all models.
    
    Args:
        model_results: List of model result dictionaries
    """
    if not model_results:
        return
        
    print("\nAnalysis Performance:")
    total_time = 0
    total_detections = 0
    
    for result in model_results:
        duration = result.get("analysis_duration_seconds", 0)
        analysis_type = result.get("model_analysis_type", "unknown")
        model_name = result.get("model_name", "unknown")
        detections = len(result.get("detections", []))
        
        total_time += duration
        total_detections += detections
        
        print(f"  • {model_name}: {duration:.2f}s ({analysis_type} analysis)")
    
    print(f"  • Total analysis time: {total_time:.2f}s")
    print(f"\n{total_detections} total detections from {len(model_results)} model(s)")