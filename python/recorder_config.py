"""
BirdPipe Recorder Configuration - Compatibility Shim

This module provides backwards compatibility with the old configuration style.
Configuration is now stored in ~/.config/birdpipe/config.toml

To create a new configuration file, run:
    birdpipe create-config

To edit the configuration:
    birdpipe edit-config

To show current configuration:
    birdpipe show-config
"""

import sys
import os

# Try to import from new config module
try:
    from command.config import get_config, config_exists, DEFAULT_CONFIG
    
    # Load config if it exists, otherwise use defaults
    if config_exists():
        _config = get_config()
    else:
        # Use hardcoded defaults for backwards compatibility
        _config = None
except ImportError:
    _config = None

# If new config system is available and config exists, use it
if _config is not None:
    # Audio recording settings
    MICROPHONE_NAME = _config.MICROPHONE_NAME
    RECORDING_SAMPLE_RATE = _config.RECORDING_SAMPLE_RATE
    RECORDING_DURATION = _config.RECORDING_DURATION

    # Test mode configuration
    TEST_MODE = _config.TEST_MODE
    TEST_AUDIO_PATH = _config.TEST_AUDIO_PATH
    TEST_AUDIO_FILE = _config.TEST_AUDIO_FILE
    SHOW_SEGMENT_DETECTIONS = _config.SHOW_SEGMENT_DETECTIONS

    # Audio storage settings
    SAVE_AUDIO_TO_DISK = _config.SAVE_AUDIO_TO_DISK
    AUDIO_SAVE_PATH = _config.AUDIO_SAVE_PATH

    # Audio compression settings
    USE_FLAC_COMPRESSION = _config.USE_FLAC_COMPRESSION
    FLAC_COMPRESSION_LEVEL = _config.FLAC_COMPRESSION_LEVEL

    # Model information and debugging settings
    MODEL_INFORMATION = _config.MODEL_INFORMATION
    SHOW_MODEL_TIMING = _config.SHOW_MODEL_TIMING
    SHOW_MEMORY_USAGE = _config.SHOW_MEMORY_USAGE

    # GeoJSON saving settings
    GEOJSON_SAVE_PATH = _config.GEOJSON_SAVE_PATH

    # GPS configuration
    GPS_REQUIRED = _config.GPS_REQUIRED
    GPS_TIMEOUT_SECONDS = _config.GPS_TIMEOUT_SECONDS
    GPS_MIN_ACCURACY_METERS = _config.GPS_MIN_ACCURACY_METERS
    GPS_RETRY_ATTEMPTS = _config.GPS_RETRY_ATTEMPTS
    GPS_RETRY_DELAY = _config.GPS_RETRY_DELAY

    # Default coordinates when GPS fails
    DEFAULT_LATITUDE = _config.DEFAULT_LATITUDE
    DEFAULT_LONGITUDE = _config.DEFAULT_LONGITUDE

    TEMP_PATH = _config.TEMP_PATH

    # Model configuration
    MODELS_TO_RUN = _config.MODELS_TO_RUN

    # Filename settings
    SHORT_AUDIO_FILENAME = _config.SHORT_AUDIO_FILENAME

else:
    # Fallback to hardcoded defaults if new config system is not available
    # Audio recording settings
    MICROPHONE_NAME = "UAC 1.0 Microphone & HID-Mediak: USB Audio (hw:1,0)"
    RECORDING_SAMPLE_RATE = 48000
    RECORDING_DURATION = 10

    # Test mode configuration
    TEST_MODE = True
    TEST_AUDIO_PATH = "sample_audio"
    TEST_AUDIO_FILE = "FullSpectrum_Pipistrellus_pipistrellus_384KHz.wav"
    SHOW_SEGMENT_DETECTIONS = True

    # Audio storage settings
    SAVE_AUDIO_TO_DISK = True
    AUDIO_SAVE_PATH = "/var/data/audiofiles"

    # Audio compression settings
    USE_FLAC_COMPRESSION = True
    FLAC_COMPRESSION_LEVEL = 5

    # Model information and debugging settings
    MODEL_INFORMATION = True
    SHOW_MODEL_TIMING = True
    SHOW_MEMORY_USAGE = True

    # GeoJSON saving settings
    GEOJSON_SAVE_PATH = "/var/data/geojson"

    # GPS configuration
    GPS_REQUIRED = False
    GPS_TIMEOUT_SECONDS = 60
    GPS_MIN_ACCURACY_METERS = 100
    GPS_RETRY_ATTEMPTS = 1
    GPS_RETRY_DELAY = 2

    # Default coordinates when GPS fails
    DEFAULT_LATITUDE = 62.178
    DEFAULT_LONGITUDE = 25.710

    TEMP_PATH = "/var/data/temp"

    # Model configuration
    MODELS_TO_RUN = [
        {
            "name": "finland_birds", 
            "region": "finland",
            "analysis_type": "birds", 
            "description": "Finland bird species classifier",
            "version": "v3.5",
            "model_path": "bsg_fin_v3.5.tflite",
            "class_path": "classes_finland.csv",
            "threshold": 0.25,
            "run": True
        },
        {
            "name": "madagascar_birds", 
            "region": "madagascar",
            "analysis_type": "birds", 
            "description": "Madagascar bird species classifier",
            "version": "v3.0",
            "model_path": "bsg_mad_v3.tflite",
            "class_path": "classes_madagascar.csv",
            "threshold": 0.25,
            "run": False
        },
        {
            "name": "europe_bats", 
            "region": "europe",
            "analysis_type": "bats", 
            "description": "European bat species classifier for ultrasonic recordings",
            "version": "v1.0",
            "model_path": "bsg_europe_bat_v1.tflite",
            "class_path": "classes_bats.csv",
            "threshold": 0.5,
            "run": True
        }
    ]

    # Filename settings
    SHORT_AUDIO_FILENAME = True

# Validate sample rate
SUPPORTED_SAMPLE_RATES = [44100, 48000, 192000, 384000]
assert RECORDING_SAMPLE_RATE in SUPPORTED_SAMPLE_RATES, f"RECORDING_SAMPLE_RATE must be one of {SUPPORTED_SAMPLE_RATES}, got {RECORDING_SAMPLE_RATE}"

# Validate FLAC compression level
if USE_FLAC_COMPRESSION:
    assert 0 <= FLAC_COMPRESSION_LEVEL <= 8, f"FLAC_COMPRESSION_LEVEL must be 0-8, got {FLAC_COMPRESSION_LEVEL}"

