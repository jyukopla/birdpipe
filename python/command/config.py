"""
BirdPipe Configuration Management

Handles reading and writing configuration from ~/.config/birdpipe/config.toml
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    import tomli_w
except ImportError:
    tomli_w = None


CONFIG_DIR = Path.home() / ".config" / "birdpipe"
CONFIG_FILE = CONFIG_DIR / "config.toml"


# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "audio": {
        "microphone_name": "default",
        "sample_rate": 48000,
        "duration": 10,
        "save_to_disk": True,
        "save_path": "/var/data/audiofiles",
        "use_flac": True,
        "flac_compression_level": 5,
    },
    "test": {
        "enabled": False,
        "audio_path": "sample_audio",
        "audio_file": "FullSpectrum_Pipistrellus_pipistrellus_384KHz.wav",
        "show_segment_detections": True,
        "day_of_year": None,  # Override day of year for testing (1-366), None = use current date
    },
    "debug": {
        "model_information": True,
        "show_model_timing": True,
        "show_memory_usage": True,
    },
    "geojson": {
        "save_path": "/var/data/geojson",
    },
    "gps": {
        "required": False,
        "timeout_seconds": 60,
        "min_accuracy_meters": 100,
        "retry_attempts": 1,
        "retry_delay": 2,
        "default_latitude": 62.178,
        "default_longitude": 25.710,
    },
    "paths": {
        "temp_path": "/var/data/temp",
    },
    "filenames": {
        "short_audio_filename": True,
    },
    "models": [
        {
            "name": "finland_birds",
            "region": "finland",
            "analysis_type": "birds",
            "description": "Finland bird species classifier",
            "version": "v3.5",
            "model_path": "bsg_fin_v3.5.tflite",
            "class_path": "classes_finland.csv",
            "threshold": 0.25,
            "run": True,
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
            "run": False,
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
            "run": True,
        },
    ],
}


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def config_exists() -> bool:
    """Check if configuration file exists."""
    return CONFIG_FILE.exists()


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file.

    Returns:
        Configuration dictionary, or default config if file doesn't exist.
    """
    if not config_exists():
        return DEFAULT_CONFIG.copy()

    if tomllib is None:
        print("Warning: tomllib/tomli not available, using default config")
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)

        # Merge with defaults for any missing keys
        merged = _deep_merge(DEFAULT_CONFIG.copy(), config)
        return merged
    except Exception as e:
        print(f"Warning: Failed to load config from {CONFIG_FILE}: {e}")
        print("Using default configuration.")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save.

    Returns:
        True if successful, False otherwise.
    """
    if tomli_w is None:
        print("Error: tomli_w not installed. Install it with: pip install tomli-w")
        return False

    try:
        ensure_config_dir()
        # Remove None values as TOML doesn't support them
        clean_config = _remove_none_values(config)
        with open(CONFIG_FILE, "wb") as f:
            tomli_w.dump(clean_config, f)
        return True
    except Exception as e:
        print(f"Error saving config to {CONFIG_FILE}: {e}")
        return False


def create_default_config(overwrite: bool = False) -> bool:
    """
    Create default configuration file.

    Args:
        overwrite: If True, overwrite existing config file.

    Returns:
        True if config was created, False otherwise.
    """
    if config_exists() and not overwrite:
        return False

    return save_config(DEFAULT_CONFIG)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _remove_none_values(data: Any) -> Any:
    """
    Recursively remove None values from dictionaries and lists.

    Args:
        data: Data structure to clean (dict, list, or other).

    Returns:
        Cleaned data structure with None values removed.
    """
    if isinstance(data, dict):
        return {k: _remove_none_values(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [_remove_none_values(item) for item in data if item is not None]
    else:
        return data


def get_config_path() -> Path:
    """Get the configuration file path."""
    return CONFIG_FILE


# Convenience functions to get specific config values
def get_audio_config() -> Dict[str, Any]:
    """Get audio configuration section."""
    return load_config().get("audio", DEFAULT_CONFIG["audio"])


def get_test_config() -> Dict[str, Any]:
    """Get test configuration section."""
    return load_config().get("test", DEFAULT_CONFIG["test"])


def get_debug_config() -> Dict[str, Any]:
    """Get debug configuration section."""
    return load_config().get("debug", DEFAULT_CONFIG["debug"])


def get_geojson_config() -> Dict[str, Any]:
    """Get GeoJSON configuration section."""
    return load_config().get("geojson", DEFAULT_CONFIG["geojson"])


def get_gps_config() -> Dict[str, Any]:
    """Get GPS configuration section."""
    return load_config().get("gps", DEFAULT_CONFIG["gps"])


def get_paths_config() -> Dict[str, Any]:
    """Get paths configuration section."""
    return load_config().get("paths", DEFAULT_CONFIG["paths"])


def get_models_config() -> List[Dict[str, Any]]:
    """Get models configuration list."""
    return load_config().get("models", DEFAULT_CONFIG["models"])


class Config:
    """
    Configuration class that provides attribute-style access to config values.

    This class provides backwards compatibility with the old recorder_config.py style.
    """

    def __init__(self):
        self._config = load_config()
        self._setup_attributes()

    def _setup_attributes(self):
        """Set up attributes from config dictionary."""
        audio = self._config.get("audio", {})
        test = self._config.get("test", {})
        debug = self._config.get("debug", {})
        geojson = self._config.get("geojson", {})
        gps = self._config.get("gps", {})
        paths = self._config.get("paths", {})
        filenames = self._config.get("filenames", {})

        # Audio settings
        self.MICROPHONE_NAME = audio.get(
            "microphone_name", DEFAULT_CONFIG["audio"]["microphone_name"]
        )
        self.RECORDING_SAMPLE_RATE = audio.get(
            "sample_rate", DEFAULT_CONFIG["audio"]["sample_rate"]
        )
        self.RECORDING_DURATION = audio.get(
            "duration", DEFAULT_CONFIG["audio"]["duration"]
        )
        self.SAVE_AUDIO_TO_DISK = audio.get(
            "save_to_disk", DEFAULT_CONFIG["audio"]["save_to_disk"]
        )
        self.AUDIO_SAVE_PATH = os.path.expanduser(
            audio.get("save_path", DEFAULT_CONFIG["audio"]["save_path"])
        )
        self.USE_FLAC_COMPRESSION = audio.get(
            "use_flac", DEFAULT_CONFIG["audio"]["use_flac"]
        )
        self.FLAC_COMPRESSION_LEVEL = audio.get(
            "flac_compression_level", DEFAULT_CONFIG["audio"]["flac_compression_level"]
        )

        # Test settings
        self.TEST_MODE = test.get("enabled", DEFAULT_CONFIG["test"]["enabled"])
        self.TEST_AUDIO_PATH = os.path.expanduser(
            test.get("audio_path", DEFAULT_CONFIG["test"]["audio_path"])
        )
        self.TEST_AUDIO_FILE = test.get(
            "audio_file", DEFAULT_CONFIG["test"]["audio_file"]
        )
        self.SHOW_SEGMENT_DETECTIONS = test.get(
            "show_segment_detections", DEFAULT_CONFIG["test"]["show_segment_detections"]
        )
        self.TEST_DAY_OF_YEAR = test.get(
            "day_of_year", DEFAULT_CONFIG["test"]["day_of_year"]
        )

        # Debug settings
        self.MODEL_INFORMATION = debug.get(
            "model_information", DEFAULT_CONFIG["debug"]["model_information"]
        )
        self.SHOW_MODEL_TIMING = debug.get(
            "show_model_timing", DEFAULT_CONFIG["debug"]["show_model_timing"]
        )
        self.SHOW_MEMORY_USAGE = debug.get(
            "show_memory_usage", DEFAULT_CONFIG["debug"]["show_memory_usage"]
        )

        # GeoJSON settings
        self.GEOJSON_SAVE_PATH = os.path.expanduser(
            geojson.get("save_path", DEFAULT_CONFIG["geojson"]["save_path"])
        )

        # GPS settings
        self.GPS_REQUIRED = gps.get("required", DEFAULT_CONFIG["gps"]["required"])
        self.GPS_TIMEOUT_SECONDS = gps.get(
            "timeout_seconds", DEFAULT_CONFIG["gps"]["timeout_seconds"]
        )
        self.GPS_MIN_ACCURACY_METERS = gps.get(
            "min_accuracy_meters", DEFAULT_CONFIG["gps"]["min_accuracy_meters"]
        )
        self.GPS_RETRY_ATTEMPTS = gps.get(
            "retry_attempts", DEFAULT_CONFIG["gps"]["retry_attempts"]
        )
        self.GPS_RETRY_DELAY = gps.get(
            "retry_delay", DEFAULT_CONFIG["gps"]["retry_delay"]
        )
        self.DEFAULT_LATITUDE = gps.get(
            "default_latitude", DEFAULT_CONFIG["gps"]["default_latitude"]
        )
        self.DEFAULT_LONGITUDE = gps.get(
            "default_longitude", DEFAULT_CONFIG["gps"]["default_longitude"]
        )

        # Paths
        self.TEMP_PATH = os.path.expanduser(
            paths.get("temp_path", DEFAULT_CONFIG["paths"]["temp_path"])
        )

        # Filenames
        self.SHORT_AUDIO_FILENAME = filenames.get(
            "short_audio_filename", DEFAULT_CONFIG["filenames"]["short_audio_filename"]
        )

        # Models
        self.MODELS_TO_RUN = self._config.get("models", DEFAULT_CONFIG["models"])

    def reload(self):
        """Reload configuration from file."""
        self._config = load_config()
        self._setup_attributes()


# Global config instance for backwards compatibility
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config() -> Config:
    """Reload the global configuration instance."""
    global _config_instance
    _config_instance = Config()
    return _config_instance
