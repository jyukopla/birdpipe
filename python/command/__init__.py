"""
BirdPipe - Multi-modal audio analysis system for bird and bat species detection.

This package provides tools for:
- Recording audio from microphones
- Classifying bird and bat species from audio
- GPS-tagged observations
- GeoJSON output for analysis
"""

__version__ = "0.1.0"

from .config import Config, get_config, load_config, reload_config

__all__ = ["Config", "get_config", "load_config", "reload_config", "__version__"]
