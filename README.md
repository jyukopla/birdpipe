# BirdPipe Audio Recorder

A multi-modal audio analysis system for real-time bird and bat species detection with GPS-tagged observations. Designed for autonomous wildlife monitoring in field deployments.

## Features

- **Multi-Species Detection**: Simultaneous bird and bat classification using AI models
- **GPS Integration**: High-precision location tagging with GNSS/PPS timing support
- **Multi-Sample Rate Support**:
  - 48kHz for bird vocalization analysis
  - 384kHz for ultrasonic bat echolocation detection
- **Multi-Model Analysis**: Run multiple regional models simultaneously (e.g., Finland + Madagascar)
- **Flexible Storage**: RAM-only or disk storage modes with FLAC compression
- **GeoJSON Output**: Standards-compliant geographic data format for analysis and visualization
- **Test Mode**: Validate configuration with sample audio files before deployment
- **Production Ready**: Atomic file operations, robust error handling, and performance monitoring

## System Requirements

### Hardware

- **Microphone**: USB audio interface supporting your target sample rate
  - Standard (48kHz): Any USB microphone for bird monitoring
  - Ultrasonic (192kHz-384kHz): Specialized ultrasonic microphone for bat monitoring
- **GPS Module** (optional): GNSS receiver with PPS (Pulse Per Second) for precise timing
- **Recommended Platform**: Raspberry Pi 4 or equivalent (2GB+ RAM)
- **Storage**: Minimum 8GB for system + data storage (varies with recording duration)

### Software

- Python 3.10 or higher
- Linux OS (tested on Raspberry Pi OS)
- ALSA audio system

## Installation

### 1. Install System Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y chrony git libgdal-dev libportaudio2 gpsd gpsd-clients
```

### 2. Clone the Repository

```bash
git clone <repository-url>
cd python
```

### 3. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 4. Verify Audio Devices

List available audio devices on your system:

```bash
uv run birdpipe list-devices
```

Note the exact name of your microphone for configuration.

## Command Line Interface

BirdPipe provides a command-line interface that can be run with `uv run`:

```bash
uv run birdpipe --help
```

### Available Commands

```bash
# List all available commands
uv run birdpipe --help

# List audio devices
uv run birdpipe list-devices
uv run birdpipe list-devices --verbose  # Show supported sample rates

# Create configuration file
uv run birdpipe create-config
uv run birdpipe create-config --force  # Overwrite existing

# Show current configuration
uv run birdpipe show-config
uv run birdpipe show-config --path  # Just show file path

# Edit configuration
uv run birdpipe edit-config

# Validate configuration
uv run birdpipe validate-config

# Test microphone
uv run birdpipe test-microphone
uv run birdpipe test-microphone --microphone "UltraMic" --sample-rate 384000

# Record and analyze audio
uv run birdpipe record
uv run birdpipe record --test  # Use test mode with sample audio
uv run birdpipe record --duration 30 --sample-rate 48000

# Show system info
uv run birdpipe info
```

## Configuration

Configuration is stored in `~/.config/birdpipe/config.toml`. Create a default configuration:

```bash
uv run birdpipe create-config
```

Edit the configuration:

```bash
uv run birdpipe edit-config
```

### Configuration File Format (TOML)

```toml
[audio]
microphone_name = "default"
sample_rate = 48000
duration = 10
save_to_disk = true
save_path = "~/birdpipe-audiofiles"
use_flac = true
flac_compression_level = 5

[test]
enabled = false
audio_path = "sample_audio"
audio_file = "sample.wav"
show_segment_detections = true

[debug]
model_information = true
show_model_timing = true
show_memory_usage = true

[geojson]
save_path = "~/birdpipe-geojson"

[gps]
required = false
timeout_seconds = 60
min_accuracy_meters = 100
retry_attempts = 1
retry_delay = 2
default_latitude = 62.178
default_longitude = 25.710

[paths]
temp_path = "/tmp"

[[models]]
name = "finland_birds"
region = "finland"
analysis_type = "birds"
description = "Finland bird species classifier"
version = "v3.5"
model_path = "bsg_fin_v3.5.tflite"
class_path = "classes_finland.csv"
threshold = 0.25
run = true

[[models]]
name = "europe_bats"
region = "europe"
analysis_type = "bats"
description = "European bat species classifier"
version = "v1.0"
model_path = "bsg_europe_bat_v1.tflite"
class_path = "classes_bats.csv"
threshold = 0.5
run = true
```

### Legacy Configuration

The old `recorder_config.py` file is still supported for backwards compatibility.
It will automatically use settings from `~/.config/birdpipe/config.toml` if available.

### Basic Recording Settings (Legacy)

```python
# Microphone configuration - microphone name and properties can be found by using: `uv run birdpipe list-devices`
MICROPHONE_NAME = "UAC 1.0 Microphone & HID-Mediak: USB Audio (hw:1,0)"
RECORDING_SAMPLE_RATE = 48000  # Hz (48000 for birds, 384000 for bats)
RECORDING_DURATION = 10  # seconds per recording

# Audio storage
SAVE_AUDIO_TO_DISK = True
AUDIO_SAVE_PATH = "/var/data/audiofiles"
USE_FLAC_COMPRESSION = True
FLAC_COMPRESSION_LEVEL = 5  # 0-8, higher = smaller files
```

### GPS Settings

```python
GPS_REQUIRED = False  # Set True to abort if no GPS fix
GPS_TIMEOUT_SECONDS = 60
GPS_MIN_ACCURACY_METERS = 100

# Fallback coordinates (used when GPS_REQUIRED = False)
DEFAULT_LATITUDE = 62.178
DEFAULT_LONGITUDE = 25.710
```

### Multi-Model Configuration

```python
ENABLE_MULTI_MODEL = True  # Run multiple models simultaneously

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
        "run": True  # Enable/disable this model
    },
    {
        "name": "europe_bats",
        "region": "europe",
        "analysis_type": "bats",
        "description": "European bat species classifier",
        "version": "v1.0",
        "model_path": "bsg_europe_bat_v1.tflite",
        "class_path": "classes_bats.csv",
        "threshold": 0.5,
        "run": True
    }
]
```

### Output Paths

```python
GEOJSON_SAVE_PATH = "/var/data/geojson"
TEMP_PATH = "/var/data/temp"
```

## Usage

### Test Mode

Test your configuration with sample audio files before live recording:

```bash
# Using CLI
uv run birdpipe record --test

# Or enable test mode in config and run
uv run birdpipe edit-config  # Set test.enabled = true
uv run birdpipe record
```

### Production Mode

```bash
# Disable test mode in config
uv run birdpipe edit-config  # Set test.enabled = false

# Run recorder
uv run birdpipe record
```

### Continuous Monitoring

For continuous monitoring, use systemd or cron:

**systemd service example** (`/etc/systemd/system/birdpipe-recorder.service`):

```ini
[Unit]
Description=BirdPipe Audio Recorder
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/birdpipe/python
ExecStart=/home/pi/.local/bin/uv run birdpipe record
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable birdpipe-recorder
sudo systemctl start birdpipe-recorder
sudo systemctl status birdpipe-recorder
```

**Cron example** (run every 15 minutes):

```bash
crontab -e
# Add:
*/15 * * * * cd /home/pi/birdpipe/recorder && /usr/local/bin/uv run recorder.py
```

## Output Files

### Audio Files

Saved to `AUDIO_SAVE_PATH` (if `SAVE_AUDIO_TO_DISK = True`):

- Format: `{DEVICE_NAME}_{TIMESTAMP}.{flac|wav}`
- Example: `raspberry-pi_2025-11-14T12-30-45Z.flac`

### GeoJSON Files

Saved to `GEOJSON_SAVE_PATH`:

- Format: `{DEVICE_NAME}.geojson`
- Contains all detections with GPS coordinates, timestamps, confidence scores
- Follows GeoJSON Feature Collection specification
- Can be imported into QGIS, ArcGIS, or any GIS software

**GeoJSON Structure:**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [25.71, 62.178]
      },
      "properties": {
        "species": "Parus major",
        "common_name": "Great Tit",
        "confidence": 0.87,
        "timestamp": "2025-11-14T12:30:45Z",
        "model": "finland_birds",
        "audio_file": "raspberry-pi_2025-11-14T12-30-45Z.flac"
      }
    }
  ]
}
```

## Troubleshooting

### Microphone Not Found

```bash
# List all audio devices
uv run birdpipe list-devices

# Update MICROPHONE_NAME in recorder_config.py with exact name
```

### Sample Rate Not Supported

- Check microphone specifications
- Some devices only support specific sample rates (e.g., 48kHz, 96kHz)
- Ultrasonic microphones required for bat monitoring (≥192kHz)

### GPS Not Working

- Verify GPS module is connected and has clear sky view
- Check GPS device permissions: `sudo usermod -a -G dialout $USER`
- Set `GPS_REQUIRED = False` to use fallback coordinates during testing

### Low Detection Confidence

- Check microphone positioning and background noise
- Adjust model thresholds in `MODELS_TO_RUN` configuration
- Verify correct regional model is selected for your location

### Memory Issues

- Reduce `RECORDING_DURATION` for longer recordings
- Enable `SAVE_AUDIO_TO_DISK = False` for RAM-only processing
- Disable unused models in `MODELS_TO_RUN`

## Project Structure

```
python/
├── pyproject.toml                     # Python project configuration
├── uv.lock                            # Dependency lock file
├── recorder.py                        # Main application entry point
├── recorder_config.py                 # Configuration settings
├── recorder_analysis.py               # Multi-model analysis coordination
├── recorder_record_audio.py           # Audio capture and file handling
├── recorder_generate_geojson.py       # GeoJSON output generation
├── recorder_get_gnss_location.py      # GPS/GNSS interface
├── recorder_get_pps_status.py         # PPS timing signal detection
├── recorder_get_day_of_the_year.py    # Date utilities
├── recorder_local_identification.py   # Local species identification
├── tf_model_bird_classifier.py        # Bird AI model wrapper
├── tf_model_bat_classifier.py         # Bat AI model wrapper
├── tf_model_base_classifier.py        # Base classifier interface
├── bird_classifier_utils.py           # Bird classifier utilities
├── list_audio_devices.py              # Audio device enumeration utility
├── rclone_sync_folder.py              # Remote sync utility
├── bsg_fin_v3.5.tflite                # Finland bird classifier model
├── bsg_mad_v3.tflite                  # Madagascar bird classifier model
├── bsg_europe_bat_v1.tflite           # European bat classifier model
├── classes_finland.csv                # Finland species labels
├── classes_madagascar.csv             # Madagascar species labels
├── classes_bats.csv                   # Bat species labels
├── command/                           # CLI module
│   ├── __init__.py
│   ├── cli.py                         # Main CLI entry point
│   ├── audio_devices.py               # Audio device commands
│   ├── config.py                      # Configuration commands
│   └── recorder_main.py               # Recorder commands
├── BN-2.4_tflite/                     # BirdNET model
│   └── BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
└── Pred_adjustment/                   # Prediction adjustment data
    ├── calibration_params.npy
    ├── migration_params.npy
    └── distribution_maps/
```

## Performance

Typical performance on Raspberry Pi 4 (8GB):

- **Bird Analysis (48kHz)**: ~2-3x realtime (10s audio analyzed in 3-5s)
- **Bat Analysis (384kHz)**: ~1-2x realtime (10s audio analyzed in 5-10s)
- **Multi-Model**: Processes models sequentially, time scales linearly
- **Memory Usage**: 200-500MB depending on recording duration and models

## Supported Species

### Birds

- **Finland Model**: 250+ species common to Northern Europe
- **Madagascar Model**: 200+ endemic and regional species

### Bats

- **Europe Model**: 30+ bat species across Europe
- Requires ultrasonic microphone (≥192kHz sampling)

## Support

- Issues: [GitHub Issues URL]
- Documentation: [Documentation URL]
- Contact: [Contact information]

## Version History

- **v0.1.0** (2026-01-25): Initial release
  - Multi-model bird and bat analysis
  - GPS integration
  - GeoJSON output format
  - Test mode for validation
