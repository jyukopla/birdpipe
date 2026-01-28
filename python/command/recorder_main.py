"""
BirdPipe Recorder Main Module

Main recording and analysis logic, adapted to use the new configuration system.
"""

import os
import sys
import socket
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import soundfile as sf

from .config import get_config, reload_config


DEVICE_NAME = socket.gethostname()


def load_sample_audio(test_audio_path: str, test_audio_file: str):
    """
    Load sample audio file for testing purposes.

    Args:
        test_audio_path: Directory containing test audio.
        test_audio_file: Name of the test audio file.

    Returns:
        tuple: (audio_data, sample_rate) or (None, None) if file not found
    """
    sample_file_path = os.path.join(test_audio_path, test_audio_file)

    if not os.path.exists(sample_file_path):
        print(f"Warning: Sample audio file not found at {sample_file_path}")
        return None, None

    try:
        audio_data, sample_rate = sf.read(sample_file_path)

        print(f"Loaded sample audio: {test_audio_file}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"  Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1}")
        print(f"  Memory usage: {audio_data.nbytes / (1024 * 1024):.1f} MB")
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading sample audio file: {e}")
        return None, None


def get_reboot_segment():
    """
    Return a reboot segment identifier derived from system boot time.
    """
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.readline().split()[0])
        boot_time = datetime.utcnow() - timedelta(seconds=uptime_seconds)
        return boot_time.isoformat(timespec="seconds") + "Z"
    except Exception:
        return "unknown"


def main(overrides: Optional[Dict[str, Any]] = None):
    """
    Main recording and analysis function.

    Args:
        overrides: Optional dictionary of config overrides from CLI.
    """
    # Get configuration
    config = get_config()

    # Apply any CLI overrides
    if overrides:
        if "test_mode" in overrides:
            config.TEST_MODE = overrides["test_mode"]
        if "duration" in overrides:
            config.RECORDING_DURATION = overrides["duration"]
        if "sample_rate" in overrides:
            config.RECORDING_SAMPLE_RATE = overrides["sample_rate"]
        if "microphone_name" in overrides:
            config.MICROPHONE_NAME = overrides["microphone_name"]

    # Import the legacy modules - they should work with the config object
    # We add the parent directory to path to import legacy modules
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from recorder_analysis import (
            run_multi_model_analysis,
            print_analysis_performance,
        )
        from recorder_generate_geojson import generate_geojson
        from recorder_get_day_of_the_year import get_day_of_year
        from recorder_get_gnss_location import get_gnss_location
        from recorder_get_pps_status import get_pps_status
        from recorder_record_audio import find_microphone, record_audio
    except ImportError as e:
        print(f"Error importing recorder modules: {e}")
        print("Make sure you're running from the correct directory.")
        sys.exit(1)

    try:
        # Suppress NumPy warnings in production mode only
        if not config.TEST_MODE:
            warnings.filterwarnings(
                "ignore", message="The value of the smallest subnormal"
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="numpy.core.getlimits"
            )

        # Show current operation mode
        if config.TEST_MODE:
            print("=== BIRDPIPE RECORDER - TEST MODE ===")
            print(f"Using sample audio: {config.TEST_AUDIO_FILE}")
        else:
            print("=== BIRDPIPE RECORDER - PRODUCTION MODE ===")
            print(f"Microphone: {config.MICROPHONE_NAME}")
            print(f"Sample rate: {config.RECORDING_SAMPLE_RATE} Hz")
            print(f"Duration: {config.RECORDING_DURATION} seconds")
            save_format = (
                f"FLAC (level {config.FLAC_COMPRESSION_LEVEL})"
                if config.USE_FLAC_COMPRESSION
                else "WAV"
            )
            print(
                f"Save audio to disk: {'Yes' if config.SAVE_AUDIO_TO_DISK else 'No (RAM only)'} ({save_format})"
            )
        print()

        # Ensure output directories exist
        os.makedirs(config.GEOJSON_SAVE_PATH, exist_ok=True)
        os.makedirs(config.AUDIO_SAVE_PATH, exist_ok=True)
        os.makedirs(config.TEMP_PATH, exist_ok=True)

        # PPS status (hardware GNSS timing) - only check if GPS is required
        if config.GPS_REQUIRED:
            if get_pps_status():
                print("GNSS PPS Signal OK")
            else:
                print("GNSS PPS Signal not detected. Continuing without PPS.")

        # Day-of-year is used by the analysis model
        if config.TEST_MODE and config.TEST_DAY_OF_YEAR is not None:
            day_of_year = config.TEST_DAY_OF_YEAR
            print(f"Day of the year: {day_of_year} (test override)")
        else:
            day_of_year = get_day_of_year()
            print(f"Day of the year: {day_of_year}")

        # Handle GPS location based on GPS_REQUIRED setting
        if config.GPS_REQUIRED:
            # GPS is required - attempt to get location
            try:
                gps_result = get_gnss_location()

                lat = lon = fix_status = accuracy_m = gps_source = None
                if isinstance(gps_result, tuple):
                    if len(gps_result) == 5:
                        lat, lon, fix_status, accuracy_m, gps_source = gps_result
                    elif len(gps_result) == 4:
                        lat, lon, fix_status, accuracy_m = gps_result
                        gps_source = "unknown"
                    elif len(gps_result) == 3:
                        lat, lon, fix_status = gps_result
                        gps_source = "unknown"
                    elif len(gps_result) == 2:
                        lat, lon = gps_result
                        gps_source = "unknown"

                # GPS status interpretation
                fix_descriptions = {0: "No Fix", 1: "GPS", 2: "DGPS", 3: "3D FIX"}
                fix_description = fix_descriptions.get(fix_status, "Unknown")

                if accuracy_m == float("inf"):
                    accuracy_desc = "Unknown"
                else:
                    accuracy_desc = f"{accuracy_m:.1f} m"

                print(
                    f"GPS fix: {fix_description}, Horizontal accuracy: {accuracy_desc}"
                )
                print(f"Latitude: {lat:.9f}, Longitude: {lon:.9f}")
                print(f"GPS source: {gps_source}")

            except Exception as e:
                print(f"GPS error: {e}")
                print("GPS is required but failed. Aborting recording.")
                return
        else:
            # GPS is not required - use default coordinates
            print("GPS not required. Using default coordinates.")
            lat, lon, fix_status, accuracy_m, gps_source = (
                config.DEFAULT_LATITUDE,
                config.DEFAULT_LONGITUDE,
                0,
                float("inf"),
                "default",
            )

        # Locate microphone by name (skip in test mode)
        mic_index = None
        if not config.TEST_MODE:
            mic_index = find_microphone(config.MICROPHONE_NAME)
            if mic_index is None:
                print(f"\n{'=' * 80}")
                print("MICROPHONE NOT FOUND")
                print(f"{'=' * 80}")
                print(f"\nCould not find microphone: '{config.MICROPHONE_NAME}'")
                print(f"\nSOLUTION:")
                print(f"1. Check microphone is connected")
                print(f"2. Run 'birdpipe list-devices' to see available devices")
                print(f"3. Update microphone_name in config: birdpipe edit-config")
                print(f"{'=' * 80}\n")
                return

            print(f"Microphone found at index {mic_index}")

        reboot_segment = get_reboot_segment()

        # Perform the recording or load sample audio
        if config.TEST_MODE:
            print("=== TEST MODE ENABLED ===")
            print(f"Loading sample audio instead of recording...")
            audio_buffer, recorded_samplerate = load_sample_audio(
                config.TEST_AUDIO_PATH, config.TEST_AUDIO_FILE
            )
            if audio_buffer is None:
                print("Failed to load sample audio. Exiting.")
                return

            # Create mock timing info for test mode
            from time import time_ns

            current_time = time_ns()
            timing_info = {
                "start_utc": datetime.fromtimestamp(current_time / 1e9).isoformat()
                + "Z",
                "end_utc": datetime.fromtimestamp(
                    (current_time + len(audio_buffer) / recorded_samplerate * 1e9) / 1e9
                ).isoformat()
                + "Z",
                "wall_duration_s": len(audio_buffer) / recorded_samplerate,
                "nominal_duration_s": len(audio_buffer) / recorded_samplerate,
                "delta_ms": 0.0,
                "start_unix_ns": current_time,
            }
            audio_filepath = None
            temp_audio_filepath = None
        else:
            # Perform live recording
            audio_buffer, timing_info, audio_filepath, recorded_samplerate = (
                record_audio(
                    mic_index,
                    duration=config.RECORDING_DURATION,
                    save_to_disk=config.SAVE_AUDIO_TO_DISK,
                    save_path=config.AUDIO_SAVE_PATH,
                    device_name=DEVICE_NAME,
                    sample_rate=config.RECORDING_SAMPLE_RATE,
                    lat=lat,
                    lon=lon,
                    fix_status=fix_status,
                    accuracy_m=accuracy_m,
                    reboot_segment=reboot_segment,
                    model_file=None,
                    use_flac_compression=config.USE_FLAC_COMPRESSION,
                    flac_compression_level=config.FLAC_COMPRESSION_LEVEL,
                )
            )

            # Move recorded WAV into a temp area for analysis
            temp_audio_filepath = (
                os.path.join(config.TEMP_PATH, os.path.basename(audio_filepath))
                if audio_filepath
                else None
            )
            if audio_filepath:
                os.replace(audio_filepath, temp_audio_filepath)

        # Proceed only if we have audio in memory or a temp file on disk
        if (audio_buffer is not None) or temp_audio_filepath:
            print("Recording complete. Starting analysis...")

            # Unpack timing info
            if timing_info:
                recording_time_unix_ns = timing_info.get("start_unix_ns")
            else:
                recording_time_unix_ns = None

            # Run multi-model analysis
            model_results = run_multi_model_analysis(
                audio_buffer if audio_buffer is not None else temp_audio_filepath,
                recorded_samplerate,
                latitude=lat,
                longitude=lon,
                day_of_year=day_of_year,
            )

            # Generate audio filename
            if audio_filepath:
                audio_filename = os.path.basename(audio_filepath)
            else:
                from datetime import timezone

                if recording_time_unix_ns:
                    start_dt_utc = datetime.fromtimestamp(
                        recording_time_unix_ns / 1e9, tz=timezone.utc
                    )
                    from recorder_record_audio import safe_iso8601

                    timestamp_str = safe_iso8601(start_dt_utc)
                    extension = ".flac" if config.USE_FLAC_COMPRESSION else ".wav"
                    audio_filename = f"{DEVICE_NAME}_{timestamp_str}{extension}"
                else:
                    extension = ".flac" if config.USE_FLAC_COMPRESSION else ".wav"
                    audio_filename = f"{DEVICE_NAME}_unknown_time{extension}"

            final_geojson_filepath = os.path.join(
                config.GEOJSON_SAVE_PATH, f"{DEVICE_NAME}.geojson"
            )
            temp_geojson_filepath = os.path.join(
                config.TEMP_PATH, f"{DEVICE_NAME}.geojson"
            )

            # Calculate actual audio duration
            if audio_buffer is not None:
                if hasattr(audio_buffer, "seek"):
                    audio_buffer.seek(0)
                    data, _ = sf.read(audio_buffer)
                    actual_duration = len(data) / recorded_samplerate
                    audio_buffer.seek(0)
                else:
                    actual_duration = len(audio_buffer) / recorded_samplerate
            elif temp_audio_filepath:
                info = sf.info(temp_audio_filepath)
                actual_duration = info.duration
            else:
                actual_duration = config.RECORDING_DURATION

            print(f"Audio duration: {actual_duration:.2f} seconds")

            audio_saved_to_disk = bool(audio_filepath) or config.SAVE_AUDIO_TO_DISK

            # Generate GeoJSON
            temp_geojson_filepath = generate_geojson(
                model_results,
                recording_time_unix_ns,
                actual_duration,
                lat,
                lon,
                config.TEMP_PATH,
                DEVICE_NAME,
                reboot_segment,
                None,
                audio_filename,
                fix_status,
                accuracy_m,
                timing_info=timing_info,
                recording_mode="test" if config.TEST_MODE else "live",
                test_audio_source=config.TEST_AUDIO_FILE if config.TEST_MODE else None,
                existing_geojson_path=final_geojson_filepath,
                audio_saved_to_disk=audio_saved_to_disk,
                gps_source=gps_source,
            )

            # Atomically move the new GeoJSON into place
            os.replace(temp_geojson_filepath, final_geojson_filepath)
            print(f"Results saved to GeoJSON")

            # Clean up temp file
            if os.path.exists(temp_geojson_filepath):
                os.remove(temp_geojson_filepath)

            # Handle audio file saving
            if config.SAVE_AUDIO_TO_DISK:
                if temp_audio_filepath:
                    final_audio_filepath = os.path.join(
                        config.AUDIO_SAVE_PATH, os.path.basename(temp_audio_filepath)
                    )
                    os.replace(temp_audio_filepath, final_audio_filepath)
                    print(f"Audio moved to final location: {final_audio_filepath}")
                elif audio_buffer is not None:
                    from datetime import timezone
                    from recorder_record_audio import safe_iso8601

                    start_dt_utc = datetime.fromtimestamp(
                        recording_time_unix_ns / 1e9, tz=timezone.utc
                    )
                    timestamp_str = safe_iso8601(start_dt_utc)

                    if config.USE_FLAC_COMPRESSION:
                        extension = ".flac"
                        audio_format = "FLAC"
                        subtype = None
                    else:
                        extension = ".wav"
                        audio_format = "WAV"
                        subtype = "PCM_16"

                    final_audio_filename = f"{DEVICE_NAME}_{timestamp_str}{extension}"
                    final_audio_filepath = os.path.join(
                        config.AUDIO_SAVE_PATH, final_audio_filename
                    )

                    os.makedirs(config.AUDIO_SAVE_PATH, exist_ok=True)

                    if hasattr(audio_buffer, "seek"):
                        audio_buffer.seek(0)
                        data, samplerate = sf.read(audio_buffer)
                    else:
                        data = audio_buffer
                        samplerate = recorded_samplerate

                    if config.USE_FLAC_COMPRESSION:
                        sf.write(
                            final_audio_filepath, data, samplerate, format=audio_format
                        )
                        print(f"Audio saved as FLAC: {final_audio_filepath}")
                    else:
                        sf.write(
                            final_audio_filepath,
                            data,
                            samplerate,
                            format=audio_format,
                            subtype=subtype,
                        )
                        print(f"Audio saved as WAV: {final_audio_filepath}")
                else:
                    print("Warning: SAVE_AUDIO_TO_DISK=True but no audio to save")
            else:
                if temp_audio_filepath and os.path.exists(temp_audio_filepath):
                    os.remove(temp_audio_filepath)
                    print("Analysis complete (RAM only - audio not saved)")
                else:
                    print("Analysis complete (RAM only)")

            # Print analysis performance summary
            print_analysis_performance(model_results)
        else:
            print("\n" + "=" * 80)
            print("RECORDING FAILED - NO AUDIO DATA")
            print("=" * 80)
            print("\nThe recording did not produce any audio data.")
            print("Please check the error messages above for details.")
            print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
