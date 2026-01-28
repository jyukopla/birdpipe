"""
BirdPipe Audio Recorder

Multi-modal audio analysis system for bird and bat species detection.

Features:
- Simultaneous bird and bat species classification
- GPS-tagged detections with high precision timing
- Multi-sample rate support (48kHz birds, 384kHz bats)
- GeoJSON output for analysis and visualization
- Live recording and test mode capabilities

Supported Analysis Types:
- Birds: BirdNET-based classification at 48kHz
- Bats: Ultrasonic echolocation analysis at 384kHz

Configure analysis models in recorder_config.py using MODELS_TO_RUN.
"""

import os
import sys
import socket
import time
import psutil
import soundfile as sf
import warnings
from datetime import datetime, timedelta

from recorder_config import (
    AUDIO_SAVE_PATH,
    SAVE_AUDIO_TO_DISK,
    USE_FLAC_COMPRESSION,
    FLAC_COMPRESSION_LEVEL,
    GEOJSON_SAVE_PATH,
    MICROPHONE_NAME,
    RECORDING_DURATION,
    RECORDING_SAMPLE_RATE,
    MODELS_TO_RUN,
    TEMP_PATH,
    TEST_MODE,
    TEST_AUDIO_PATH,
    TEST_AUDIO_FILE,
    MODEL_INFORMATION,
    SHOW_MODEL_TIMING,
    SHOW_MEMORY_USAGE,
    SHOW_SEGMENT_DETECTIONS,
)
import recorder_config as config
from recorder_analysis import (
    run_multi_model_analysis,
    print_analysis_performance,
    log_model_info,
)
from recorder_generate_geojson import generate_geojson
from recorder_get_day_of_the_year import get_day_of_year
from recorder_get_gnss_location import get_gnss_location
from recorder_get_pps_status import get_pps_status
from recorder_record_audio import find_microphone, record_audio

DEVICE_NAME = socket.gethostname()


def load_sample_audio():
    """
    Load sample audio file for testing purposes.

    Returns:
        tuple: (audio_data, sample_rate) or (None, None) if file not found
    """
    sample_file_path = os.path.join(TEST_AUDIO_PATH, TEST_AUDIO_FILE)

    if not os.path.exists(sample_file_path):
        print(f"Warning: Sample audio file not found at {sample_file_path}")
        return None, None

    try:
        audio_data, sample_rate = sf.read(sample_file_path)

        print(f"Loaded sample audio: {TEST_AUDIO_FILE}")
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
        return boot_time.isoformat() + "Z"
    except Exception:
        return "unknown"


def main():
    try:
        # Suppress NumPy warnings in production mode only (keep them visible in test mode)
        if not TEST_MODE:
            warnings.filterwarnings(
                "ignore", message="The value of the smallest subnormal"
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="numpy.core.getlimits"
            )

        # Show current operation mode
        if TEST_MODE:
            print("=== MESHBIRD RECORDER - TEST MODE ===")
            print(f"Using sample audio: {TEST_AUDIO_FILE}")
        else:
            print("=== MESHBIRD RECORDER - PRODUCTION MODE ===")
            print(f"Microphone: {MICROPHONE_NAME}")
            print(f"Sample rate: {RECORDING_SAMPLE_RATE} Hz")
            print(f"Duration: {RECORDING_DURATION} seconds")
            save_format = (
                f"FLAC (level {FLAC_COMPRESSION_LEVEL})"
                if USE_FLAC_COMPRESSION
                else "WAV"
            )
            print(
                f"Save audio to disk: {'Yes' if SAVE_AUDIO_TO_DISK else 'No (RAM only)'} ({save_format})"
            )
        print()

        # Ensure output directories exist
        os.makedirs(GEOJSON_SAVE_PATH, exist_ok=True)
        os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)
        os.makedirs(TEMP_PATH, exist_ok=True)

        # PPS status (hardware GNSS timing) - only check if GPS is required
        if config.GPS_REQUIRED:
            if get_pps_status():
                print("GNSS PPS Signal OK")
            else:
                print("GNSS PPS Signal not detected. Continuing without PPS.")

        # Day-of-year is used by the analysis model
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
                print(
                    f"Latitude: {lat:.9f}, Longitude: {lon:.9f}, GPS fix: {fix_status}, Accuracy: {accuracy_desc}"
                )
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
        if not TEST_MODE:
            mic_index = find_microphone(MICROPHONE_NAME)
            if mic_index is None:
                print(f"\n{'=' * 80}")
                print("MICROPHONE NOT FOUND")
                print(f"{'=' * 80}")
                print(f"\nCould not find microphone: '{MICROPHONE_NAME}'")
                print(f"\nSOLUTION:")
                print(f"1. Check microphone is connected")
                print(f"2. Run 'uv run list_audio_devices.py' to see available devices")
                print(f"3. Update MICROPHONE_NAME in recorder_config.py")
                print(f"{'=' * 80}\n")
                return

            # Note: Validation disabled due to ALSA/PortAudio false negatives
            # The actual recording works even if check_input_settings fails
            # Validation moved to actual recording attempt
            print(f"Microphone found at index {mic_index}")

        reboot_segment = get_reboot_segment()

        # Ready to record immediately - scheduling handled by systemctl

        # Perform the recording or load sample audio
        if TEST_MODE:
            print("=== TEST MODE ENABLED ===")
            print(f"Loading sample audio instead of recording...")
            audio_buffer, recorded_samplerate = load_sample_audio()
            if audio_buffer is None:
                print("Failed to load sample audio. Exiting.")
                return

            # Create mock timing info for test mode
            from time import time_ns
            from datetime import datetime as dt

            current_time = time_ns()
            timing_info = {
                "start_utc": dt.fromtimestamp(current_time / 1e9).isoformat() + "Z",
                "end_utc": dt.fromtimestamp(
                    (current_time + len(audio_buffer) / recorded_samplerate * 1e9) / 1e9
                ).isoformat()
                + "Z",
                "wall_duration_s": len(audio_buffer) / recorded_samplerate,
                "nominal_duration_s": len(audio_buffer) / recorded_samplerate,
                "delta_ms": 0.0,
                "start_unix_ns": current_time,  # Add this for compatibility
            }
            audio_filepath = None  # No file saved in test mode
            temp_audio_filepath = None
        else:
            # Perform live recording immediately
            audio_buffer, timing_info, audio_filepath, recorded_samplerate = (
                record_audio(
                    mic_index,
                    duration=RECORDING_DURATION,
                    save_to_disk=SAVE_AUDIO_TO_DISK,
                    save_path=AUDIO_SAVE_PATH,
                    device_name=DEVICE_NAME,
                    sample_rate=RECORDING_SAMPLE_RATE,
                    lat=lat,
                    lon=lon,
                    fix_status=fix_status,
                    accuracy_m=accuracy_m,
                    reboot_segment=reboot_segment,
                    model_file=None,  # Model info now in MODELS_TO_RUN list
                )
            )

            # Move recorded WAV into a temp area for analysis (atomic swap pattern)
            temp_audio_filepath = (
                os.path.join(TEMP_PATH, os.path.basename(audio_filepath))
                if audio_filepath
                else None
            )
            if audio_filepath:
                os.replace(audio_filepath, temp_audio_filepath)

        # Proceed only if we have audio in memory or a temp file on disk
        if (audio_buffer is not None) or temp_audio_filepath:
            print("Recording complete. Starting analysis...")

            # Unpack timing info for analysis/geojson
            if timing_info:
                recording_time_unix_ns = timing_info.get("start_unix_ns")
            else:
                recording_time_unix_ns = None

            # Run multi-model analysis
            model_results = run_multi_model_analysis(
                audio_buffer if audio_buffer is not None else temp_audio_filepath,
                recorded_samplerate,
            )

            # Generate audio filename even if not initially saved to disk
            if audio_filepath:
                audio_filename = os.path.basename(audio_filepath)
            else:
                # Generate filename based on recording timestamp (even for RAM-only recordings)
                from datetime import datetime, timezone

                if recording_time_unix_ns:
                    start_dt_utc = datetime.fromtimestamp(
                        recording_time_unix_ns / 1e9, tz=timezone.utc
                    )
                    from recorder_record_audio import safe_iso8601

                    timestamp_str = safe_iso8601(start_dt_utc)
                    extension = ".flac" if USE_FLAC_COMPRESSION else ".wav"
                    audio_filename = f"{DEVICE_NAME}_{timestamp_str}{extension}"
                else:
                    # Fallback filename
                    extension = ".flac" if USE_FLAC_COMPRESSION else ".wav"
                    audio_filename = f"{DEVICE_NAME}_unknown_time{extension}"

            final_geojson_filepath = os.path.join(
                GEOJSON_SAVE_PATH, f"{DEVICE_NAME}.geojson"
            )
            temp_geojson_filepath = os.path.join(TEMP_PATH, f"{DEVICE_NAME}.geojson")

            # Calculate actual audio duration from the audio data
            if audio_buffer is not None:
                if hasattr(audio_buffer, "seek"):
                    # audio_buffer is a file-like object, need to read it to get length
                    audio_buffer.seek(0)
                    data, _ = sf.read(audio_buffer)
                    actual_duration = len(data) / recorded_samplerate
                    audio_buffer.seek(0)  # Reset for any future operations
                else:
                    # audio_buffer is a numpy array (test mode)
                    actual_duration = len(audio_buffer) / recorded_samplerate
            elif temp_audio_filepath:
                # Audio is in file, read file to get duration
                info = sf.info(temp_audio_filepath)
                actual_duration = info.duration
            else:
                # Fallback to configured duration
                actual_duration = RECORDING_DURATION

            print(f"Audio duration: {actual_duration:.2f} seconds")

            # Determine if audio will be/was saved to disk
            audio_saved_to_disk = bool(audio_filepath) or SAVE_AUDIO_TO_DISK

            # Generate updated GeoJSON with multi-model results
            temp_geojson_filepath = generate_geojson(
                model_results,  # Pass all model results instead of detected_species
                recording_time_unix_ns,
                actual_duration,  # Use actual duration instead of RECORDING_DURATION
                lat,
                lon,
                TEMP_PATH,
                DEVICE_NAME,
                reboot_segment,
                None,  # model_file now handled per model
                audio_filename,
                fix_status,
                accuracy_m,
                timing_info=timing_info,
                recording_mode="test" if TEST_MODE else "live",
                test_audio_source=TEST_AUDIO_FILE if TEST_MODE else None,
                existing_geojson_path=final_geojson_filepath,  # Pass existing file path
                audio_saved_to_disk=audio_saved_to_disk,
                gps_source=gps_source,  # Add GPS source information
            )

            # Atomically move the new GeoJSON into place
            os.replace(temp_geojson_filepath, final_geojson_filepath)
            print(f"Results saved to GeoJSON")

            # Clean up any leftover temp file
            if os.path.exists(temp_geojson_filepath):
                os.remove(temp_geojson_filepath)

            # Handle audio file saving/moving after analysis
            if SAVE_AUDIO_TO_DISK:
                if temp_audio_filepath:
                    # Audio was recorded to disk, moved to temp for analysis, now move back to final location
                    final_audio_filepath = os.path.join(
                        AUDIO_SAVE_PATH, os.path.basename(temp_audio_filepath)
                    )
                    os.replace(temp_audio_filepath, final_audio_filepath)
                    print(f"Audio moved to final location: {final_audio_filepath}")
                elif audio_buffer is not None:
                    # Audio was recorded in RAM, save to disk now
                    from datetime import datetime, timezone

                    # Generate filename using same logic as recorder_record_audio.py
                    start_dt_utc = datetime.fromtimestamp(
                        recording_time_unix_ns / 1e9, tz=timezone.utc
                    )
                    from recorder_record_audio import safe_iso8601

                    timestamp_str = safe_iso8601(start_dt_utc)

                    # Choose file format and extension
                    if USE_FLAC_COMPRESSION:
                        extension = ".flac"
                        audio_format = "FLAC"
                        subtype = None  # FLAC doesn't use subtype
                    else:
                        extension = ".wav"
                        audio_format = "WAV"
                        subtype = "PCM_16"

                    final_audio_filename = f"{DEVICE_NAME}_{timestamp_str}{extension}"
                    final_audio_filepath = os.path.join(
                        AUDIO_SAVE_PATH, final_audio_filename
                    )

                    # Create directory and save
                    os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

                    # Handle different types of audio_buffer
                    if hasattr(audio_buffer, "seek"):
                        # audio_buffer is a file-like object (BytesIO from live recording)
                        audio_buffer.seek(0)
                        data, samplerate = sf.read(audio_buffer)
                    else:
                        # audio_buffer is a numpy array (from test mode)
                        data = audio_buffer
                        samplerate = recorded_samplerate

                    # Save with appropriate format and compression
                    if USE_FLAC_COMPRESSION:
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

                    # Update audio_filename for GeoJSON and regenerate with correct filename
                    audio_filename = final_audio_filename

                    # Regenerate GeoJSON with correct audio filename (audio definitely saved now)
                    temp_geojson_filepath = generate_geojson(
                        model_results,
                        recording_time_unix_ns,
                        actual_duration,  # Use actual duration instead of RECORDING_DURATION
                        lat,
                        lon,
                        TEMP_PATH,
                        DEVICE_NAME,
                        reboot_segment,
                        None,
                        audio_filename,
                        fix_status,
                        accuracy_m,
                        timing_info=timing_info,
                        recording_mode="test" if TEST_MODE else "live",
                        test_audio_source=TEST_AUDIO_FILE if TEST_MODE else None,
                        existing_geojson_path=final_geojson_filepath,  # Pass existing file path
                        audio_saved_to_disk=True,  # Audio was definitely saved here
                        gps_source=gps_source,  # Add GPS source information
                    )
                else:
                    print("Warning: SAVE_AUDIO_TO_DISK=True but no audio to save")
            else:
                # RAM only mode - clean up temp files, don't save audio
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
            print("This usually means there was a configuration problem.")
            print("\nPlease check the error messages above for details.")
            print("=" * 80 + "\n")

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
