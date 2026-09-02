import os
import json
from datetime import datetime


def generate_geojson(
    model_results,  # List of model results instead of detected_species
    recording_time_unix_ns,
    duration,
    lat,
    lon,
    save_path,
    device_name,
    reboot_segment,
    model_file,  # Now optional, handled per model
    audio_filename,
    fix_status=None,
    accuracy_m=None,
    timing_info: dict | None = None,
    recording_mode="live",  # "live" or "test"
    test_audio_source=None,  # Source file name when in test mode
    existing_geojson_path=None,  # Path to existing GeoJSON to append to
    audio_saved_to_disk=True,  # Whether audio file was actually saved to disk
    gps_source="unknown",  # GPS source: "gps", "default", "cached", "error_fallback"
):
    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)

    filename = f"{device_name}.geojson"
    file_path = os.path.join(save_path, filename)

    # Load existing GeoJSON or initialize new structure
    if existing_geojson_path and os.path.exists(existing_geojson_path):
        try:
            with open(existing_geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "features" not in data:
                raise ValueError("Invalid GeoJSON structure, creating new.")
        except (json.JSONDecodeError, ValueError):
            data = {"type": "FeatureCollection", "features": []}
    elif os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "features" not in data:
                raise ValueError("Invalid GeoJSON structure, creating new.")
        except (json.JSONDecodeError, ValueError):
            data = {"type": "FeatureCollection", "features": []}
    else:
        data = {"type": "FeatureCollection", "features": []}

    # Use recording start time for timestamp (not GeoJSON creation time)
    timestamp_str = datetime.fromtimestamp(recording_time_unix_ns / 1e9).isoformat(timespec='seconds') + "Z"

    # Optional richer timing from timing_info
    start_utc = timing_info.get("start_utc") if timing_info else None
    end_utc = timing_info.get("end_utc") if timing_info else None
    wall_duration_s = timing_info.get("wall_duration_s") if timing_info else None
    nominal_duration_s = timing_info.get("nominal_duration_s") if timing_info else None
    delta_ms = timing_info.get("delta_ms") if timing_info else None

    feature = {
        "type": "Feature",
        "properties": {
            "device": device_name,
            "sampling_event": reboot_segment,
            "timestamp": timestamp_str,
            "timestamp_ns": recording_time_unix_ns,
            "duration": duration,
            "recording_mode": recording_mode,  # "live" or "test"
            "audio_saved_to_disk": audio_saved_to_disk,  # Whether audio file exists on disk
            "test_audio_source": test_audio_source if recording_mode == "test" else None,
            # Rich timing diagnostics
            "recording_start_utc": start_utc,
            "recording_end_utc": end_utc,
            "wall_duration_s": wall_duration_s,
            "nominal_duration_s": nominal_duration_s,
            "timing_delta_ms": delta_ms,
            "audio_filename": audio_filename,
            "gps_fix_status": fix_status,
            "gps_accuracy": float(accuracy_m) if accuracy_m is not None else None,
            "gps_source": gps_source,  # Source of GPS data: "gps", "default", "cached", "error_fallback"
            "models": [],  # New structure for multi-model results
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }

    # Add model results
    if model_results:
        for model_result in model_results:
            # Add detections for this model
            model_entry = {
                "model_name": model_result["model_name"],
                "model_region": model_result["model_region"],
                "model_analysis_type": model_result["model_analysis_type"],
                "model_description": model_result["model_description"],
                "model_file": model_result["model_file"],
                "model_version": model_result["model_version"],
                "threshold": model_result["threshold"],
                "input_sample_rate_hz": model_result["input_sample_rate_hz"],
                "processed_sample_rate_hz": model_result["processed_sample_rate_hz"],
                "detections": []
            }
            
            # Add detections for this model
            for detection in model_result["detections"]:
                detection_entry = {
                    "species_class": detection["species_class"],
                    "species_name": detection["species_name"],
                    "scientific_name": detection["scientific_name"],
                    "classification_score": detection["probability"],
                    "offset_s": detection["timestamp_offset"],
                }
                model_entry["detections"].append(detection_entry)
            
            feature["properties"]["models"].append(model_entry)

    data["features"].append(feature)

    # Write updated GeoJSON data to disk
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path
