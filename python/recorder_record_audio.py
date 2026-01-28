import io
import os
import time
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import soundfile as sf
# scipy imports moved to tf_model_bird_classifier.py
from recorder_config import SHORT_AUDIO_FILENAME


def find_microphone(device_name: str) -> int:
    """Return the device index of the first microphone that matches the given name (case-insensitive)."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in device["name"].lower():
                return i
        return None
    except Exception:
        return None


def validate_microphone_settings(device_index: int, sample_rate: int, device_name: str) -> tuple[bool, str]:
    """
    Validate that the microphone supports the requested sample rate.
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        devices = sd.query_devices()
        device = devices[device_index]
        
        # Check if device supports the requested sample rate
        # Try with device's actual number of channels
        max_channels = device['max_input_channels']
        try:
            sd.check_input_settings(
                device=device_index,
                channels=max_channels,
                samplerate=sample_rate
            )
            return True, ""
        except Exception as e:
            # Build helpful error message
            error_msg = f"\n{'='*80}\n"
            error_msg += "MICROPHONE CONFIGURATION ERROR\n"
            error_msg += f"{'='*80}\n\n"
            error_msg += f"Microphone: {device['name']}\n"
            error_msg += f"Configuration requests: {sample_rate} Hz ({sample_rate/1000:.0f}kHz)\n"
            error_msg += f"Microphone default: {device['default_samplerate']} Hz ({device['default_samplerate']/1000:.0f}kHz)\n\n"
            
            # Provide specific guidance based on sample rate mismatch
            if sample_rate >= 192000:
                error_msg += "PROBLEM: Configuration requests ultrasonic sample rate,\n"
                error_msg += "         but this microphone doesn't support it.\n\n"
                error_msg += "SOLUTIONS:\n"
                error_msg += "1. For bat/ultrasonic detection:\n"
                error_msg += "   - Use an ultrasonic microphone (UltraMic 192K or 384K)\n"
                error_msg += "   - Keep RECORDING_SAMPLE_RATE at 384000 or 192000\n\n"
                error_msg += "2. For bird detection with this microphone:\n"
                error_msg += f"   - Change RECORDING_SAMPLE_RATE in recorder_config.py\n"
                error_msg += f"   - Set it to {int(device['default_samplerate'])} Hz (microphone's default)\n"
                error_msg += "   - Bird models work at standard sample rates\n\n"
            elif sample_rate != device['default_samplerate']:
                error_msg += f"PROBLEM: Configuration requests {sample_rate} Hz,\n"
                error_msg += f"         but microphone expects {int(device['default_samplerate'])} Hz.\n\n"
                error_msg += "SOLUTION: Change RECORDING_SAMPLE_RATE in recorder_config.py\n"
                error_msg += f"          to {int(device['default_samplerate'])} Hz (microphone's default)\n\n"
            else:
                # Sample rates match but still failed - unusual case
                error_msg += f"PROBLEM: Validation failed even though sample rates match.\n"
                error_msg += f"         This may be a microphone or driver issue.\n"
                error_msg += f"         Technical error: {str(e)}\n\n"
                error_msg += "SOLUTIONS:\n"
                error_msg += "- Check microphone connection and permissions\n"
                error_msg += "- Try unplugging and reconnecting the microphone\n"
                error_msg += "- Check if another application is using the microphone\n\n"
            
            error_msg += "TIP: Run 'uv run list_audio_devices.py' to see all available devices\n"
            error_msg += f"{'='*80}\n"
            
            return False, error_msg
            
    except Exception as e:
        return False, f"Error validating microphone settings: {e}"


# lowpass_filter moved to tf_model_bird_classifier.py as part of bird-specific preprocessing


def safe_iso8601(dt: datetime) -> str:
    """Format datetime in a safe ISO8601-like string for filenames (colons replaced)."""
    # Example: 2025-09-05T20_21_30.123456Z
    return dt.strftime("%Y-%m-%dT%H_%M_%S.%fZ")




def record_audio(
    device_index: int,
    duration: int = 60,
    save_to_disk=False,
    save_path="./recorder",
    device_name="device",
    sample_rate=48000,
    lat=None,
    lon=None,
    fix_status=None,
    accuracy_m=None,
    reboot_segment=None,
    model_file=None,
    use_flac_compression=False,
    flac_compression_level=5,
):
    """
    Records audio with the given parameters.

    Args:
        device_index: Index of the input device.
        duration: Recording length in seconds.
        save_to_disk: If True, saves to WAV or FLAC file on disk. Otherwise, returns BytesIO buffer.
        save_path: Folder to save the file if save_to_disk=True.
        device_name: Name to include in the filename.
        sample_rate: Sample rate in Hz (e.g., 48000, 192000, 384000).
        lat, lon: Optional GPS coordinates.
        fix_status: GPS fix quality.
        accuracy_m: Accuracy in meters.
        reboot_segment: Identifier for segmenting after reboot.
        model_file: Optional model filename for metadata.
        use_flac_compression: If True, saves as FLAC. If False, saves as WAV.
        flac_compression_level: FLAC compression level (0-8, default 5).

    Returns:
        - audio_buffer (BytesIO) OR None if saved to disk
        - timing_info (dict): timing & duration details
        - audio_filepath (str) OR None if in memory
        - samplerate (int): actual sample rate of recorded audio
    """
    if device_index is None:
        return None, None, None, None

    # Validate microphone settings before attempting to record
    is_valid, error_message = validate_microphone_settings(device_index, sample_rate, device_name)
    if not is_valid:
        print(error_message)
        return None, None, None, None

    samplerate = sample_rate

    try:
        # 1) Prepare all metadata strings (does not affect timestamp)
        gps_str = f"{lat:.5f}_{lon:.5f}" if (lat is not None and lon is not None) else "nogps"
        fix_str = f"_fix{fix_status}d" if fix_status is not None else "_fixNA"
        acc_str = f"_{accuracy_m}m" if accuracy_m is not None else "_accNA"
        segment_str = reboot_segment.replace(":", "").replace(".", "").replace("Z", "") if reboot_segment else "unknown"
        segment_part = f"seg{segment_str}"
        dur_str = f"dur{duration}s"
        mode_str = f"sr{sample_rate}"
        model_name = os.path.splitext(os.path.basename(model_file))[0] if model_file else "unknown"
        model_str = f"_model{model_name}"

        # 2) Start recording immediately after timestamp preparation
        frames = int(duration * samplerate)

        # 3) Capture the start timestamp immediately before recording
        recording_time_unix_ns = time.time_ns()

        print("Starting recording immediately...")

        audio_data = sd.rec(
            frames,
            samplerate=samplerate,
            channels=1,
            dtype=np.int16,
            device=device_index,
        )
        sd.wait()  # block until finished

        # 5) Capture end timestamp immediately after recording stopped
        recording_end_time_unix_ns = time.time_ns()

        # 6) Build filename using the start timestamp
        start_dt_utc = datetime.fromtimestamp(recording_time_unix_ns / 1e9, tz=timezone.utc)
        timestamp_str = safe_iso8601(start_dt_utc)
        file_extension = ".flac" if use_flac_compression else ".wav"
        if SHORT_AUDIO_FILENAME:
            filename = f"{device_name}_{timestamp_str}{file_extension}"
        else:
            recording_iso = timestamp_str.replace(":", "").replace(".", "").replace("Z", "Z")
            recording_part = f"rt{recording_iso}"
            filename = (
                f"{device_name}_{segment_part}_{recording_part}_"
                f"{gps_str}{fix_str}{acc_str}_{dur_str}_{mode_str}{model_str}{file_extension}"
            )

        # 7) Audio processing moved to classifiers - return raw audio at original sample rate
        # Each classifier handles its own preprocessing requirements

        # 8) Save to disk or memory
        if save_to_disk:
            os.makedirs(save_path, exist_ok=True)
            audio_filepath = os.path.join(save_path, filename)
            if use_flac_compression:
                sf.write(audio_filepath, audio_data, samplerate, format="FLAC", subtype=None)
            else:
                sf.write(audio_filepath, audio_data, samplerate, format="WAV", subtype="PCM_16")
            audio_buffer = None
        else:
            audio_filepath = None
            audio_buffer = io.BytesIO()
            if use_flac_compression:
                sf.write(audio_buffer, audio_data, samplerate, format="FLAC", subtype=None)
            else:
                sf.write(audio_buffer, audio_data, samplerate, format="WAV", subtype="PCM_16")
            audio_buffer.seek(0)

        # 9) Calculate timing info for metadata (keep calculations but remove verbose output)
        end_dt_utc = datetime.fromtimestamp(recording_end_time_unix_ns / 1e9, tz=timezone.utc)
        end_calc_ns = recording_time_unix_ns + int(duration * 1e9)

        wall_duration_ns = recording_end_time_unix_ns - recording_time_unix_ns
        wall_duration_s = wall_duration_ns / 1e9
        calc_duration_s = duration
        delta_ns = recording_end_time_unix_ns - end_calc_ns
        delta_ms = delta_ns / 1e6

        # 10) Bundle timing/duration info
        timing_info = {
            "start_unix_ns": int(recording_time_unix_ns),
            "end_unix_ns": int(recording_end_time_unix_ns),
            "start_utc": safe_iso8601(start_dt_utc),
            "end_utc": safe_iso8601(end_dt_utc),
            "wall_duration_s": float(wall_duration_s),
            "nominal_duration_s": float(calc_duration_s),
            "delta_ms": float(delta_ms),
        }

        # 11) Return values including sample rate
        return audio_buffer, timing_info, audio_filepath, samplerate

    except Exception as e:
        error_msg = f"\n{'='*80}\n"
        error_msg += "AUDIO RECORDING ERROR\n"
        error_msg += f"{'='*80}\n\n"
        error_msg += f"Failed to record audio: {e}\n\n"
        
        # Check if it's a sample rate related error
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['sample rate', 'samplerate', 'rate', 'invalid', 'not supported']):
            error_msg += "This looks like a sample rate configuration issue.\n\n"
            error_msg += f"Current setting: {sample_rate} Hz ({sample_rate/1000:.0f}kHz)\n\n"
            error_msg += "SOLUTIONS:\n"
            if sample_rate >= 192000:
                error_msg += "1. For ultrasonic/bat recording:\n"
                error_msg += "   - Use an ultrasonic microphone (UltraMic 192K or 384K)\n\n"
                error_msg += "2. For bird recording:\n"
                error_msg += "   - Change RECORDING_SAMPLE_RATE to 48000 in recorder_config.py\n\n"
            else:
                error_msg += "- Change RECORDING_SAMPLE_RATE in recorder_config.py\n"
                error_msg += "- Run 'uv run list_audio_devices.py' to see supported rates\n\n"
        
        error_msg += f"{'='*80}\n"
        print(error_msg)
        return None, None, None, None
