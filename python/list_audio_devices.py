"""
Audio Device Lister
Lists all available audio recording devices on the system.
Helps identify the correct device name for MICROPHONE_NAME in recorder_config.py
"""

import contextlib
import os
import re
import sys
import warnings

import sounddevice as sd

# Suppress ALSA warnings and errors
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr output temporarily (including C-level errors)."""
    try:
        # Save original stderr file descriptor
        stderr_fd = sys.stderr.fileno()
        with open(os.devnull, 'w') as devnull:
            old_stderr = os.dup(stderr_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                # Restore stderr
                os.dup2(old_stderr, stderr_fd)
                os.close(old_stderr)
    except:
        # Fallback if file descriptor manipulation fails
        yield

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def format_sample_rate(rate):
    """Format sample rate in a human-readable way."""
    if rate >= 1000:
        return f"{int(rate/1000)}kHz"
    return f"{int(rate)}Hz"

def list_audio_devices():
    """
    List all audio recording devices available on the system.
    Returns list of recording devices.
    """
    try:
        devices = sd.query_devices()
        recording_devices = [device for device in devices if device['max_input_channels'] > 0]
        return recording_devices
    except Exception as e:
        print(f"\n[ERROR] Error querying audio devices: {e}")
        print("\n[TIP] Make sure sounddevice and portaudio are properly installed.")
        print("   Try: uv pip install sounddevice")
        sys.exit(1)

def find_device_by_name(device_name):
    """
    Find a specific device by name (partial match).
    
    Args:
        device_name: Name or partial name of the device to find
        
    Returns:
        Device info dict if found, None otherwise
    """
    devices = sd.query_devices()
    device_name_lower = device_name.lower()
    
    for device in devices:
        if device['max_input_channels'] > 0:  # Only recording devices
            if device_name_lower in device['name'].lower():
                return device
    
    return None

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  MESHBIRD AUDIO DEVICE CHECKER")
    print("=" * 80)
    
    # List all devices
    devices = list_audio_devices()
    
    # Update header with total count
    print_header(f"AVAILABLE MICROPHONES ({len(devices)} found)")
    
    for device in devices:
        device_name = device['name']
        device_index = device['index']
        
        # Split name into main part and details
        if ':' in device_name:
            name_parts = device_name.split(':', 1)
            main_name = name_parts[0].strip()
            
            # Extract base name - remove patterns like "16bit r0", "24bit", etc.
            base_name = re.sub(r'\s+\d+bit.*$', '', main_name).strip()
            if not base_name:  # If nothing left, use full main_name
                base_name = main_name
            
            print(f"\n[{device_index}] {base_name}")
            print(f"    System name: {device_name}")
        else:
            print(f"\n[{device_index}] {device_name}")
        
        # Basic info on one line
        channels = "mono" if device['max_input_channels'] == 1 else f"{device['max_input_channels']} channels"
        print(f"    {channels} | Default: {format_sample_rate(device['default_samplerate'])}")
        
        # Check supported sample rates (only key ones)
        rate_tests = [
            (48000, "48kHz", "birds/audio"),
            (192000, "192kHz", "ultrasonic"),
            (384000, "384kHz", "full ultrasonic"),
        ]
        
        supported_rates = []
        for rate, rate_str, purpose in rate_tests:
            try:
                with suppress_stderr():
                    sd.check_input_settings(
                        device=device_index,
                        channels=1,
                        samplerate=rate
                    )
                supported_rates.append(f"{rate_str} ({purpose})")
            except:
                pass
        
        if supported_rates:
            print(f"    Supports: {', '.join(supported_rates)}")
        else:
            print(f"    [Note] Use default rate: {format_sample_rate(device['default_samplerate'])}")
    
    # Check if the configured microphone exists
    if devices:
        print(f"\n{'-'*80}")
        print("CURRENT CONFIGURATION")
        print(f"{'-'*80}")
        
        try:
            from recorder_config import MICROPHONE_NAME, RECORDING_SAMPLE_RATE
            
            device = find_device_by_name(MICROPHONE_NAME)
            
            if device:
                print(f"\n  MICROPHONE_NAME = \"{MICROPHONE_NAME}\"")
                print(f"  Sample rate: {format_sample_rate(RECORDING_SAMPLE_RATE)}")
                print(f"  Status: [OK] Matches device [{device['index']}]")
            else:
                print(f"\n  MICROPHONE_NAME = \"{MICROPHONE_NAME}\"")
                print(f"  Status: [ERROR] No matching device found")
                print(f"  Tip: Use a device name from the list above")
                
        except ImportError:
            print("\n  recorder_config.py not found")
        except Exception as e:
            print(f"\n  Error: {e}")
        
        print(f"\n  Note: Partial names work (e.g., 'UltraMic' matches full device name)")
        print(f"  To change: Edit MICROPHONE_NAME in recorder_config.py")
    
    print("=" * 80 + "\n")
