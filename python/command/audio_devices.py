"""
Audio Device Management for BirdPipe

Provides functionality to list, test, and manage audio recording devices.
"""

import contextlib
import json
import os
import re
import sys
import warnings

import click

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


def format_sample_rate(rate):
    """Format sample rate in a human-readable way."""
    if rate >= 1000:
        return f"{int(rate/1000)}kHz"
    return f"{int(rate)}Hz"


def get_audio_devices():
    """
    Get all audio recording devices available on the system.
    
    Returns:
        List of recording device dictionaries.
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        recording_devices = [device for device in devices if device['max_input_channels'] > 0]
        return recording_devices
    except Exception as e:
        click.echo(f"Error querying audio devices: {e}", err=True)
        click.echo("Make sure sounddevice and portaudio are properly installed.", err=True)
        return []


def find_device_by_name(device_name):
    """
    Find a specific device by name (partial match).
    
    Args:
        device_name: Name or partial name of the device to find
        
    Returns:
        Device info dict if found, None otherwise
    """
    import sounddevice as sd
    devices = sd.query_devices()
    device_name_lower = device_name.lower()
    
    for device in devices:
        if device['max_input_channels'] > 0:  # Only recording devices
            if device_name_lower in device['name'].lower():
                return device
    
    return None


def get_supported_sample_rates(device_index):
    """
    Get supported sample rates for a device.
    
    Args:
        device_index: Device index to check.
        
    Returns:
        List of tuples (rate, rate_str, purpose) for supported rates.
    """
    import sounddevice as sd
    
    rate_tests = [
        (48000, "48kHz", "birds/audio"),
        (192000, "192kHz", "ultrasonic"),
        (384000, "384kHz", "full ultrasonic"),
    ]
    
    supported = []
    for rate, rate_str, purpose in rate_tests:
        try:
            with suppress_stderr():
                sd.check_input_settings(
                    device=device_index,
                    channels=1,
                    samplerate=rate
                )
            supported.append((rate, rate_str, purpose))
        except:
            pass
    
    return supported


def list_audio_devices_cli(json_output=False, verbose=False):
    """
    List audio devices for CLI output.
    
    Args:
        json_output: If True, output in JSON format.
        verbose: If True, show detailed information.
    """
    devices = get_audio_devices()
    
    if not devices:
        click.echo("No audio recording devices found.")
        return
    
    if json_output:
        output = []
        for device in devices:
            device_info = {
                "index": device['index'],
                "name": device['name'],
                "channels": device['max_input_channels'],
                "default_samplerate": device['default_samplerate'],
            }
            if verbose:
                supported = get_supported_sample_rates(device['index'])
                device_info["supported_rates"] = [
                    {"rate": r[0], "label": r[1], "purpose": r[2]}
                    for r in supported
                ]
            output.append(device_info)
        click.echo(json.dumps(output, indent=2))
        return
    
    # Human-readable output
    click.echo()
    click.echo("=" * 80)
    click.echo(f"  AVAILABLE MICROPHONES ({len(devices)} found)")
    click.echo("=" * 80)
    
    for device in devices:
        device_name = device['name']
        device_index = device['index']
        
        # Split name into main part and details
        if ':' in device_name:
            name_parts = device_name.split(':', 1)
            main_name = name_parts[0].strip()
            
            # Extract base name - remove patterns like "16bit r0", "24bit", etc.
            base_name = re.sub(r'\s+\d+bit.*$', '', main_name).strip()
            if not base_name:
                base_name = main_name
            
            click.echo(f"\n[{device_index}] {base_name}")
            click.echo(f"    System name: {device_name}")
        else:
            click.echo(f"\n[{device_index}] {device_name}")
        
        # Basic info
        channels = "mono" if device['max_input_channels'] == 1 else f"{device['max_input_channels']} channels"
        click.echo(f"    {channels} | Default: {format_sample_rate(device['default_samplerate'])}")
        
        if verbose:
            # Check supported sample rates
            supported = get_supported_sample_rates(device_index)
            if supported:
                rate_strs = [f"{r[1]} ({r[2]})" for r in supported]
                click.echo(f"    Supports: {', '.join(rate_strs)}")
            else:
                click.echo(f"    [Note] Use default rate: {format_sample_rate(device['default_samplerate'])}")
    
    # Show current configuration if available
    try:
        from .config import config_exists, get_config
        
        if config_exists():
            config = get_config()
            click.echo(f"\n{'-'*80}")
            click.echo("CURRENT CONFIGURATION")
            click.echo(f"{'-'*80}")
            
            device = find_device_by_name(config.MICROPHONE_NAME)
            
            if device:
                click.echo(f"\n  microphone_name = \"{config.MICROPHONE_NAME}\"")
                click.echo(f"  sample_rate = {config.RECORDING_SAMPLE_RATE}")
                click.echo(f"  Status: [OK] Matches device [{device['index']}]")
            else:
                click.echo(f"\n  microphone_name = \"{config.MICROPHONE_NAME}\"")
                click.echo(f"  Status: [WARNING] No matching device found")
                click.echo(f"  Tip: Use a device name from the list above")
            
            click.echo(f"\n  Note: Partial names work (e.g., 'UltraMic' matches full device name)")
    except ImportError:
        pass
    except Exception as e:
        click.echo(f"\n  Configuration error: {e}")
    
    click.echo("=" * 80 + "\n")


def test_microphone_cli(microphone_name=None, duration=2.0, sample_rate=48000):
    """
    Test a microphone by recording a short audio sample.
    
    Args:
        microphone_name: Name or partial name of microphone to test.
        duration: Recording duration in seconds.
        sample_rate: Sample rate to use for test.
    """
    import sounddevice as sd
    import numpy as np
    
    # Get microphone name from config if not specified
    if microphone_name is None:
        try:
            from .config import config_exists, get_config
            if config_exists():
                config = get_config()
                microphone_name = config.MICROPHONE_NAME
        except ImportError:
            pass
    
    if microphone_name is None:
        microphone_name = "default"
    
    click.echo(f"Testing microphone: {microphone_name}")
    click.echo(f"Sample rate: {format_sample_rate(sample_rate)}")
    click.echo(f"Duration: {duration}s")
    click.echo()
    
    # Find the device
    if microphone_name.lower() == "default":
        device_index = None
        click.echo("Using default input device")
    else:
        device = find_device_by_name(microphone_name)
        if device is None:
            click.echo(f"Error: No device matching '{microphone_name}' found", err=True)
            click.echo("Run 'birdpipe list-devices' to see available devices.", err=True)
            return
        device_index = device['index']
        click.echo(f"Found device: [{device_index}] {device['name']}")
    
    click.echo()
    click.echo("Recording...", nl=False)
    
    try:
        # Record audio (don't suppress stderr here as it interferes with sounddevice)
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_index,
            dtype='float32'
        )
        sd.wait()
        
        click.echo(" Done!")
        click.echo()
        
        # Analyze the recording
        audio = recording.flatten()
        
        # Calculate statistics
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        # Convert to dB
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        click.echo("Recording statistics:")
        click.echo(f"  Duration: {len(audio) / sample_rate:.2f}s")
        click.echo(f"  Samples: {len(audio)}")
        click.echo(f"  Peak level: {peak_db:.1f} dB")
        click.echo(f"  RMS level: {rms_db:.1f} dB")
        
        if peak_db < -60:
            click.echo()
            click.echo("⚠️  Warning: Very low audio level detected.")
            click.echo("   Check that your microphone is connected and working.")
        elif peak_db > -3:
            click.echo()
            click.echo("⚠️  Warning: Audio may be clipping (very high level).")
            click.echo("   Consider reducing microphone gain.")
        else:
            click.echo()
            click.echo("✓ Microphone is working correctly!")
            
    except Exception as e:
        click.echo(f" Error!")
        click.echo(f"Failed to record: {e}", err=True)
        click.echo()
        click.echo("Troubleshooting tips:", err=True)
        click.echo("  - Check that the microphone is connected", err=True)
        click.echo("  - Try a different sample rate with -r/--sample-rate", err=True)
        click.echo("  - Run 'birdpipe list-devices -v' to see supported sample rates", err=True)
