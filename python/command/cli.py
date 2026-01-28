"""
BirdPipe Command Line Interface

Main entry point for the birdpipe command.
Provides subcommands for recording, listing devices, and managing configuration.
"""

import sys
from pathlib import Path

import click

from .config import (
    Config,
    CONFIG_FILE,
    DEFAULT_CONFIG,
    config_exists,
    create_default_config,
    get_config,
    get_config_path,
    load_config,
    save_config,
)


@click.group()
@click.version_option(version="0.1.0", prog_name="birdpipe")
@click.pass_context
def cli(ctx):
    """
    BirdPipe - Multi-modal audio analysis system for bird and bat species detection.
    
    Run 'birdpipe COMMAND --help' for more information on a command.
    """
    ctx.ensure_object(dict)


@cli.command("record")
@click.option("--test", "-t", is_flag=True, help="Run in test mode with sample audio")
@click.option("--duration", "-d", type=int, help="Recording duration in seconds")
@click.option("--sample-rate", "-r", type=int, help="Sample rate in Hz")
@click.option("--microphone", "-m", type=str, help="Microphone name (partial match)")
@click.pass_context
def record(ctx, test, duration, sample_rate, microphone):
    """
    Record and analyze audio for bird and bat species detection.
    
    This command captures audio from the configured microphone, runs
    species classification models, and saves results as GeoJSON.
    """
    # Import here to avoid loading heavy dependencies if not needed
    from . import recorder_main
    
    # Build config overrides from CLI options
    overrides = {}
    if test:
        overrides["test_mode"] = True
    if duration:
        overrides["duration"] = duration
    if sample_rate:
        overrides["sample_rate"] = sample_rate
    if microphone:
        overrides["microphone_name"] = microphone
    
    recorder_main.main(overrides)


@cli.command("list-devices")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed device information")
def list_devices(json_output, verbose):
    """
    List all available audio recording devices.
    
    Shows microphones and audio input devices available on the system,
    including supported sample rates for bird and bat recording.
    """
    from . import audio_devices
    
    audio_devices.list_audio_devices_cli(json_output=json_output, verbose=verbose)


@cli.command("create-config")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing configuration")
@click.option("--show", "-s", is_flag=True, help="Show the configuration after creating")
def create_config(force, show):
    """
    Create a default configuration file.
    
    Creates ~/.config/birdpipe/config.toml with default settings.
    Use --force to overwrite an existing configuration.
    """
    config_path = get_config_path()
    
    if config_exists() and not force:
        click.echo(f"Configuration already exists at: {config_path}")
        click.echo("Use --force to overwrite.")
        return
    
    if create_default_config(overwrite=force):
        click.echo(f"Configuration created at: {config_path}")
        
        if show:
            click.echo("\nConfiguration contents:")
            click.echo("-" * 40)
            with open(config_path, "r") as f:
                click.echo(f.read())
    else:
        click.echo("Failed to create configuration file.", err=True)
        click.echo("Make sure tomli-w is installed: pip install tomli-w", err=True)
        sys.exit(1)


@cli.command("show-config")
@click.option("--path", "-p", is_flag=True, help="Only show the configuration file path")
def show_config(path):
    """
    Show the current configuration.
    
    Displays the contents of the configuration file or the path to it.
    """
    config_path = get_config_path()
    
    if path:
        click.echo(config_path)
        return
    
    if not config_exists():
        click.echo(f"No configuration file found at: {config_path}")
        click.echo("Run 'birdpipe create-config' to create one.")
        return
    
    click.echo(f"Configuration file: {config_path}")
    click.echo("-" * 40)
    with open(config_path, "r") as f:
        click.echo(f.read())


@cli.command("edit-config")
def edit_config():
    """
    Open the configuration file in your default editor.
    
    Uses the EDITOR environment variable, falls back to nano/vi.
    """
    import os
    import subprocess
    
    config_path = get_config_path()
    
    if not config_exists():
        click.echo(f"No configuration file found at: {config_path}")
        if click.confirm("Create default configuration?"):
            create_default_config()
        else:
            return
    
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
    
    try:
        subprocess.run([editor, str(config_path)], check=True)
    except FileNotFoundError:
        # Try vi as fallback
        try:
            subprocess.run(["vi", str(config_path)], check=True)
        except FileNotFoundError:
            click.echo(f"No editor found. Edit the file manually: {config_path}", err=True)
            sys.exit(1)


@cli.command("validate-config")
def validate_config():
    """
    Validate the current configuration file.
    
    Checks that the configuration file is valid TOML and contains
    all required settings.
    """
    config_path = get_config_path()
    
    if not config_exists():
        click.echo(f"No configuration file found at: {config_path}")
        click.echo("Run 'birdpipe create-config' to create one.")
        sys.exit(1)
    
    try:
        config = load_config()
        
        # Check required sections
        required_sections = ["audio", "test", "debug", "geojson", "gps", "paths", "models"]
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            click.echo(f"Warning: Missing sections: {', '.join(missing_sections)}")
            click.echo("These will use default values.")
        
        # Validate audio settings
        audio = config.get("audio", {})
        sample_rate = audio.get("sample_rate", 48000)
        supported_rates = [44100, 48000, 192000, 384000]
        if sample_rate not in supported_rates:
            click.echo(f"Warning: Sample rate {sample_rate} not in supported rates: {supported_rates}")
        
        # Validate FLAC compression level
        if audio.get("use_flac", True):
            compression_level = audio.get("flac_compression_level", 5)
            if not 0 <= compression_level <= 8:
                click.echo(f"Error: FLAC compression level must be 0-8, got {compression_level}")
                sys.exit(1)
        
        # Validate models
        models = config.get("models", [])
        if not models:
            click.echo("Warning: No models configured")
        
        active_models = [m for m in models if m.get("run", False)]
        if not active_models:
            click.echo("Warning: No models are enabled (all have run=false)")
        else:
            click.echo(f"Active models: {', '.join(m['name'] for m in active_models)}")
        
        click.echo("\n✓ Configuration is valid")
        
    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


@cli.command("test-microphone")
@click.option("--microphone", "-m", type=str, help="Microphone name to test (partial match)")
@click.option("--duration", "-d", type=float, default=2.0, help="Test recording duration in seconds")
@click.option("--sample-rate", "-r", type=int, default=None, help="Sample rate to test (default: from config)")
def test_microphone(microphone, duration, sample_rate):
    """
    Test the configured or specified microphone.
    
    Records a short audio sample to verify the microphone is working.
    Uses sample rate from config if not specified.
    """
    from . import audio_devices
    
    # Get sample rate from config if not specified
    if sample_rate is None:
        if config_exists():
            cfg = get_config()
            sample_rate = cfg.RECORDING_SAMPLE_RATE
        else:
            sample_rate = 48000  # fallback default
    
    audio_devices.test_microphone_cli(
        microphone_name=microphone,
        duration=duration,
        sample_rate=sample_rate
    )


@cli.command("convert-flac-to-wav")
@click.option("--input-dir", type=str, help="Override input directory (default: from config)")
@click.option("--dry-run", is_flag=True, help="Show what would be converted without converting")
def convert_flac_to_wav(input_dir, dry_run):
    """
    Convert FLAC audio files to WAV format.
    
    Converts all FLAC files found in the audio save directory (or specified
    directory) to WAV format. Output files are saved in a 'conversion-flac-wav'
    subdirectory to keep originals intact.
    
    WAV files are uncompressed, resulting in larger file sizes but maximum
    compatibility with audio tools.
    """
    from . import audio_conversion
    
    # Get source directory from config if not specified
    if input_dir is None:
        input_dir = audio_conversion.get_audio_save_path()
    
    audio_conversion.convert_audio_files(
        source_dir=input_dir,
        output_subdir="conversion-flac-wav",
        source_format="flac",
        target_format="wav",
        dry_run=dry_run
    )


@cli.command("convert-wav-to-flac")
@click.option("--input-dir", type=str, help="Override input directory (default: from config)")
@click.option("--compression", "-c", type=int, default=5, help="FLAC compression level (0-8, default: 5)")
@click.option("--dry-run", is_flag=True, help="Show what would be converted without converting")
def convert_wav_to_flac(input_dir, compression, dry_run):
    """
    Convert WAV audio files to FLAC format.
    
    Converts all WAV files found in the audio save directory (or specified
    directory) to FLAC format. Output files are saved in a 'conversion-wav-flac'
    subdirectory to keep originals intact.
    
    FLAC provides lossless compression, reducing file size by ~40-60% while
    preserving audio quality. Compression level 0 is fastest, 8 is smallest.
    Default level 5 provides good balance.
    """
    from . import audio_conversion
    
    # Validate compression level
    if not 0 <= compression <= 8:
        click.echo("Error: Compression level must be between 0 and 8", err=True)
        return
    
    # Get source directory from config if not specified
    if input_dir is None:
        input_dir = audio_conversion.get_audio_save_path()
    
    audio_conversion.convert_audio_files(
        source_dir=input_dir,
        output_subdir="conversion-wav-flac",
        source_format="wav",
        target_format="flac",
        compression=compression,
        dry_run=dry_run
    )


@cli.command("resample-to-48khz")
@click.option("--input-dir", type=str, help="Override input directory (default: from config)")
@click.option("--dry-run", is_flag=True, help="Show what would be resampled without resampling")
def resample_to_48khz(input_dir, dry_run):
    """
    Resample audio files to 48kHz sample rate.
    
    Resamples all audio files (WAV/FLAC) found in the audio save directory
    (or specified directory) to 48kHz. Output files are saved in a
    'conversion-48khz' subdirectory to keep originals intact.
    
    48kHz is the standard sample rate for bird audio analysis with BirdNET
    models. Files already at 48kHz are skipped. Higher sample rates (192kHz,
    384kHz) are downsampled, which is useful for reducing file sizes when
    ultrasonic frequencies are not needed.
    """
    from . import audio_conversion
    
    # Get source directory from config if not specified
    if input_dir is None:
        input_dir = audio_conversion.get_audio_save_path()
    
    audio_conversion.resample_audio_files(
        source_dir=input_dir,
        output_subdir="conversion-48khz",
        target_sample_rate=48000,
        dry_run=dry_run
    )


@cli.command("info")
def info():
    """
    Show system and BirdPipe information.
    
    Displays version info, configuration status, and available models.
    """
    import platform
    
    click.echo("BirdPipe Audio Analysis System")
    click.echo("=" * 40)
    click.echo(f"Version: 0.1.0")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.system()} {platform.release()}")
    click.echo()
    
    config_path = get_config_path()
    click.echo(f"Config file: {config_path}")
    click.echo(f"Config exists: {'Yes' if config_exists() else 'No'}")
    
    if config_exists():
        config = get_config()
        click.echo()
        click.echo("Current settings:")
        click.echo(f"  Microphone: {config.MICROPHONE_NAME}")
        click.echo(f"  Sample rate: {config.RECORDING_SAMPLE_RATE} Hz")
        click.echo(f"  Duration: {config.RECORDING_DURATION}s")
        click.echo(f"  Test mode: {'Enabled' if config.TEST_MODE else 'Disabled'}")
        click.echo()
        
        active_models = [m for m in config.MODELS_TO_RUN if m.get("run", False)]
        click.echo(f"Active models ({len(active_models)}):")
        for model in active_models:
            click.echo(f"  - {model['name']} ({model['analysis_type']}, {model['region']})")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
