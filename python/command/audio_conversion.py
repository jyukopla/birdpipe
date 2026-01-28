"""
Audio Format Conversion for BirdPipe

Provides functionality to convert between WAV and FLAC formats.
"""

import os
from pathlib import Path
from typing import Optional

import click
import soundfile as sf


def convert_audio_files(
    source_dir: str,
    output_subdir: str,
    source_format: str,
    target_format: str,
    compression: int = 5,
    dry_run: bool = False
):
    """
    Convert audio files from one format to another.
    
    Args:
        source_dir: Directory containing source audio files
        output_subdir: Subdirectory name for converted files (e.g., 'conversion-flac-wav')
        source_format: Source format ('wav' or 'flac')
        target_format: Target format ('wav' or 'flac')
        compression: FLAC compression level (0-8), only used when target is FLAC
        dry_run: If True, only show what would be converted
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        click.echo(f"Error: Source directory does not exist: {source_dir}", err=True)
        return
    
    if not source_path.is_dir():
        click.echo(f"Error: Source path is not a directory: {source_dir}", err=True)
        return
    
    # Create output directory
    output_path = source_path / output_subdir
    
    # Find all source files
    source_extension = f".{source_format.lower()}"
    source_files = list(source_path.glob(f"**/*{source_extension}"))
    
    if not source_files:
        click.echo(f"No {source_format.upper()} files found in {source_dir}")
        return
    
    click.echo(f"Found {len(source_files)} {source_format.upper()} file(s) in {source_dir}")
    click.echo(f"Output directory: {output_path}")
    click.echo()
    
    if dry_run:
        click.echo("DRY RUN - No files will be converted")
        click.echo()
        for source_file in source_files:
            target_filename = source_file.stem + f".{target_format.lower()}"
            click.echo(f"  Would convert: {source_file.name} → {target_filename}")
        click.echo()
        click.echo(f"Total: {len(source_files)} file(s) would be converted")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert files
    success_count = 0
    error_count = 0
    
    for idx, source_file in enumerate(source_files, 1):
        target_filename = source_file.stem + f".{target_format.lower()}"
        target_file = output_path / target_filename
        
        click.echo(f"[{idx}/{len(source_files)}] Converting {source_file.name}...", nl=False)
        
        try:
            # Read audio file
            audio_data, sample_rate = sf.read(str(source_file))
            
            # Prepare write parameters
            write_params = {
                'file': str(target_file),
                'data': audio_data,
                'samplerate': sample_rate,
            }
            
            # Set format-specific parameters
            if target_format.lower() == 'flac':
                write_params['format'] = 'FLAC'
                write_params['subtype'] = 'PCM_16'
                # FLAC compression is set via soundfile backend (libsndfile)
                # Compression level 0-8 where 5 is default
                os.environ['SF_COMPRESSION'] = str(compression)
            elif target_format.lower() == 'wav':
                write_params['format'] = 'WAV'
                write_params['subtype'] = 'PCM_16'
            
            # Write converted file
            sf.write(**write_params)
            
            # Calculate file sizes
            source_size_mb = source_file.stat().st_size / (1024 * 1024)
            target_size_mb = target_file.stat().st_size / (1024 * 1024)
            
            click.echo(f" ✓ ({source_size_mb:.2f}MB → {target_size_mb:.2f}MB)")
            success_count += 1
            
        except Exception as e:
            click.echo(f" ✗ Error: {e}", err=True)
            error_count += 1
    
    # Summary
    click.echo()
    click.echo("="*60)
    click.echo("Conversion Summary")
    click.echo("="*60)
    click.echo(f"Successfully converted: {success_count} file(s)")
    if error_count > 0:
        click.echo(f"Failed: {error_count} file(s)", err=True)
    click.echo(f"Output directory: {output_path}")
    click.echo("="*60)


def resample_audio_files(
    source_dir: str,
    output_subdir: str,
    target_sample_rate: int,
    dry_run: bool = False
):
    """
    Resample audio files to a target sample rate.
    
    Args:
        source_dir: Directory containing source audio files
        output_subdir: Subdirectory name for resampled files (e.g., 'conversion-48khz')
        target_sample_rate: Target sample rate in Hz (e.g., 48000)
        dry_run: If True, only show what would be resampled
    """
    import librosa
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        click.echo(f"Error: Source directory does not exist: {source_dir}", err=True)
        return
    
    if not source_path.is_dir():
        click.echo(f"Error: Source path is not a directory: {source_dir}", err=True)
        return
    
    # Create output directory
    output_path = source_path / output_subdir
    
    # Find all audio files (WAV and FLAC)
    source_files = list(source_path.glob("**/*.wav")) + list(source_path.glob("**/*.flac"))
    source_files += list(source_path.glob("**/*.WAV")) + list(source_path.glob("**/*.FLAC"))
    
    if not source_files:
        click.echo(f"No audio files (WAV/FLAC) found in {source_dir}")
        return
    
    click.echo(f"Found {len(source_files)} audio file(s) in {source_dir}")
    click.echo(f"Target sample rate: {target_sample_rate} Hz ({target_sample_rate/1000:.0f}kHz)")
    click.echo(f"Output directory: {output_path}")
    click.echo()
    
    if dry_run:
        click.echo("DRY RUN - No files will be resampled")
        click.echo()
        for source_file in source_files:
            try:
                # Read just the sample rate without loading full audio
                info = sf.info(str(source_file))
                current_sr = info.samplerate
                duration = info.duration
                
                if current_sr == target_sample_rate:
                    click.echo(f"  Skip (already {target_sample_rate}Hz): {source_file.name}")
                else:
                    click.echo(f"  Would resample: {source_file.name} ({current_sr}Hz → {target_sample_rate}Hz, {duration:.1f}s)")
            except Exception as e:
                click.echo(f"  Error checking: {source_file.name} - {e}")
        click.echo()
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Resample files
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, source_file in enumerate(source_files, 1):
        # Preserve original extension
        target_file = output_path / source_file.name
        
        try:
            # Read audio file
            audio_data, current_sr = sf.read(str(source_file))
            
            # Check if resampling is needed
            if current_sr == target_sample_rate:
                click.echo(f"[{idx}/{len(source_files)}] Skipping {source_file.name} (already {target_sample_rate}Hz)")
                skip_count += 1
                continue
            
            click.echo(f"[{idx}/{len(source_files)}] Resampling {source_file.name} ({current_sr}Hz → {target_sample_rate}Hz)...", nl=False)
            
            # Resample using librosa (high quality kaiser_fast)
            resampled_audio = librosa.resample(
                audio_data,
                orig_sr=current_sr,
                target_sr=target_sample_rate,
                res_type='kaiser_fast'
            )
            
            # Determine output format from extension
            file_extension = source_file.suffix.lower()
            if file_extension == '.flac':
                write_format = 'FLAC'
            else:
                write_format = 'WAV'
            
            # Write resampled file
            sf.write(
                file=str(target_file),
                data=resampled_audio,
                samplerate=target_sample_rate,
                format=write_format,
                subtype='PCM_16'
            )
            
            # Calculate file sizes
            source_size_mb = source_file.stat().st_size / (1024 * 1024)
            target_size_mb = target_file.stat().st_size / (1024 * 1024)
            
            click.echo(f" ✓ ({source_size_mb:.2f}MB → {target_size_mb:.2f}MB)")
            success_count += 1
            
        except Exception as e:
            click.echo(f" ✗ Error: {e}", err=True)
            error_count += 1
    
    # Summary
    click.echo()
    click.echo("="*60)
    click.echo("Resampling Summary")
    click.echo("="*60)
    click.echo(f"Successfully resampled: {success_count} file(s)")
    if skip_count > 0:
        click.echo(f"Skipped (already at target rate): {skip_count} file(s)")
    if error_count > 0:
        click.echo(f"Failed: {error_count} file(s)", err=True)
    click.echo(f"Output directory: {output_path}")
    click.echo("="*60)


def get_audio_save_path():
    """Get the audio save path from configuration."""
    try:
        from .config import get_config
        config = get_config()
        return config.AUDIO_SAVE_PATH
    except Exception as e:
        click.echo(f"Warning: Could not load config: {e}", err=True)
        return "/var/data/audiofiles"  # Default fallback
