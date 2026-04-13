"""
Diagnostic Test Protocol - Distortion Analysis
==============================================

Runs the same pipeline as test_pipeline.py but with comprehensive distortion
detection at each stage. Pinpoints where artifacts are introduced.

Usage:
    python test_distortion_diagnostic.py --gain-test 12.0 --duration 30 --enable-diagnostics

This will:
1. Run the full pipeline with specified gain
2. Monitor each stage for clipping/distortion
3. Report which stage introduces artifacts first
4. Save a detailed diagnostic report
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd

# Add classes directory to path
sys.path.insert(0, str(Path(__file__).parent))

from classes.Microphone import Microphone
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, PedalboardAGC, AGCChain
from classes.DistortionDiagnostics import (
    create_diagnostic_pipeline,
    print_distortion_report,
    DistortionDetector
)


def run_diagnostic_test(
    duration_seconds=30,
    device_index=None,
    sample_rate=48000,
    blocksize=960,
    adaptive_amplifier_gain=12.0,
    enable_diagnostics=True,
    output_file=None
):
    """
    Run diagnostic test on audio pipeline.
    
    Args:
        duration_seconds: Duration to test (seconds)
        device_index: Audio device index (None = default)
        sample_rate: Sample rate (Hz)
        blocksize: Block size per process call
        adaptive_amplifier_gain: Maximum gain for AdaptiveAmplifier
        enable_diagnostics: Enable distortion detection
        output_file: Optional file to save diagnostic report
    """
    
    # Setup logging
    logger = logging.getLogger("PipelineDiagnostic")
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Get audio device info
    if device_index is None:
        device_info = sd.query_devices(kind='input')
        device_index = device_info['index']
        device_name = device_info['name']
    else:
        device_info = sd.query_devices(device_index)
        device_name = device_info['name']
    
    print(f"\n{'='*70}")
    print(f"Pipeline Distortion Diagnostic Test")
    print(f"{'='*70}")
    print(f"Audio device: {device_name} (index {device_index})")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Block size: {blocksize} samples")
    print(f"Test duration: {duration_seconds} seconds")
    print(f"AdaptiveAmplifier max gain: {adaptive_amplifier_gain}x ({20*np.log10(adaptive_amplifier_gain):.1f} dB)")
    print(f"Diagnostics enabled: {enable_diagnostics}")
    print(f"{'='*70}\n")
    
    # Initialize filters
    filters = [
        BandPassFilter(
            logger=logger,
            sample_rate=sample_rate,
            low_cutoff=300.0,
            high_cutoff=4000.0,
            order=4
        ),
        SpectralSubtractionFilter(
            logger=logger,
            sample_rate=sample_rate,
            noise_factor=0.65,
            gain_floor=0.35,
            noise_alpha=0.995,
            noise_update_snr_db=8.0,
            gain_smooth_alpha=0.92,
        ),
    ]
    
    # Initialize AGC with specified max_gain
    agc = AGCChain(logger=logger, stages=[
        AdaptiveAmplifier(
            logger=logger,
            target_rms=0.08,
            min_gain=1.0,
            max_gain=adaptive_amplifier_gain,  # Use specified gain
            adapt_alpha=0.04,
            speech_activity_rms=0.00012,
            silence_decay_alpha=0.008,
            activity_hold_ms=600.0,
            peak_protect_threshold=0.35,
            peak_protect_strength=0.85,
            max_gain_warn_rms_min=0.001,
        ),
        PedalboardAGC(
            logger=logger,
            sample_rate=sample_rate,
            threshold_db=-20.0,
            ratio=3.5,
            attack_ms=3.0,
            release_ms=140.0,
            limiter_threshold_db=-1.4,
            limiter_release_ms=100.0
        ),
    ])
    
    # Wrap with diagnostics if enabled
    if enable_diagnostics:
        diag_filters, diag_agc = create_diagnostic_pipeline(filters, agc, logger)
        print("Diagnostics enabled - monitoring each stage for clipping/distortion\n")
    else:
        diag_filters = [{'process': f.apply} for f in filters]
        diag_agc = {'process': agc.process}
    
    # Audio stream
    num_blocks = int(np.ceil(duration_seconds * sample_rate / blocksize))
    stream_duration = num_blocks * blocksize / sample_rate
    
    print(f"Recording {num_blocks} blocks ({stream_duration:.1f} seconds)...\n")
    
    try:
        # Open stream
        with sd.InputStream(device=device_index, channels=1, samplerate=sample_rate, 
                            blocksize=blocksize, latency='low'):
            block_data = []
            distortion_detector = DistortionDetector(logger)
            
            for block_idx in range(num_blocks):
                # Read block from device
                audio_data, _ = sd.rec(blocksize, samplerate=sample_rate, channels=1, device=device_index)
                audio_data = np.squeeze(audio_data).astype(np.float32)
                
                if audio_data.ndim == 0:
                    audio_data = np.array([audio_data])
                
                # Process through filters
                signal = audio_data.copy()
                for filt in (diag_filters if enable_diagnostics else [f for f in filters]):
                    if enable_diagnostics:
                        signal = filt.process(signal, sample_rate)
                    else:
                        signal = filt.apply(signal)
                
                # Process through AGC
                if enable_diagnostics:
                    signal = diag_agc.process(signal, sample_rate)
                else:
                    signal = agc.process(signal, sample_rate)
                
                # Check for clipping in final output
                clipping_info = distortion_detector.detect_hard_clipping(signal)
                if clipping_info['has_hard_clipping']:
                    print(f"[Block {block_idx}] CLIPPING in output: "
                          f"{clipping_info['clipped_percentage']:.2f}% "
                          f"({clipping_info['clipped_sample_count']} samples)")
                
                block_data.append({
                    'block_idx': block_idx,
                    'audio': signal,
                    'clipping': clipping_info['has_hard_clipping'],
                })
                
                # Progress
                if (block_idx + 1) % max(1, num_blocks // 10) == 0:
                    elapsed = (block_idx + 1) * blocksize / sample_rate
                    print(f"  [{elapsed:.1f}s / {stream_duration:.1f}s] {block_idx + 1} blocks processed")
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return None
    
    print("\n" + "="*70)
    print("Diagnostic Analysis Complete")
    print("="*70)
    
    # Print reports
    if enable_diagnostics:
        print_distortion_report(diag_filters, diag_agc)
        
        # Summary
        total_blocks = len(block_data)
        clipped_blocks = sum(1 for b in block_data if b['clipping'])
        print(f"\nSummary:")
        print(f"  Total blocks: {total_blocks}")
        print(f"  Blocks with clipping: {clipped_blocks}")
        print(f"  Clipping rate: {100*clipped_blocks/max(total_blocks, 1):.1f}%")
    else:
        # Report without diagnostics
        total_blocks = len(block_data)
        clipped_blocks = sum(1 for b in block_data if b['clipping'])
        print(f"Total blocks: {total_blocks} ({stream_duration:.1f} seconds)")
        print(f"Blocks with output clipping: {clipped_blocks}")
        print(f"Clipping rate: {100*clipped_blocks/max(total_blocks, 1):.1f}%")
    
    print("="*70)
    
    # Save report if requested
    if output_file:
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'device_name': device_name,
            'sample_rate': sample_rate,
            'blocksize': blocksize,
            'max_gain': adaptive_amplifier_gain,
            'total_blocks': len(block_data),
            'clipped_blocks': sum(1 for b in block_data if b['clipping']),
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Diagnostic test for pipeline distortion analysis'
    )
    parser.add_argument('--duration', type=float, default=30,
                        help='Test duration in seconds (default: 30)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (default: system default)')
    parser.add_argument('--gain', type=float, default=12.0,
                        help='AdaptiveAmplifier max gain in dB (default: 12 dB = 4.0x)')
    parser.add_argument('--blocksize', type=int, default=960,
                        help='Block size in samples (default: 960)')
    parser.add_argument('--disable-diagnostics', action='store_true',
                        help='Disable distortion diagnostics (test without monitoring)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--output', type=str, default=None,
                        help='Save diagnostic report to JSON file')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print("=" * 70)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            try:
                is_input = dev['max_input_channels'] > 0
                channels = dev['max_input_channels']
                samplerate = int(dev['default_samplerate'])
                if is_input:
                    default_marker = ' [DEFAULT INPUT]' if i == sd.default.device[0] else ''
                    print(f"  {i:2d}: {dev['name']:<40} ({channels} channels, {samplerate} Hz){default_marker}")
            except (IndexError, TypeError):
                pass
        print("=" * 70)
        exit(0)
    
    # Convert dB to linear gain
    max_gain = 10 ** (args.gain / 20.0)
    
    # Run test
    run_diagnostic_test(
        duration_seconds=args.duration,
        device_index=args.device,
        adaptive_amplifier_gain=max_gain,
        enable_diagnostics=not args.disable_diagnostics,
        output_file=args.output
    )
    
    # Usage examples:
    #
    # List devices:
    # python test_distortion_diagnostic.py --list-devices
    #
    # Standard diagnostic test (30 sec, 12 dB gain):
    # python test_distortion_diagnostic.py --device 1 --duration 30 --gain 12
    #
    # High gain test (20 dB = 10x):
    # python test_distortion_diagnostic.py --device 1 --duration 30 --gain 20
    #
    # Extreme gain test to force distortion (24 dB = 16x):
    # python test_distortion_diagnostic.py --device 1 --duration 60 --gain 24
    #
    # Without diagnostics (baseline):
    # python test_distortion_diagnostic.py --device 1 --disable-diagnostics --duration 30 --gain 12
    #
    # Save report:
    # python test_distortion_diagnostic.py --device 1 --duration 30 --output diagnostic_report.json
