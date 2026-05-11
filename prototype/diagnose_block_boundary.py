"""
Proper block-boundary discontinuity diagnostic.
Processes several seconds of audio through beamformer and analyzes block seams.
"""
import numpy as np
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / "classes"))

from classes.Beamformer import MVDRBeamformer
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("BlockBoundaryDiag")

# Setup beamformer matching pipeline config
beamformer = MVDRBeamformer(
    logger=logger,
    mic_channel_numbers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    sample_rate=48000,
    mic_spacing_m=0.05,
    covariance_alpha=0.7,   # Aggressive smoothing: slower covariance adaptation
    diagonal_loading=0.15,
    weight_smooth_alpha=0.88,  # More aggressive weight smoothing
)

# Parameters
blocksize = 960  # 20ms at 48kHz
sample_rate = 48000
num_blocks = 240  # ~4.8 seconds of continuous processing
total_channels = 14

print("\n" + "="*80)
print("PROPER BLOCK-BOUNDARY DISCONTINUITY DIAGNOSTIC")
print("="*80)
print(f"Processing {num_blocks} blocks × {blocksize} samples = {num_blocks * blocksize / sample_rate:.1f} seconds")
print(f"Block duration: {blocksize / sample_rate * 1000:.1f} ms")
print("="*80 + "\n")

# Collect all outputs and analyze boundaries
all_outputs = []
boundary_jumps = []
boundary_diffs = []  # Per-sample difference at boundaries

# Generate continuous audio stream (simulate real input with some structure)
np.random.seed(42)

for block_idx in range(num_blocks):
    # Create continuous broadband noise (same random seed progression for realistic continuity)
    # Add some structure: modulate amplitude to simulate speech/silence patterns
    amplitude = 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * block_idx / 100))  # Slow modulation
    block = np.random.randn(blocksize, total_channels).astype(np.float32) * amplitude
    
    # Process through beamformer
    output = beamformer.apply(block, theta_deg=0.0)
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    all_outputs.append(output)
    
    # Analyze block boundary
    if len(all_outputs) > 1:
        prev_output = all_outputs[-2]
        curr_output = all_outputs[-1]
        
        last_sample_prev = prev_output[-1]
        first_sample_curr = curr_output[0]
        
        # Raw jump
        jump = abs(last_sample_prev - first_sample_curr)
        boundary_jumps.append(jump)
        
        # Per-sample difference magnitude (absolute change from last to first)
        boundary_diffs.append(last_sample_prev - first_sample_curr)

# Concatenate all outputs to analyze as continuous signal
concatenated = np.concatenate(all_outputs)

print(f"Total samples processed: {len(concatenated)}")
print(f"Total blocks: {len(all_outputs)}")
print(f"\nBlock boundary statistics:")
print(f"  Mean jump: {np.mean(boundary_jumps):.6f}")
print(f"  Median jump: {np.median(boundary_jumps):.6f}")
print(f"  Max jump: {np.max(boundary_jumps):.6f}")
print(f"  Min jump: {np.min(boundary_jumps):.6f}")
print(f"  Std dev: {np.std(boundary_jumps):.6f}")

# Analyze which boundaries have the largest jumps
if boundary_jumps:
    largest_jumps_idx = np.argsort(boundary_jumps)[-10:]
    print(f"\nTop 10 largest boundary jumps:")
    for rank, idx in enumerate(reversed(largest_jumps_idx), 1):
        print(f"  {rank}. Block {idx+1}→{idx+2}: {boundary_jumps[idx]:.6f}")

# Signal statistics
signal_rms = float(np.sqrt(np.mean(concatenated ** 2)))
signal_peak = float(np.max(np.abs(concatenated)))
print(f"\nSignal statistics:")
print(f"  Overall RMS: {signal_rms:.6f}")
print(f"  Overall peak: {signal_peak:.6f}")

# Normalized metrics
if signal_rms > 1e-6:
    mean_jump_rel = np.mean(boundary_jumps) / signal_rms
    max_jump_rel = np.max(boundary_jumps) / signal_rms
    print(f"  Mean jump / RMS: {mean_jump_rel:.3f}x")
    print(f"  Max jump / RMS: {max_jump_rel:.3f}x")

print(f"\n" + "="*80)
if np.max(boundary_jumps) > 0.001:
    print(f"⚠️  SIGNIFICANT DISCONTINUITIES DETECTED at {np.sum(np.array(boundary_jumps) > 0.001)} boundaries")
    print(f"   Max discontinuity: {np.max(boundary_jumps):.6f}")
    print(f"   This will be amplified by AGC in quiet regions.")
else:
    print(f"✓ Block boundaries are clean (max jump < 0.001)")
print("="*80)
