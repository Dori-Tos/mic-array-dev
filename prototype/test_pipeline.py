from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter, WienerFilter
from classes.AGC import AGC, TwoStageAGC, Amplifier, AGCChain
from classes.Codec import G711Codec, OpusCodec

import time
import numpy as np
from pathlib import Path
import logging

import sounddevice as sd


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    downsample_rate = None  # Process at native 48kHz, no resampling artifacts
    monitor_gain = 0.22
    
    mic_channel_numbers = [0, 1, 2, 3]
    
    logger = logging.getLogger("MicArrayTest")
    logger.setLevel(logging.DEBUG)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    # DAS for DOA estimation (faster than MVDR)
    das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    # MVDR for main beamforming (higher quality than DAS)
    mvdr_beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.93,             # MODERATE: 0.93 - Balanced covariance adaptation
                                           # Responds to source changes without over-reacting
        diagonal_loading=0.08,             # BALANCED: 0.08 - Gentle regularization for stability
                                           # Multi-source covariance stabilization without over-suppression
        spectral_whitening_factor=0.06,    # CONSERVATIVE: 0.06 - Restrained adaptive loading
                                           # Maintains noise suppression, avoids spurious whitening
        weight_smooth_alpha=0.55,           # FAST: 0.55 (was 0.88) - Weights adapt quickly
                                           # Prevents stale weights from lingering on side sources
        max_adaptive_loading_scale=5.0,    # MODERATE: 5.0 (was 6.0) - Caps adaptive loading
                                           # Prevents extreme over-regularization on low-SNR frames
    )
    
    # DIAGNOSTIC MODE: Swap beamformers to isolate instability
    # Uncomment ONE of these:
    beamformer_choice = mvdr_beamformer  # Current: MVDR with artifacts
    # beamformer_choice = das_beamformer   # Test: DAS (simpler, more stable)
    
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        beamformer=das_beamformer,  # Use DAS for fast DOA scanning
        scan_step_deg=5.0,
    )
    doa_estimator.freeze(0.0)
    
    echo_canceller = EchoCanceller(logger=logger, sample_rate=sample_rate, channels=4)
    
    # Passband to eliminate low-frequency rumble and high-frequency hiss
    # Wiener filter with speech-aware enhancements for noise reduction
    # NOW AT 48kHz (native) to avoid resampling artifacts
    filter_rate = sample_rate  # Always at native 48kHz, never at downsampled rate
    filters = [
        # Filters now at 48kHz to avoid resampling artifact amplification
        BandPassFilter(
            logger=logger, 
            sample_rate=filter_rate, 
            low_cutoff=300.0,       # Below: Rumble
            high_cutoff=4000.0,     # Above: Hiss
            order=4),
        
        WienerFilter(
            logger=logger,
            sample_rate=filter_rate,
            noise_alpha=0.995,
            gain_floor=0.015,
            gain_smooth_alpha=0.75,
            noise_update_snr_db=1.8,
            noise_update_rms=1.5e-3,
            pre_emphasis_db=3.0,
            formant_preservation_db=2.0,
            spectral_continuity_factor=0.55,
        ),
    ]
        
    agc_fast = AGC(
        logger=logger,
        target_rms=0.003,        # LOUDER: 0.003 (was 0.002, allows higher peaks)
                                 # Improves voice audibility without distortion
        min_gain=0.6,            # MODERATE: 0.6 (was 0.5 too low, 0.7 original)
                                 # Allows background suppression without over-attenuation
        max_gain=8.0,
        attack_ms=35.0,
        release_ms=200.0,
        noise_floor_rms=0.0,
        gate_gain=1.0,
        gate_open_ms=30.0,
        gate_close_ms=150.0,
        gate_hold_ms=20.0,
    )
    agc_slow = AGC(
        logger=logger,
        target_rms=0.012,        # LOUDER makeup: 0.012 (was 0.008, more generous gain)
                                 # Boosts overall volume without excessive amplification
        min_gain=0.8,            # BALANCED: 0.8 (was 0.7 too low, 0.9 original)
                                 # Suppress background moderately
        max_gain=7.0,            # MODERATE: 9.0 (was 8.0 too restrictive, 10.0 original)
                                 # Allow some makeup without over-amplifying
        attack_ms=125.0,
        release_ms=2200.0,
        noise_floor_rms=10.0,    # DISABLED GATE: Very high threshold
        gate_gain=1.0,           # DISABLED: 1.0 (no attenuation, gate always open)
        gate_open_ms=30.0,
        gate_close_ms=150.0,
        gate_hold_ms=100.0,
    )
    # Chain: Amplifier (3x gain) → TwoStageAGC (adaptive leveling with gate disabled)
    agc = AGCChain(logger=logger, stages=[
        Amplifier(logger=logger, gain=3.0, max_output=1.0),
        TwoStageAGC(logger=logger, stage1=agc_fast, stage2=agc_slow)
    ])
    codec = G711Codec(logger=logger)

    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=beamformer_choice,  # DIAGNOSTIC: Swap between mvdr_beamformer and das_beamformer
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec,
        monitor_gain=0.35,
        downsample_rate=downsample_rate,  # Process at 16kHz instead of 48kHz for ~3x speedup
        initial_silence_duration=2.0,  # Silence first 2 seconds to let filters adapt
    )

    array.start_realtime(blocksize=2048)
    array.start_output_monitoring()

    try:
        print("Realtime beamformed monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        array.stop_realtime()
        array.stop_output_monitoring()