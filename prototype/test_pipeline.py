from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter, WienerFilter, SpectralSubtractionFilter
from classes.AGC import AGC, TwoStageAGC, Amplifier, AdaptiveAmplifier, AGCChain, Limiter, PeakHoldAGC, PedalboardAGC
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
    logger.setLevel(logging.INFO)
    
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
        # covariance_alpha=0.93,             # MODERATE: 0.93 - Balanced covariance adaptation
        #                                    # Responds to source changes without over-reacting
        # diagonal_loading=0.08,             # BALANCED: 0.08 - Gentle regularization for stability
        #                                    # Multi-source covariance stabilization without over-suppression
        # spectral_whitening_factor=0.06,    # CONSERVATIVE: 0.06 - Restrained adaptive loading
        #                                    # Maintains noise suppression, avoids spurious whitening
        # weight_smooth_alpha=0.55,           # FAST: 0.55 (was 0.88) - Weights adapt quickly
        #                                    # Prevents stale weights from lingering on side sources
        # max_adaptive_loading_scale=5.0,    # MODERATE: 5.0 (was 6.0) - Caps adaptive loading
        #                                    # Prevents extreme over-regularization on low-SNR frames
                                           
        covariance_alpha=0.95,             # Controls how quickly the covariance matrix adapts to new data
        diagonal_loading=0.15,             # Higher regularization suppresses off-axis interference
                                           # Trade-off: reduces directional response slightly but cuts side voices
        spectral_whitening_factor=0.12,    # Keeps main lobe energy while suppressing diffuse noise
        weight_smooth_alpha=0.72,          # Smoother weights = cleaner main voice, less side distortion
        max_adaptive_loading_scale=4.0,    # Prevents extreme gain at low-SNR frequencies
    )


    beamformer_choice = mvdr_beamformer
    
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
    # Spectral Subtraction to supress background noise in voice region
    filter_rate = sample_rate 
    filters = [
        BandPassFilter(
            logger=logger, 
            sample_rate=sample_rate, 
            low_cutoff=300.0, 
            high_cutoff=4000.0, 
            order=4),
        
        # SpectralSubtractionFilter(
        #     logger=logger,
        #     sample_rate=sample_rate,
        #     noise_factor=0.9,
        #     gain_floor=0.2,
        #     noise_alpha=0.98,
        #     noise_update_snr_db=4.0,
        # ),
        
        SpectralSubtractionFilter(
            logger=logger,
            sample_rate=sample_rate,
            noise_factor=0.65,              # Aggressivenes of noise reduction 
            gain_floor=0.35,                # Prevents aggressive suppression causing robotification
            noise_alpha=0.99,               # Prevents chasing speech transients as noise
            noise_update_snr_db=8.0,        #  More stable noise floor estimate
        ),
        
        # WienerFilter(
        #     logger=logger,
        #     sample_rate=sample_rate,
        #     noise_alpha=0.995,
        #     gain_floor=0.015,
        #     gain_smooth_alpha=0.75,
        #     noise_update_snr_db=1.8,
        #     noise_update_rms=1.5e-3,
        #     pre_emphasis_db=3.0,
        #     formant_preservation_db=2.0,
        #     spectral_continuity_factor=0.55,
        # ),
    ]
    
    agc = AGCChain(logger=logger, stages=[
        # AdaptiveAmplifier(
        #     logger=logger,
        #     target_rms=0.1,           # Aim for -20dB baseline
        #     min_gain=1.0,             # Don't over-suppress
        #     max_gain=16.0,            # Allow 24dB max boost
        #     adapt_alpha=0.05,         # Slow adaptation (avoid pumping)
        # ),
        
        # PedalboardAGC(
        #     logger=logger,
        #     sample_rate=sample_rate,
        #     threshold_db=-30.0,
        #     ratio=4.0,
        #     attack_ms=10.0,
        #     release_ms=100.0,
        #     limiter_threshold_db=-0.1,
        #     limiter_release_ms=50.0
        # ),
        
        AdaptiveAmplifier(
            logger=logger,
            target_rms=0.08,          # Boost normal speech toward target operating level
            min_gain=1.0,             # Do NOT attenuate normal program material
            max_gain=12.0,            # Allow enough lift for quieter speech
            adapt_alpha=0.04,         # Balanced adaptation speed
            speech_activity_rms=0.00012,  # Treat lower-level speech as active (restore quiet voice audibility)
            silence_decay_alpha=0.008,    # Slower decay to avoid dropping between words/syllables
            activity_hold_ms=600.0,       # Hold gain after speech to preserve phrase continuity
            peak_protect_threshold=0.35, # Start gain damping when peaks indicate loud transients
            peak_protect_strength=0.85,  # Strong damping to reduce limiter stress on loud voice
            max_gain_warn_rms_min=0.001, # Suppress max-gain warnings for near-silence frames
        ),
        
        PedalboardAGC(
            logger=logger,
            sample_rate=sample_rate,
            threshold_db=-20.0,       # Compress only above normal speech operating zone
            ratio=3.5,                # Moderate peak control without flattening normal signal
            attack_ms=3.0,            # Faster reaction for loud transients
            release_ms=140.0,         # Smooth recovery to avoid pumping
            limiter_threshold_db=-1.4, # Protect output from hard clipping on very loud bursts
            limiter_release_ms=100.0  # Smoother limiter recovery
        ),
    ])
        
    # filters = [
    #     # Filters now at 48kHz to avoid resampling artifact amplification
    #     BandPassFilter(
    #         logger=logger, 
    #         sample_rate=filter_rate, 
    #         low_cutoff=300.0,       # Below: Rumble
    #         high_cutoff=4000.0,     # Above: Hiss
    #         order=4),
        
    #     WienerFilter(
    #         logger=logger,
    #         sample_rate=filter_rate,
    #         noise_alpha=0.995,
    #         gain_floor=0.015,
    #         gain_smooth_alpha=0.75,
    #         noise_update_snr_db=1.8,
    #         noise_update_rms=1.5e-3,
    #         pre_emphasis_db=3.0,
    #         formant_preservation_db=2.0,
    #         spectral_continuity_factor=0.55,
    #     ),
    # ]
    # agc_fast = AGC(
    #     logger=logger,
    #     target_rms=0.003,        # LOUDER: 0.003 (was 0.002, allows higher peaks)
    #                              # Improves voice audibility without distortion
    #     min_gain=0.6,            # MODERATE: 0.6 (was 0.5 too low, 0.7 original)
    #                              # Allows background suppression without over-attenuation
    #     max_gain=8.0,
    #     attack_ms=35.0,
    #     release_ms=200.0,
    #     noise_floor_rms=0.0,
    #     gate_gain=1.0,
    #     gate_open_ms=30.0,
    #     gate_close_ms=150.0,
    #     gate_hold_ms=20.0,
    # )
    # agc_slow = AGC(
    #     logger=logger,
    #     target_rms=0.012,        # LOUDER makeup: 0.012 (was 0.008, more generous gain)
    #                              # Boosts overall volume without excessive amplification
    #     min_gain=0.8,            # BALANCED: 0.8 (was 0.7 too low, 0.9 original)
    #                              # Suppress background moderately
    #     max_gain=7.0,            # MODERATE: 9.0 (was 8.0 too restrictive, 10.0 original)
    #                              # Allow some makeup without over-amplifying
    #     attack_ms=125.0,
    #     release_ms=2200.0,
    #     noise_floor_rms=5.0,    
    #     gate_gain=1.0,          
    #     gate_open_ms=30.0,
    #     gate_close_ms=150.0,
    #     gate_hold_ms=100.0,
    # )
    # agc = AGCChain(logger=logger, stages=[
    #     Amplifier(logger=logger, gain=3.0, max_output=1.0),
    #     TwoStageAGC(logger=logger, stage1=agc_fast, stage2=agc_slow)
    # ])
    output_mode = "local"  # "local" or "codec"
    if output_mode == "codec":
        codec = OpusCodec(
            logger=logger,
            bitrate=24000,
            frame_duration_ms=10,
            application="voip",
            remote_host="172.98.1.61",
            remote_port=5004,
        )
    else:
        # In local mode, use a lightweight codec object; codec transport is not used.
        codec = G711Codec(logger=logger)

    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=beamformer_choice,
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec,
        monitor_gain=monitor_gain,
        output_mode=output_mode,
        output_boundary_fade_ms=0.0,
        downsample_rate=downsample_rate,
        initial_silence_duration=2.0,  # Silence first 2 seconds to let filters adapt
    )

    array.start_realtime(blocksize=960)
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