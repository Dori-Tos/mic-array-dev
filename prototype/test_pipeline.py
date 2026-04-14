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
        covariance_alpha=0.95,             # Controls how quickly the covariance matrix adapts to new data
        diagonal_loading=0.15,             # Higher regularization suppresses off-axis interference
                                           # Trade-off: reduces directional response slightly but cuts side voices
        spectral_whitening_factor=0.12,    # Keeps main lobe energy while suppressing diffuse noise
        weight_smooth_alpha=0.72,          # Baseline smoothing (will adapt based on SNR)
        max_adaptive_loading_scale=4.0,    # Prevents extreme gain at low-SNR frequencies
        coherence_suppression_strength=0.8,  # Suppress diffuse noise via coherence-based weighting (0-1)
                                            # 0.0 = pure MVDR, 0.6 = balanced, 1.0 = aggressive room noise rejection
        weight_smooth_alpha_min=0.45,      # NEW: Sharp main lobe during steady-state (high-SNR speech)
        weight_smooth_alpha_max=0.82,      # NEW: Stable weights in noisy conditions (low-SNR)
        snr_threshold_for_sharpening=2.0,  # NEW: SNR above this triggers sharpening (lower smoothing)
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
        
        SpectralSubtractionFilter(
            logger=logger,
            sample_rate=sample_rate,
            noise_factor=0.65,              # Moderate noise suppression 
            gain_floor=0.55,                # Higher floor prevents over-suppression of low freqs (was 0.35)
            noise_alpha=0.995,              # Very slow noise learning prevents formant suppression
            noise_update_snr_db=8.0,        # Can now update during low-SNR moments (protected from onset corruption)
            gain_smooth_alpha=0.92,         # Very strong uniform gain smoothing locks formants, eliminates pops
        ),
    ]
    
    agc = AGCChain(logger=logger, stages=[
        AdaptiveAmplifier(
            logger=logger,
            target_rms=0.08,          # Boost normal speech toward target operating level
            min_gain=1.0,             # Do NOT attenuate normal program material
            max_gain=6.0,             # Aggressive limit to prevent oscillation (was 12.0)
            adapt_alpha=0.04,         # Balanced adaptation speed
            speech_activity_rms=0.00012,  # Treat lower-level speech as active (restore quiet voice audibility)
            silence_decay_alpha=0.008,    # Slower decay to avoid dropping between words/syllables
            activity_hold_ms=600.0,       # Hold gain after speech to preserve phrase continuity
            peak_protect_threshold=0.30,  # More generous headroom to prevent oscillation (was 0.25)
            peak_protect_strength=1.0,    # Maximum protection (was 0.85)
            max_gain_warn_rms_min=0.001,  # Suppress max-gain warnings for near-silence frames
        ),
        
        PedalboardAGC(
            logger=logger,
            sample_rate=sample_rate,
            threshold_db=-20.0,       # Compress only above normal speech operating zone
            ratio=2.0,                # Gentler compression (was 3.5) to prevent hunting
            attack_ms=3.0,            # Faster reaction for loud transients
            release_ms=140.0,         # Smooth recovery to avoid pumping
            limiter_threshold_db=-7.0,    # Much lower to catch peaks before clipping (was -3.0)
            limiter_release_ms=50.0       # Faster limiter response (was 100.0)
        ),
    ])
        
    
    output_mode = "local"  # "local" or "codec"
    if output_mode == "codec":
        codec = OpusCodec(
            logger=logger,
            bitrate=24000,
            frame_duration_ms=10,
            application="voip",
            remote_host="172.98.1.59",
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
        initial_silence_duration=2.0,  # Silence period for baseline learning (filter protects against corruption during onset)
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