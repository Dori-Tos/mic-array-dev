"""
Component Isolation Test - Find the source of artifacts on loud voices
Tests 5 configurations to pinpoint the culprit
"""

from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import BandPassFilter, WienerFilter, SpectralSubtractionFilter
from classes.AGC import AGC, TwoStageAGC, Amplifier, AdaptiveAmplifier, Limiter, PeakHoldAGC, AGCChain, PedalboardAGC
from classes.Codec import G711Codec

import time
import numpy as np
from pathlib import Path
import logging

# TEST MODE: Choose which configuration to test
TEST_MODE = 0  # 0-9, see descriptions below

"""
DIAGNOSTIC MODES (find where distortion starts):
MODE 0: BEAMFORMER ONLY (minimal AGC) - isolate beamformer
MODE 1: Beamformer ONLY (8x amplifier) - Switch MVDR/DAS

MAIN MODES:
MODE 2: BandPass + SLOW AGC (gate disabled)
MODE 3: Wiener + SLOW AGC (gate disabled) - DIAGNOSE WIENER ARTIFACTS
MODE 4: BandPass + Spectral Subtraction (NEW - simpler noise reduction)
MODE 5: SpectralSubtraction + PedalboardAGC (adaptive amp, natural voice)
MODE 6: Full Clean (BandPass + SpectralSubtraction + PedalboardAGC)

REFERENCE MODES:
MODE 7: REFERENCE - BandPass + Wiener + FAST AGC (old problem baseline)
MODE 8: REFERENCE - BandPass + Wiener + SLOW AGC (old test)
MODE 9+: LEGACY - BandPass + Wiener + AMPLIFIER + PEAK-HOLD AGC
"""

# BEAMFORMER CHOICE: Switch between MVDR and DAS to isolate beamformer artifacts
USE_DAS = False # Set to True to test with DAS instead of MVDR

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    mic_channel_numbers = [0, 1, 2, 3]
    
    logger = logging.getLogger("ComponentTest")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    das_beamformer = DASBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )
    
    mvdr_beamformer = MVDRBeamformer(
        logger=logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.93,
        diagonal_loading=0.08,
        spectral_whitening_factor=0.06,
        weight_smooth_alpha=0.55,
        max_adaptive_loading_scale=5.0,
    )
    
    doa_estimator = IterativeDOAEstimator(
        logger=logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        beamformer=das_beamformer,
        scan_step_deg=5.0,
    )
    doa_estimator.freeze(0.0)
    
    echo_canceller = EchoCanceller(logger=logger, sample_rate=sample_rate, channels=4)
       
    # SELECT BEAMFORMER: MVDR (current, may have artifacts) vs DAS (simpler, cleaner)
    if USE_DAS:
        selected_beamformer = das_beamformer
        beamformer_desc = "DAS (simpler, more stable)"
    else:
        selected_beamformer = mvdr_beamformer
        beamformer_desc = "MVDR (adaptive, may have artifacts)"
    
    # Configure filters and AGC based on test mode
    # Fast AGC (current, causes artifacts):
    agc_fast_normal = AGC(logger=logger, target_rms=0.003, min_gain=0.6, max_gain=8.0,
                   attack_ms=35.0, release_ms=200.0, noise_floor_rms=0.0,
                   gate_gain=1.0, gate_open_ms=30.0, gate_close_ms=150.0, gate_hold_ms=20.0)
    agc_slow_normal = AGC(logger=logger, target_rms=0.012, min_gain=0.8, max_gain=7.0,
                   attack_ms=125.0, release_ms=2200.0, noise_floor_rms=0.00012,
                   gate_gain=0.35, gate_open_ms=30.0, gate_close_ms=150.0, gate_hold_ms=100.0)
    
    # SLOW AGC (potential fix - very gradual gain changes):
    agc_fast_slow = AGC(logger=logger, target_rms=0.003, min_gain=0.6, max_gain=4.0,  # REDUCED: 8.0 → 4.0
                   attack_ms=200.0, release_ms=1000.0, noise_floor_rms=0.0,
                   gate_gain=1.0, gate_open_ms=100.0, gate_close_ms=500.0, gate_hold_ms=100.0)
    agc_slow_slow = AGC(logger=logger, target_rms=0.012, min_gain=0.8, max_gain=3.0,   # REDUCED: 7.0 → 3.0
                   attack_ms=500.0, release_ms=5000.0, noise_floor_rms=10.0,  # DISABLED GATE (very high threshold)
                   gate_gain=1.0, gate_open_ms=100.0, gate_close_ms=500.0, gate_hold_ms=500.0)  # gate_gain=1.0 (no attenuation)
    
    if TEST_MODE == 0:
        # DIAGNOSTIC: Beamformer ONLY (no echo canceller, check beamformer itself)
        filters = []
        agc = AGCChain(logger=logger, stages=[
            Amplifier(logger=logger, gain=12.0, max_output=1.0)  # Unity gain
        ])
        desc = f"MODE 0: BEAMFORMER ONLY (unity gain) - using {beamformer_desc}"
    elif TEST_MODE == 1:
        # Baseline: Beamformer only
        filters = []
        agc_fast = AGC(logger=logger, target_rms=10.0, min_gain=1.0, max_gain=1.0,
                       attack_ms=1000.0, release_ms=1000.0, noise_floor_rms=100.0,
                       gate_gain=1.0, gate_open_ms=1000.0, gate_close_ms=1000.0, gate_hold_ms=100.0)
        agc_slow = AGC(logger=logger, target_rms=10.0, min_gain=1.0, max_gain=1.0,
                       attack_ms=1000.0, release_ms=1000.0, noise_floor_rms=100.0,
                       gate_gain=1.0, gate_open_ms=1000.0, gate_close_ms=1000.0, gate_hold_ms=100.0)
        agc = AGCChain(logger=logger, stages=[
            Amplifier(logger=logger, gain=8.0, max_output=1.0),
            TwoStageAGC(logger=logger, stage1=agc_fast, stage2=agc_slow)  # Pass-through
        ])
        desc = f"MODE 1: Beamformer ONLY (8x amplifier) - using {beamformer_desc}"
    elif TEST_MODE == 2:
        # BandPass + SLOW AGC (with gate disabled)
        filters = [
            BandPassFilter(
                logger=logger, 
                sample_rate=sample_rate, 
                low_cutoff=300.0, 
                high_cutoff=4000.0, 
                order=4),
        ]
        agc = TwoStageAGC(logger=logger, stage1=agc_fast_slow, stage2=agc_slow_slow)
        desc = "MODE 2: BandPass + SLOW AGC (gate disabled, no artifacts on loud sounds?)"
    elif TEST_MODE == 3:
        # Wiener + SLOW AGC
        filters = [
            WienerFilter(
                logger=logger,
                sample_rate=sample_rate,
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
        agc = TwoStageAGC(logger=logger, stage1=agc_fast_slow, stage2=agc_slow_slow)
        desc = "MODE 3: Wiener + SLOW AGC"
    elif TEST_MODE == 4:
        # BandPass + SpectralSubtraction (simpler noise reduction)
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
                noise_factor=0.8,
                gain_floor=0.3,
                noise_alpha=0.98,
                noise_update_snr_db=3.0,
            ),
        ]
        agc = TwoStageAGC(logger=logger, stage1=agc_fast_slow, stage2=agc_slow_slow)
        desc = "MODE 4: BandPass + Spectral Subtraction (gentle, natural voice)"
    elif TEST_MODE == 5:
        # SpectralSubtraction + PedalboardAGC (clean protection without modulation)
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
                noise_factor=0.8,
                gain_floor=0.3,
                noise_alpha=0.98,
                noise_update_snr_db=3.0,
            ),
        ]
        agc = AGCChain(logger=logger, stages=[
            Amplifier(logger=logger, gain=8.0, max_output=1.0),
            PedalboardAGC(
                logger=logger,
                sample_rate=sample_rate,
                threshold_db=-30.0,
                ratio=4.0,
                attack_ms=10.0,
                release_ms=100.0,
                limiter_threshold_db=-0.1,
                limiter_release_ms=50.0
            ),
        ])
        desc = "MODE 5: BandPass + SpectralSubtraction + PedalboardAGC (adaptive amp, natural voice)"
    elif TEST_MODE == 6:
        # Full clean pipeline: BandPass + SpectralSubtraction + PedalboardAGC
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
                noise_factor=0.8,
                gain_floor=0.3,
                noise_alpha=0.98,
                noise_update_snr_db=3.0,
            ),
        ]
        agc = AGCChain(logger=logger, stages=[
            AdaptiveAmplifier(
                logger=logger,
                target_rms=0.1,
                min_gain=1.0,
                max_gain=16.0,
                adapt_alpha=0.05,
            ),
            PedalboardAGC(
                logger=logger,
                sample_rate=sample_rate,
                threshold_db=-30.0,
                ratio=4.0,
                attack_ms=10.0,
                release_ms=100.0,
                limiter_threshold_db=-0.1,
                limiter_release_ms=50.0
            ),
        ])
        desc = "MODE 6: Full Clean Pipeline (BandPass + SpectralSub + PedalboardAGC)"
    elif TEST_MODE == 7:
        # Old mode: Full pipeline with FAST AGC (current problem - reference)
        filters = [
            BandPassFilter(
                logger=logger, 
                sample_rate=sample_rate, 
                low_cutoff=300.0, 
                high_cutoff=4000.0, 
                order=4),
            WienerFilter(
                logger=logger,
                sample_rate=sample_rate,
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
        agc = TwoStageAGC(logger=logger, stage1=agc_fast_normal, stage2=agc_slow_normal)
        desc = "MODE 7: REFERENCE - BandPass + Wiener + FAST AGC (old problem baseline)"
    elif TEST_MODE == 8:
        # Old mode: Full pipeline with SLOW AGC (Wiener reference)
        filters = [
            BandPassFilter(
                logger=logger, 
                sample_rate=sample_rate, 
                low_cutoff=300.0, 
                high_cutoff=4000.0, 
                order=4),
            WienerFilter(
                logger=logger,
                sample_rate=sample_rate,
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
        agc = TwoStageAGC(logger=logger, stage1=agc_fast_slow, stage2=agc_slow_slow)
        desc = "MODE 8: REFERENCE - BandPass + Wiener + SLOW AGC (old test)"
    else:  # TEST_MODE == 9+
        # Legacy mode: Full pipeline with AMPLIFIER + PEAK-HOLD AGC (fixed gain + adaptive limiting)
        filters = [
            BandPassFilter(
                logger=logger, 
                sample_rate=sample_rate, 
                low_cutoff=300.0, 
                high_cutoff=4000.0, 
                order=4),
            WienerFilter(
                logger=logger,
                sample_rate=sample_rate,
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
        # Chain: Amplifier (2x gain, let PeakHoldAGC handle most limiting)
        agc = AGCChain(logger=logger, stages=[
            Amplifier(logger=logger, gain=2.0, max_output=1.0),
            PeakHoldAGC(logger=logger, target_peak=0.85, min_gain=0.5, max_gain=6.0,
                       attack_ms=10.0, release_ms=500.0, peak_hold_ms=50.0)
        ])
        desc = "MODE 9+: LEGACY - BandPass + Wiener + AMPLIFIER (2x) → PEAK-HOLD AGC"
    
    codec = G711Codec(logger=logger)
    
    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mic_list,
        sampling_rate=sample_rate,
        doa_estimator=doa_estimator,
        beamformer=selected_beamformer,  # DIAGNOSTIC: Switch between MVDR and DAS
        echo_canceller=echo_canceller,
        filters=filters,
        agc=agc,
        codec=codec,
        monitor_gain=0.35,
        downsample_rate=None,
        initial_silence_duration=2.0,
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(desc)
    logger.info(f"Beamformer: {beamformer_desc}")
    logger.info("Listen for artifacts on LOUD VOICES and SIDE SOURCES")
    logger.info("Press Ctrl+C to stop")
    logger.info(f"{'='*70}\n")
    
    array.start_realtime(blocksize=2048)
    array.start_output_monitoring()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        array.stop_realtime()
        array.stop_output_monitoring()
