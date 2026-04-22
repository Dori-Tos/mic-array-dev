from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import  IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import HighPassFilter, LowPassFilter, BandPassFilter, BandStopFilter, WienerFilter, SpectralSubtractionFilter
from classes.AGC import AGC, TwoStageAGC, Amplifier, AdaptiveAmplifier, NoiseAwareAdaptiveAmplifier, AGCChain, Limiter, PeakHoldAGC, PedalboardAGC
from classes.Codec import G711Codec, OpusCodec

import time
import os
from pathlib import Path
import logging
from contextlib import contextmanager
import select

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


MODE_FULL = 0
MODE_SINGLE = 1


@contextmanager
def _key_capture_mode():
    """Enable non-blocking key capture on POSIX; no-op on Windows."""
    if os.name != "nt" and os.isatty(0):
        fd = 0
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        yield


def _poll_mode_control_key() -> int | None:
    """
    Poll keyboard and return:
    -1 for left arrow, +1 for right arrow, 99 for quit, None for no action.
    """
    if os.name == "nt":
        if not msvcrt.kbhit():
            return None
        key = msvcrt.getch()
        if key in (b"\x00", b"\xe0") and msvcrt.kbhit():
            key2 = msvcrt.getch()
            if key2 == b"K":
                return -1
            if key2 == b"M":
                return 1
            return None
        if key in (b"q", b"Q"):
            return 99
        return None

    if not os.isatty(0):
        return None

    ready, _, _ = select.select([0], [], [], 0)
    if not ready:
        return None

    seq = os.read(0, 3)
    if seq.startswith(b"\x1b[D"):
        return -1
    if seq.startswith(b"\x1b[C"):
        return 1
    if seq in (b"q", b"Q"):
        return 99
    return None


def _build_mode_components(
    *,
    mode: int,
    logger: logging.Logger,
    doa_logger: logging.Logger,
    beamformer_logger: logging.Logger,
    echo_canceller_logger: logging.Logger,
    filter_logger: logging.Logger,
    agc_logger: logging.Logger,
    codec_logger: logging.Logger,
    sample_rate: int,
    mic_positions,
    agc,
    filters,
    codec,
):
    use_single_mic = mode == MODE_SINGLE
    mic_channel_numbers = [0] if use_single_mic else [0, 1, 2, 3]
    mic_list = [Microphone(logger=logger, channel_number=i, sampling_rate=sample_rate) for i in mic_channel_numbers]

    if use_single_mic:
        return {
            "mic_list": mic_list,
            "doa_estimator": None,
            "beamformer": None,
            "echo_canceller": EchoCanceller(logger=logger, sample_rate=sample_rate, channels=1),
            "filters": filters,
            "agc": agc,
            "codec": codec,
            "processing_input_channel": 0,
            "mode_label": "SINGLE MIC",
        }

    das_beamformer = DASBeamformer(
        logger=beamformer_logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
    )

    mvdr_beamformer = MVDRBeamformer(
        logger=beamformer_logger,
        mic_channel_numbers=mic_channel_numbers,
        sample_rate=sample_rate,
        mic_positions_m=mic_positions,
        covariance_alpha=0.95,
        diagonal_loading=0.15,
        spectral_whitening_factor=0.12,
        weight_smooth_alpha=0.72,
        max_adaptive_loading_scale=4.0,
        coherence_suppression_strength=0.8,
        weight_smooth_alpha_min=0.45,
        weight_smooth_alpha_max=0.82,
        snr_threshold_for_sharpening=2.0,
        backward_null_strength=0.9,
    )

    doa_estimator = IterativeDOAEstimator(
        logger=doa_logger,
        update_rate=3.0,
        angle_range=(-25, 25),
        beamformer=das_beamformer,
        scan_step_deg=5.0,
        local_search_radius_deg=10.0,
        periodic_full_scan_blocks=20,
    )
    # doa_estimator.freeze(0.0)

    return {
        "mic_list": mic_list,
        "doa_estimator": doa_estimator,
        "beamformer": mvdr_beamformer,
        "echo_canceller": EchoCanceller(logger=echo_canceller_logger, sample_rate=sample_rate, channels=4),
        "filters": filters,
        "agc": agc,
        "codec": codec,
        "processing_input_channel": 0,
        "mode_label": "FULL PIPELINE",
    }


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000
    downsample_rate = None  # Process at native 48kHz, no resampling artifacts
    monitor_gain = 0.22
    
    mic_channel_numbers = [0, 1, 2, 3]
    
    """
    With 4 mics => Beamforming = 4-5ms
    With 8 mics => Beamforming = 11-15ms
    
    """
        
    blocksize = 960
    # => 20ms blocks for the buffer (960/48kHz = 0.02s)
    
    logger = logging.getLogger("MicArrayTest")
    logger.setLevel(logging.INFO)
    
    doa_logger = logging.getLogger("DOAEstimator")
    doa_logger.setLevel(logging.INFO)
    
    beamformer_logger = logging.getLogger("Beamformer")
    beamformer_logger.setLevel(logging.INFO)
    
    echo_canceller_logger = logging.getLogger("EchoCanceller")
    echo_canceller_logger.setLevel(logging.INFO)
    
    filter_logger = logging.getLogger("Filters")
    filter_logger.setLevel(logging.INFO)
    
    agc_logger = logging.getLogger("AGC")
    agc_logger.setLevel(logging.INFO)
    
    codec_logger = logging.getLogger("Codec")
    codec_logger.setLevel(logging.INFO)
    
    # Create console handler and formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to all loggers
    logger.addHandler(console_handler)
    doa_logger.addHandler(console_handler)
    beamformer_logger.addHandler(console_handler)
    echo_canceller_logger.addHandler(console_handler)
    filter_logger.addHandler(console_handler)
    agc_logger.addHandler(console_handler)
    codec_logger.addHandler(console_handler)

    
    geometry_path = script_dir / "array_geometries" / "1_square.xml"
    mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))
    
    # Passband to eliminate low-frequency rumble and high-frequency hiss
    # Spectral Subtraction to supress background noise in voice region
    filter_rate = sample_rate 
    filters = [
        BandPassFilter(
            logger=filter_logger, 
            sample_rate=sample_rate, 
            low_cutoff=300.0, 
            high_cutoff=4000.0, 
            order=4),
        
        SpectralSubtractionFilter(
            logger=filter_logger,
            sample_rate=sample_rate,
            noise_factor=0.65,              # Moderate noise suppression 
            gain_floor=0.55,                # Higher floor prevents over-suppression of low freqs (was 0.35)
            noise_alpha=0.995,              # Very slow noise learning prevents formant suppression
            noise_update_snr_db=8.0,        # Can now update during low-SNR moments (protected from onset corruption)
            gain_smooth_alpha=0.92,         # Very strong uniform gain smoothing locks formants, eliminates pops
        ),
    ]
    
    agc = AGCChain(logger=logger, stages=[
        NoiseAwareAdaptiveAmplifier(
            logger=agc_logger,
            target_rms=0.08,          # Target 0.08 (-21.9 dB) for normal speech baseline
            min_gain=0.7,             # Allow attenuation to 0.7x to suppress noise/silence
            max_gain_baseline=6.0,    # Baseline max (actual will be capped by noise floor)
            gain_up_alpha=0.008,      # VERY SLOW gain increase (prevents noise chasing)
            gain_down_alpha=0.15,     # Fast gain decrease to immediately suppress peaks
            snr_threshold_db=8.0,     # Only boost when SNR > 8 dB (speech, not noise)
            noise_floor_alpha=0.997,  # Very slow noise floor adaptation (prevents jitter)
            activity_hold_ms=100.0,   # Hold 200ms after speech ends before full decay
            peak_protect_threshold=0.30,  # Reduce gain when peaks exceed 0.30
            peak_protect_strength=1.0,    # Strong peak-based damping
        ),
        
        PedalboardAGC(
            logger=agc_logger,
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
            logger=codec_logger,
            bitrate=24000,
            frame_duration_ms=10,
            application="voip",
            remote_host="172.98.1.59",
            remote_port=5004,
        )
    else:
        # In local mode, use a lightweight codec object; codec transport is not used.
        codec = G711Codec(logger=codec_logger)

    mode = MODE_FULL
    mode_cfg = _build_mode_components(
        mode=mode,
        logger=logger,
        doa_logger=doa_logger,
        beamformer_logger=beamformer_logger,
        echo_canceller_logger=echo_canceller_logger,
        filter_logger=filter_logger,
        agc_logger=agc_logger,
        codec_logger=codec_logger,
        sample_rate=sample_rate,
        mic_positions=mic_positions,
        agc=agc,
        filters=filters,
        codec=codec,
    )

    array = Array_RealTime(
        id_vendor=0x2752,
        id_product=0x0019,
        logger=logger,
        mic_list=mode_cfg["mic_list"],
        sampling_rate=sample_rate,
        doa_estimator=mode_cfg["doa_estimator"],
        beamformer=mode_cfg["beamformer"],
        echo_canceller=mode_cfg["echo_canceller"],
        filters=mode_cfg["filters"],
        agc=mode_cfg["agc"],
        codec=mode_cfg["codec"],
        monitor_gain=monitor_gain,
        output_mode=output_mode,
        output_boundary_fade_ms=0.0,
        downsample_rate=downsample_rate,
        initial_silence_duration=2.0,  # Silence period for baseline learning (filter protects against corruption during onset)
    )

    array.start_realtime(blocksize=blocksize)
    array.start_output_monitoring()

    # Timing logging
    timing_log_interval = 2.0  # Log timing every 2 seconds
    last_timing_log = time.time()

    try:
        print("Realtime monitoring started.")
        print("Right arrow: switch to SINGLE MIC, Left arrow: switch to FULL PIPELINE, q: quit")
        with _key_capture_mode():
            while True:
                action = _poll_mode_control_key()
                if action is None:                    
                    time.sleep(0.05)
                    continue

                if action == 99:
                    print("Quit requested from keyboard.")
                    break

                target_mode = mode
                if action == 1:
                    target_mode = MODE_SINGLE
                elif action == -1:
                    target_mode = MODE_FULL

                if target_mode == mode:
                    continue

                mode_cfg = _build_mode_components(
                    mode=target_mode,
                    logger=logger,
                    sample_rate=sample_rate,
                    mic_positions=mic_positions,
                    agc=agc,
                    filters=filters,
                    codec=codec,
                    doa_logger=doa_logger,
                    beamformer_logger=beamformer_logger,
                    echo_canceller_logger=echo_canceller_logger,
                    filter_logger=filter_logger,
                    agc_logger=agc_logger,
                    codec_logger=codec_logger,
                )

                print(f"Switching mode: {mode_cfg['mode_label']}")
                array.reconfigure_runtime(
                    mic_list=mode_cfg["mic_list"],
                    doa_estimator=mode_cfg["doa_estimator"],
                    beamformer=mode_cfg["beamformer"],
                    echo_canceller=mode_cfg["echo_canceller"],
                    filters=mode_cfg["filters"],
                    agc=mode_cfg["agc"],
                    processing_input_channel=mode_cfg["processing_input_channel"],
                    restart_if_running=True,
                    blocksize=blocksize,
                    restore_output_monitoring=True,
                )
                mode = target_mode

    except KeyboardInterrupt:
        print("\nStopping...")
        
    finally:
        array.stop_realtime()
        array.stop_output_monitoring()