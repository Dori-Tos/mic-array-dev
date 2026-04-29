"""
Component Isolation Test
========================

Refactored to mirror test_pipeline.py and disable blocks cumulatively by mode.
You can switch modes at runtime with arrow keys:
- Left arrow: previous mode
- Right arrow: next mode
- q: quit

MODE 0: Full chain (MVDR + BandPass + SpectralSubtraction + AdaptiveAmp + PedalboardAGC)
MODE 1: Disable PedalboardAGC
MODE 2: Disable SpectralSubtraction
MODE 3: Disable BandPass
MODE 4: Disable Beamformer and use a single microphone channel
"""

from contextlib import contextmanager
import argparse
import logging
import os
from pathlib import Path
import select
import time

from classes.Array_RealTime import Array_RealTime
from classes.Microphone import Microphone
from classes.DOAEstimator import IterativeDOAEstimator
from classes.Beamformer import DASBeamformer, MVDRBeamformer
from classes.EchoCanceller import EchoCanceller
from classes.Filter import BandPassFilter, SpectralSubtractionFilter
from classes.AGC import AdaptiveAmplifier, AGCChain, PedalboardAGC
from classes.Codec import G711Codec

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty


TEST_MODE = 0
MIN_MODE = 0
MAX_MODE = 4

MODE_DESCRIPTIONS = {
    0: "MODE 0: Full chain active (test_pipeline equivalent)",
    1: "MODE 1: PedalboardAGC disabled",
    2: "MODE 2: PedalboardAGC + SpectralSubtraction disabled",
    3: "MODE 3: PedalboardAGC + SpectralSubtraction + BandPass disabled",
    4: "MODE 4: Beamformer disabled and single microphone input",
}


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


def _poll_control_key() -> int | None:
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
            if key2 == b"K":  # Left arrow
                return -1
            if key2 == b"M":  # Right arrow
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


def _build_mode_components(mode: int, logger: logging.Logger, script_dir: Path, sample_rate: int) -> dict:
    enable_pedalboard = mode < 1
    enable_spectral = mode < 2
    enable_bandpass = mode < 3
    enable_beamformer = mode < 4
    use_single_mic = mode >= 4

    mic_channel_numbers = [0] if use_single_mic else [0, 1, 2, 3]
    mic_list = [
        Microphone(logger=logger, channel_number=ch, sampling_rate=sample_rate)
        for ch in mic_channel_numbers
    ]

    selected_beamformer = None
    doa_estimator = None

    if enable_beamformer:
        geometry_path = script_dir / "array_geometries" / "1_Square.xml"
        mic_positions = MVDRBeamformer.load_positions_from_xml(str(geometry_path))

        das_beamformer = DASBeamformer(
            logger=logger,
            mic_channel_numbers=mic_channel_numbers,
            sample_rate=sample_rate,
            mic_positions_m=mic_positions,
        )

        selected_beamformer = MVDRBeamformer(
            logger=logger,
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
        )

        doa_estimator = IterativeDOAEstimator(
            logger=logger,
            update_rate=3.0,
            angle_range=(-25, 25),
            doa_beamformer=das_beamformer,
            scan_step_deg=5.0,
        )
        doa_estimator.freeze(0.0)

    filters = []
    if enable_bandpass:
        filters.append(
            BandPassFilter(
                logger=logger,
                sample_rate=sample_rate,
                low_cutoff=300.0,
                high_cutoff=4000.0,
                order=4,
            )
        )

    if enable_spectral:
        filters.append(
            SpectralSubtractionFilter(
                logger=logger,
                sample_rate=sample_rate,
                noise_factor=0.65,
                gain_floor=0.55,
                noise_alpha=0.995,
                noise_update_snr_db=8.0,
                gain_smooth_alpha=0.92,
            )
        )

    agc_stages: list[object] = [
        AdaptiveAmplifier(
            logger=logger,
            target_rms=0.08,
            min_gain=1.0,
            max_gain=6.0,
            adapt_alpha=0.04,
            speech_activity_rms=0.00012,
            silence_decay_alpha=0.008,
            activity_hold_ms=600.0,
            peak_protect_threshold=0.30,
            peak_protect_strength=1.0,
            max_gain_warn_rms_min=0.001,
        )
    ]

    if enable_pedalboard:
        agc_stages.append(
            PedalboardAGC(
                logger=logger,
                sample_rate=sample_rate,
                threshold_db=-20.0,
                ratio=2.0,
                attack_ms=3.0,
                release_ms=140.0,
                limiter_threshold_db=-7.0,
                limiter_release_ms=50.0,
            )
        )

    agc = AGCChain(logger=logger, stages=agc_stages)

    codec = G711Codec(logger=logger)
    echo_canceller = EchoCanceller(
        logger=logger,
        sample_rate=sample_rate,
        channels=len(mic_channel_numbers),
    )

    meta = {
        "enable_beamformer": enable_beamformer,
        "enable_bandpass": enable_bandpass,
        "enable_spectral": enable_spectral,
        "enable_pedalboard": enable_pedalboard,
        "input_channels": len(mic_channel_numbers),
    }
    return {
        "mic_list": mic_list,
        "doa_estimator": doa_estimator,
        "beamformer": selected_beamformer,
        "echo_canceller": echo_canceller,
        "filters": filters,
        "agc": agc,
        "codec": codec,
        "processing_input_channel": 0,
        "meta": meta,
    }


def _log_mode_banner(logger: logging.Logger, mode: int, meta: dict):
    logger.info("\n%s", "=" * 70)
    logger.info("%s", MODE_DESCRIPTIONS[mode])
    logger.info("Active blocks:")
    logger.info("  Beamformer: %s", "ON" if meta["enable_beamformer"] else "OFF")
    logger.info("  BandPass: %s", "ON" if meta["enable_bandpass"] else "OFF")
    logger.info("  SpectralSubtraction: %s", "ON" if meta["enable_spectral"] else "OFF")
    logger.info("  AdaptiveAmplifier: ON")
    logger.info("  PedalboardAGC: %s", "ON" if meta["enable_pedalboard"] else "OFF")
    logger.info("  Input channels: %d", meta["input_channels"])
    logger.info("Runtime controls: Left/Right arrows switch mode, q quits")
    logger.info("%s\n", "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run component isolation modes with cumulative block disabling"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=TEST_MODE,
        help="Isolation mode (0-4). Defaults to TEST_MODE constant.",
    )
    args = parser.parse_args()

    mode = int(args.mode)
    if mode < MIN_MODE or mode > MAX_MODE:
        raise ValueError(f"Unsupported mode={mode}. Supported modes: {MIN_MODE}-{MAX_MODE}")

    script_dir = Path(__file__).resolve().parent
    sample_rate = 48000

    logger = logging.getLogger("ComponentTest")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    array = None
    blocksize = 960

    try:
        mode_cfg = _build_mode_components(mode=mode, logger=logger, script_dir=script_dir, sample_rate=sample_rate)
        meta = mode_cfg["meta"]

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
            monitor_gain=0.22,
            downsample_rate=None,
            initial_silence_duration=2.0,
        )
        _log_mode_banner(logger, mode, meta)

        array.start_realtime(blocksize=blocksize)
        array.start_output_monitoring()

        with _key_capture_mode():
            while True:
                action = _poll_control_key()
                if action is None:
                    time.sleep(0.05)
                    continue

                if action == 99:
                    logger.info("Quit requested from keyboard.")
                    break

                new_mode = max(MIN_MODE, min(MAX_MODE, mode + action))
                if new_mode == mode:
                    continue

                logger.info("Switching mode: %d -> %d", mode, new_mode)

                mode_cfg = _build_mode_components(
                    mode=new_mode,
                    logger=logger,
                    script_dir=script_dir,
                    sample_rate=sample_rate,
                )
                mode = new_mode
                meta = mode_cfg["meta"]

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
                _log_mode_banner(logger, mode, meta)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if array is not None:
            try:
                array.stop_output_monitoring()
            except Exception:
                pass
            try:
                array.stop_realtime()
            except Exception:
                pass
