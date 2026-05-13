# Respeaker Mic Array

This project implements a real-time audio processing pipeline for a Microphone Array prototype, it also includes small elements for the USB 4-mic array from Seeed Studio. 
The main focus is the prototype folder which includes:
- `classes/`: Core classes for real-time audio processing, these classes handle microphone input, beamforming, post-beamforming processing, and output management.
- `test_pipeline.py`: A test script that sets up and runs the audio processing pipeline.
- `test_protocol/`: A folder containing test scripts for specific functionalities, such as generating polar patterns and saving outputs.

## Classes
- `Array_RealTime`: This class manages the real-time audio processing pipeline, including microphone input, beamforming, post-beamforming processing, and output handling. It allows for configuration of various parameters such as sample rate, block size, and various real-time options. This class is bloated as it includes all the output and thread management, this would be a good candidate for refactoring into smaller, more focused classes.
- `Beamformer`: This class is responsible for performing beamforming on the input audio data (1st stage). It takes into account the geometry of the microphone array and the desired beamforming angle to produce a focused audio output. It proposes a simple delay-and-sum beamforming approach, and a more complex MVDR beamforming approach, the latter is more computationally intensive and can cause artifacts due to estimation updates but its performance are superior to the DAS.
- `Filter`: This class is responsible for applying various audio filters to the beamformed data (2nd stage), such as low-pass, high-pass, and band-pass filters. It allows for configuration of filter parameters and provides a simple interface for applying filters to the audio data. It also inmplements two adaptive filters, a spectral filter that estimates noise in the spectral domain and applies a gain reduction, and a wiener filter that estimates the clean signal power and applies a gain reduction based on the SNR estimation. The first is more stable and has less impact on the audio quality, the second should in theory provide better noise reduction but its current implementation can be upgraded. The filters are intended to be used in a list so that each one feeds into the next.
- `AGC`: This class is responsible for implementing Automatic Gain Control (AGC) on the processed audio data (3rd stage). It offers multiple AGCs, including a simple fixed amplifier, a peak-hold limiter, a simpler limiter, a noise aware adaptive amplifier that adjusts gain based on noise estimation. Another adaptive amplifier is present in the PedalBoardAGC that is based on Spotify's Pedalboard library, it is much more agressive and thus more adapted to fast changing audio compared to the noise-aware AGC. A legacy AdaptiveAmplifier class is also present. All the AGCs are meant to be used in an AGCChain that implements them one after the other, this allows to use multiple AGCs on the same audio stream.
- `Codec`: This class is responsible for encoding and decoding audio data for transmission over a codec stream. It provides a simple interface for applying codec-specific processing to the audio data. Its current implementation only supports Opus and its library, the G711 implementation is only a placeholder and is not functional.
- `Microphone`: This class is only used to represent a microphone object with its channel number. Future implementations could maybe include the microphone's position to lighten the work of the beamformer and detail the microphone's metadata.
- `EchoCanceller`: This class is responsible for implementing echo cancellation on the processed audio data. But as it was an optional feature, this is only a placeholder and is not functional. The initial plan was to implement WebRTC or an equivalent in this class to provide a simple echo canceller against the Larsen effect and a second more complex element for room echo cancellation.
- `Array_SingleMic`: This class offers a simplified version of Array_RealTime for single microphone pipeline without beamforming, it is meant to represent the Prototype's "Alert Mode".

## Array Geometries
The geometry files are placed in a specific folders and are all xml files.

- `1_Square.xml`: This file describes a square geometry with 4 microphones, it is meant to serve as the baseline for the rest. It is the only one were the microphones are placed on the smaller frame (1st prototype).
- `2_Corners.xml`: This file describes a geometry with 8 microphones placed on the corners of the larger frame (2nd prototype), it is used as a middle ground between the smaller square and the larger arrays. Its performances are similar to the larger 14 mic array.
- `3_Rim.xml`: This file describes a geometry with 10 microphones, placed on the rim of the rectangle shaped frame. It was meant to be the larger element until the fifth pattern was added.
- `4_Single_Corner.xml`: This file describes a geometry with 4 microphones placed on a single corner of the frame, it was developped to test the use case were only 4 mics are used for the whole array on the large frame. Its performances are lacking compared to the other arrays as its asymmetric nature doesn't allow for easy angle estimation and beamforming.
- `5_Max_Rim.xml`: This file describes a geometry with 14 microphones, all placed on the rim of the 2nd prototype. It boasts the best performance in terms of directivity and noise reduction but is also the most computationally intensive. With the current pipeline it is also the most prone to artifacts due to the stronger effect of coherence estimations with the MVDR beamformer, improving the stability of the MVDR would make this pattern the best in all aspects.

## Test Files
Without CLI control:
- `test_pipeline.py`: This script sets up and runs the real-time audio processing pipeline using the `Array_RealTime` class. It uses the most efficient parameters for each class as tweaking each object can be quite time consuming. It allows to switch between free and frozen doa estimation before runtime and between beamformed or single mic output during runtime (right arrow).
- `test_single_mic.py`: This script allows to test the output of each mic connected to the MCH Streamer Kit individually, to switch between mics CTRL+C must be used AFTER the settling period (2s), as pressing these keys during this moment shuts down the scipt.

With CLI control:
- `0_Evaluate_Gain.py`: This script evaluates the gain of the pipeline, it may be outdated due to changes in Array_RealTime. The older implementation is still there under the name `test_protocol/0_Gain_Drift.py`.
- `1_Polar_Pattern.py`: This script measures the input rms, output rms, and gain across different angles, these informations are then saved in a CSV file used to both plot the polar pattern and generate frequency response plots.
- `1_Plot_Directivity.py` & `1_Plot_Frequency_Response.py`: These scripts plot the polar pattern and frequency response of the beamformer based on the CSV file generated by `1_Polar_Pattern.py`.
- The DI files are outdated and proved to be unusable in non-anechoic environments, if perfected and tested in a controlled room thy may proved to be useful to compare between array results.
-`3_Free_Beam_Loss.py` & `3_Plot_Free_Beam_Loss.py`: These scripts evaluate the gain loss when the beamformer is free to follow the DOA estimations. The intended use is to put the array at different angle, not in a continuous rotation, as the estimations take some time to stabilize. 
-`4_Save_Output.py`: This script uses the `Array_RealTime` class to save the output of the pipeline to a WAV file, it also allows to listen to the output in real-time.

## Miscellaneous
Other interesting repositories:
- https://github.com/respeaker/usb_4_mic_array.git => Base implementation for the reSpeaker USB 4-mic array
- https://github.com/introlab/odas.git => Could be interesting for 3D audio mapping, supported by the ReSpeaker Mic Array

