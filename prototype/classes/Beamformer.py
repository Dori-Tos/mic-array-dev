
class Beamformer:
    def __init__(self, mic_channel_numbers: list[int]):
        self.mic_channel_numbers = mic_channel_numbers
        self.channel_count = len(mic_channel_numbers)
    
class DASBeamformer(Beamformer):
    def __init__(self, mic_channel_numbers: list[int]):
        super().__init__(mic_channel_numbers)

class MVDRBeamformer(Beamformer):
    def __init__(self, mic_channel_numbers: list[int]):
        super().__init__(mic_channel_numbers)