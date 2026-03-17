class Microphone:
    def __init__(self, channel_number: int, sampling_rate: int):
        # Channel index in the multichannel USB stream (0-based)
        self.channel_number = channel_number
        self.sampling_rate = sampling_rate
        
    def __str__(self):
        return f"Microphone(channel={self.channel_number}, rate={self.sampling_rate}Hz)"
