import pyaudio

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, int(numdevices)):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
            
            
# Output:
# Input Device id  0  -  Mappeur de sons Microsoft - Input
# Input Device id  1  -  ReSpeaker 4 Mic Array (UAC1.0) 
# Input Device id  2  -  Microphone (Realtek(R) Audio)
# Input Device id  3  -  Casque (Hesh Evo)