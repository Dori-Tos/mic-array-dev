from custom_tuning import Tuning
import usb.core
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if dev:
    Mic_tuning = Tuning(dev)
    print(Mic_tuning.direction)
    while True:
        try:
            print(Mic_tuning.direction)
            print(Mic_tuning.is_voice)
            time.sleep(1)
        except KeyboardInterrupt:
            break