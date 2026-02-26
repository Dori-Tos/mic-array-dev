from custom_tuning import Tuning
import usb.core
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

def lock_doa(angle=0):
    if dev:
        Mic_tuning = Tuning(dev)
        Mic_tuning.lock_doa
        print(Mic_tuning.direction)
        while True:
            try:
                print(Mic_tuning.direction)
                print(Mic_tuning.is_voice)
                time.sleep(1)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    lock_doa(angle=0)
    
    if KeyboardInterrupt:
        if dev:
            Mic_tuning = Tuning(dev)
            Mic_tuning.unlock_doa