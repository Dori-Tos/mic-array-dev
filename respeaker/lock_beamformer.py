from custom_tuning import Tuning
import usb.core
import sys


def lock_beamformer():
    """Lock the adaptive beamformer (freeze DOA)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('FREEZEONOFF', 1)
    status = tuning.read('FREEZEONOFF')
    tuning.close()
    
    if status == 1:
        print("✓ Beamformer LOCKED (adaptive updates disabled)")
        return True
    else:
        print("✗ Failed to lock beamformer")
        return False


def unlock_beamformer():
    """Unlock the adaptive beamformer (enable DOA adaptation)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('FREEZEONOFF', 0)
    status = tuning.read('FREEZEONOFF')
    tuning.close()
    
    if status == 0:
        print("✓ Beamformer UNLOCKED (adaptive updates enabled)")
        return True
    else:
        print("✗ Failed to unlock beamformer")
        return False


def get_beamformer_status():
    """Check current beamformer lock status"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return None
    
    tuning = Tuning(mic_array)
    status = tuning.read('FREEZEONOFF')
    tuning.close()
    
    if status == 1:
        print("Beamformer status: LOCKED (frozen)")
    else:
        print("Beamformer status: UNLOCKED (adaptive)")
    
    return status

def get_doa_angle():
    """Get current DOA angle"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return None
    
    tuning = Tuning(mic_array)
    doa_angle = tuning.read('DOAANGLE')
    tuning.close()
    
    if doa_angle is not None:
        print(f"Current DOA Angle: {doa_angle}°")
        return doa_angle
    else:
        print("ERROR: Failed to read DOA angle")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python lock_beamformer.py lock    - Lock the beamformer")
        print("  python lock_beamformer.py unlock  - Unlock the beamformer")
        print("  python lock_beamformer.py status  - Check current status")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'lock':
        lock_beamformer()
    elif command == 'unlock':
        unlock_beamformer()
    elif command == 'status':
        get_beamformer_status()
    elif command == 'get_doa':
        get_doa_angle()
    else:
        print(f"Unknown command: {command}")
        print("Use: lock, unlock, status, or get_doa")
        sys.exit(1)
