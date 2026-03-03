from custom_tuning import Tuning
import usb.core
import sys


def activate_stationary_noise_suppression():
    """Activate stationary noise suppression (enable noise reduction of static background noise)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('STATNOISEONOFF', 1)
    status = tuning.read('STATNOISEONOFF')
    tuning.close()
    
    if status == 1:
        print("✓ Stationary noise suppression ACTIVATED (static background noise reduction enabled)")
        return True
    else:
        print("✗ Failed to activate stationary noise suppression")
        return False


def deactivate_stationary_noise_suppression():
    """Deactivate stationary noise suppression (disable noise reduction of static background noise)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('STATNOISEONOFF', 0)
    status = tuning.read('STATNOISEONOFF')
    tuning.close()
    
    if status == 0:
        print("✓ Stationary noise suppression DEACTIVATED (static background noise reduction disabled)")
        return True
    else:
        print("✗ Failed to deactivate stationary noise suppression")
        return False


def get_stationary_noise_suppression_status():
    """Check current stationary noise suppression status"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return None
    
    tuning = Tuning(mic_array)
    status = tuning.read('STATNOISEONOFF')
    tuning.close()
    
    if status == 1:
        print("Stationary noise suppression status: ACTIVATED")
    else:
        print("Stationary noise suppression status: DEACTIVATED")
    
    return status

def activate_non_stationary_noise_suppression():
    """Activate non-stationary noise suppression (enable noise reduction of dynamic background noise)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('NONSTATNOISEONOFF', 1)
    status = tuning.read('NONSTATNOISEONOFF')
    tuning.close()
    
    if status == 1:
        print("✓ Non-stationary noise suppression ACTIVATED (dynamic background noise reduction enabled)")
        return True
    else:
        print("✗ Failed to activate non-stationary noise suppression")
        return False
    
def deactivate_non_stationary_noise_suppression():
    """Deactivate non-stationary noise suppression (disable noise reduction of dynamic background noise)"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    tuning = Tuning(mic_array)
    tuning.write('NONSTATNOISEONOFF', 0)
    status = tuning.read('NONSTATNOISEONOFF')
    tuning.close()
    
    if status == 0:
        print("✓ Non-stationary noise suppression DEACTIVATED (dynamic background noise reduction disabled)")
        return True
    else:
        print("✗ Failed to deactivate non-stationary noise suppression")
        return False
    
def get_non_stationary_noise_suppression_status():
    """Check current non-stationary noise suppression status"""
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return None
    
    tuning = Tuning(mic_array)
    status = tuning.read('NONSTATNOISEONOFF')
    tuning.close()
    
    if status == 1:
        print("Non-stationary noise suppression status: ACTIVATED")
    else:
        print("Non-stationary noise suppression status: DEACTIVATED")
    
    return status

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python controle_noise_suppression.py activate_stationary    - Activate stationary noise suppression")
        print("  python controle_noise_suppression.py deactivate_stationary  - Deactivate stationary noise suppression")
        print("  python controle_noise_suppression.py status_stationary      - Check current stationary noise suppression status")
        print("  python controle_noise_suppression.py activate_non_stationary    - Activate non-stationary noise suppression")
        print("  python controle_noise_suppression.py deactivate_non_stationary  - Deactivate non-stationary noise suppression")
        print("  python controle_noise_suppression.py status_non_stationary      - Check current non-stationary noise suppression status")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'activate_stationary':
        activate_stationary_noise_suppression()
    elif command == 'deactivate_stationary':
        deactivate_stationary_noise_suppression()
    elif command == 'status_stationary':
        get_stationary_noise_suppression_status()
    elif command == 'activate_non_stationary':
        activate_non_stationary_noise_suppression()
    elif command == 'deactivate_non_stationary':
        deactivate_non_stationary_noise_suppression()
    elif command == 'status_non_stationary':
        get_non_stationary_noise_suppression_status()
    else:
        print(f"Unknown command: {command}")
        print("Use: activate_stationary, deactivate_stationary, status_stationary, activate_non_stationary, deactivate_non_stationary, or status_non_stationary")
        sys.exit(1)
