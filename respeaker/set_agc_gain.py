#!/usr/bin/env python3
"""
Set the AGC gain of the ReSpeaker 4 mic array.
"""

from custom_tuning import Tuning
import usb.core
import argparse
import math


def set_agc_gain(gain_value):
    """
    Set the AGC gain of the ReSpeaker array.
    
    Args:
        gain_value: AGC gain value (typical range: 1.0 to 30.0)
                    Higher values = more amplification for quiet signals
    """
    # Find the Respeaker 4 mic array USB device
    mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    
    if not mic_array:
        print("ERROR: Respeaker 4 mic array not found")
        return False
    
    try:
        tuning = Tuning(mic_array)
        
        # Read current gain
        current_gain = tuning.read('AGCGAIN')
        current_gain_db = 20 * math.log10(current_gain) if current_gain and current_gain > 0 else 0
        
        print(f"Current AGC Gain: {current_gain:.2f} ({current_gain_db:.1f} dB)")
        
        # Set new gain
        tuning.write('AGCGAIN', gain_value)
        
        # Verify the change
        new_gain = tuning.read('AGCGAIN')
        new_gain_db = 20 * math.log10(new_gain) if new_gain and new_gain > 0 else 0
        
        print(f"New AGC Gain:     {new_gain:.2f} ({new_gain_db:.1f} dB)")
        
        tuning.close()
        
        print("\nAGC gain updated successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to set AGC gain: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set ReSpeaker AGC gain')
    parser.add_argument('gain', type=float, help='AGC gain value (typical: 1.0 to 30.0)')
    parser.add_argument('--show-only', action='store_true', help='Only show current gain without changing')
    
    args = parser.parse_args()
    
    # Typical AGC gains:

    #   1.0 (0 dB) = No additional gain
    #   2.6 (8 dB) = Initial setup (for close source)
    #   10.0 (20 dB) = Moderate gain for 1m distance
    #   20.0 (26 dB) = High gain for distant/quiet sources
    #   30.0 (30 dB) = Maximum recommended
    
    if args.show_only:
        # Just show current gain
        mic_array = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if mic_array:
            tuning = Tuning(mic_array)
            current_gain = tuning.read('AGCGAIN')
            current_gain_db = 20 * math.log10(current_gain) if current_gain and current_gain > 0 else 0
            print(f"Current AGC Gain: {current_gain:.2f} ({current_gain_db:.1f} dB)")
            tuning.close()
        else:
            print("ERROR: Respeaker 4 mic array not found")
    else:
        set_agc_gain(args.gain)
