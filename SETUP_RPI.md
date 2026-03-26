# Raspberry Pi Setup Guide

## Problem
`python3-gi` and GStreamer packages are **system packages**, not pip packages. On Python 3.12, they must be installed via `apt-get`, not `pip install`.

## Solution

### Automatic Setup (Recommended)
Run the automated setup script:

```bash
bash setup_rpi.sh
```

This script will:
- Update system packages
- Install Python 3.12 development headers
- Install GStreamer and audio libraries
- Create a virtual environment
- Install pip dependencies

### Manual Setup

If you prefer manual installation:

#### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get upgrade -y

# Install Python development headers for 3.12
sudo apt-get install -y python3.12-dev python3.12-venv

# Install GStreamer and GObject dependencies
sudo apt-get install -y \
    python3-gi \
    python3-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-tools

# Install audio and media libraries
sudo apt-get install -y \
    libasound2-dev \
    libatlas-base-dev \
    libjasper-dev \
    libopenjp2-7 \
    libtiff5 \
    libtiffxx5
```

#### 2. Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Python Packages
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'gi'`
**Cause**: GObject Introspection not installed as system package
**Solution**: 
```bash
sudo apt-get install -y python3-gi gir1.2-gstreamer-1.0
```

### Issue: `ImportError: libgstreamer-1.0.so.0`
**Cause**: GStreamer development files missing
**Solution**:
```bash
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### Issue: `audio device not found`
**Cause**: Audio libraries or USB permissions missing
**Solutions**:
```bash
# Install audio support
sudo apt-get install -y pulseaudio alsa-utils

# Add user to audio group for USB device access
sudo usermod -a -G audio $USER
sudo usermod -a -G dialout $USER

# Log out and log back in for group changes to take effect
```

### Issue: `requirements.txt` still has old system packages
**Solution**: Use the new separated files:
- `requirements.txt` - pip packages only
- `requirements-system.txt` - system packages (for reference)

## File Structure

```
requirements.txt          # pip packages (numpy, scipy, sounddevice, etc.)
requirements-system.txt   # system packages (documentation only)
setup_rpi.sh              # Automated setup script
README.md                 # This file
```

## Quick Start (After Setup)

```bash
# Activate environment
source .venv/bin/activate

# Run the microphone array test
python Python/Tests/mic-array-dev/prototype/test_array.py
```

## Notes

- Python 3.12 is recommended (3.11 works but may have compatibility issues)
- Virtual environment keeps system Python clean
- System packages are installed globally with `sudo`
- Virtual environment packages are user-local

## Alternative: Docker

If setting up directly on RPi is problematic, consider using Docker:

```dockerfile
FROM raspios:latest
RUN apt-get update && apt-get install -y python3.12 python3-gi gstreamer1.0-plugins-base
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
```
