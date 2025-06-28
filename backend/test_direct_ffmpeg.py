#!/usr/bin/env python3
"""
Test direct FFmpeg conversion
"""

import os
import tempfile
import wave
import numpy as np
from audio_processor import convert_with_ffmpeg_direct, FFMPEG_PATH

def create_test_mp4():
    """Create a simple test file that looks like MP4"""
    # Create a simple audio file first
    sample_rate = 16000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create a WAV file first
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = temp_file.name
    temp_file.close()
    
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Convert WAV to MP4 using FFmpeg
    mp4_path = wav_path.replace(".wav", ".mp4")
    cmd = [
        FFMPEG_PATH,
        '-i', wav_path,
        '-c:a', 'aac',
        '-y',
        mp4_path
    ]
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(mp4_path):
        os.remove(wav_path)  # Clean up WAV file
        return mp4_path
    else:
        # If FFmpeg conversion fails, just return the WAV file
        return wav_path

def test_direct_ffmpeg():
    """Test direct FFmpeg conversion"""
    print("Testing Direct FFmpeg Conversion")
    print("=" * 40)
    
    # Create test file
    print("1. Creating test audio file...")
    test_file = create_test_mp4()
    print(f"✓ Created test file: {test_file}")
    
    # Test direct FFmpeg conversion
    print("\n2. Testing direct FFmpeg conversion...")
    try:
        converted_path = convert_with_ffmpeg_direct(test_file, "wav")
        if converted_path and os.path.exists(converted_path):
            print(f"✓ Direct FFmpeg conversion successful: {converted_path}")
            print(f"  - File size: {os.path.getsize(converted_path)} bytes")
            
            # Clean up
            os.remove(converted_path)
            return True
        else:
            print("✗ Direct FFmpeg conversion failed")
            return False
    except Exception as e:
        print(f"✗ Direct FFmpeg conversion error: {e}")
        return False
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    success = test_direct_ffmpeg()
    print(f"\n{'=' * 40}")
    if success:
        print("✓ Direct FFmpeg conversion test passed!")
    else:
        print("✗ Direct FFmpeg conversion test failed!") 