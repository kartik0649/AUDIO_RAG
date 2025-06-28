#!/usr/bin/env python3
"""
Debug script to analyze audio data from frontend
"""

import os
import tempfile
import base64
from audio_processor import detect_audio_format, decode_audio_data

def analyze_audio_data(audio_data: bytes):
    """Analyze audio data to understand its format"""
    print("Audio Data Analysis")
    print("=" * 40)
    
    print(f"Data size: {len(audio_data)} bytes")
    print(f"First 20 bytes: {audio_data[:20]}")
    print(f"Last 20 bytes: {audio_data[-20:]}")
    
    # Check if it's base64 encoded
    try:
        decoded = base64.b64decode(audio_data)
        print(f"✓ Base64 decoded successfully: {len(decoded)} bytes")
        print(f"Decoded first 20 bytes: {decoded[:20]}")
        audio_data = decoded
    except:
        print("✗ Not base64 encoded")
    
    # Detect format
    format_detected = detect_audio_format(audio_data)
    print(f"Detected format: {format_detected}")
    
    # Check for common patterns
    if audio_data.startswith(b'data:audio/'):
        print("✓ Data URL format detected")
        # Extract the actual audio data
        header_end = audio_data.find(b',')
        if header_end != -1:
            audio_data = audio_data[header_end + 1:]
            print(f"Extracted audio data: {len(audio_data)} bytes")
            try:
                decoded = base64.b64decode(audio_data)
                print(f"✓ Data URL decoded: {len(decoded)} bytes")
                audio_data = decoded
            except:
                print("✗ Failed to decode data URL")
    
    # Save to file for inspection
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
    temp_path = temp_file.name
    temp_file.write(audio_data)
    temp_file.close()
    
    print(f"Saved raw data to: {temp_path}")
    print(f"File size: {os.path.getsize(temp_path)} bytes")
    
    return temp_path, audio_data

def test_with_sample_data():
    """Test with sample audio data"""
    print("\nTesting with sample data...")
    
    # Create a simple WAV file
    import wave
    import numpy as np
    
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = temp_file.name
    temp_file.close()
    
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Read the WAV file as bytes
    with open(wav_path, 'rb') as f:
        wav_bytes = f.read()
    
    print(f"Sample WAV file: {len(wav_bytes)} bytes")
    analyze_audio_data(wav_bytes)
    
    # Clean up
    os.remove(wav_path)

if __name__ == "__main__":
    test_with_sample_data() 