#!/usr/bin/env python3
"""
Test script to verify audio processing fixes
"""

import os
import tempfile
import wave
import numpy as np
from audio_processor import AudioProcessor, process_audio_for_whisper, decode_audio_data

def create_test_wav():
    """Create a simple test WAV file"""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 1  # 1 second
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = temp_file.name
    temp_file.close()
    
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return wav_path

def test_audio_processing():
    """Test the audio processing functions"""
    print("Testing Audio Processing Fixes")
    print("=" * 40)
    
    # Test 1: Create a test WAV file
    print("\n1. Creating test WAV file...")
    test_wav_path = create_test_wav()
    print(f"✓ Created test WAV: {test_wav_path}")
    
    # Test 2: Read the WAV file as bytes
    print("\n2. Reading WAV file as bytes...")
    with open(test_wav_path, 'rb') as f:
        audio_bytes = f.read()
    print(f"✓ Read {len(audio_bytes)} bytes")
    
    # Test 3: Test decode_audio_data function
    print("\n3. Testing decode_audio_data...")
    decoded_bytes = decode_audio_data(audio_bytes)
    print(f"✓ Decoded {len(decoded_bytes)} bytes")
    
    # Test 4: Test process_audio_for_whisper
    print("\n4. Testing process_audio_for_whisper...")
    try:
        temp_path, converted_path = process_audio_for_whisper(audio_bytes)
        print(f"✓ Processed audio successfully")
        print(f"  - Temp path: {temp_path}")
        print(f"  - Converted path: {converted_path}")
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)
            
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
    
    # Test 5: Test base64 encoding/decoding
    print("\n5. Testing base64 encoding/decoding...")
    import base64
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
    decoded_base64 = decode_audio_data(base64_audio)
    print(f"✓ Base64 test: {len(decoded_base64)} bytes decoded")
    
    # Clean up test file
    if os.path.exists(test_wav_path):
        os.remove(test_wav_path)
    
    print("\n" + "=" * 40)
    print("Audio processing tests completed!")

if __name__ == "__main__":
    test_audio_processing() 