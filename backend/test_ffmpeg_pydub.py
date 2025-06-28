#!/usr/bin/env python3
"""
Test script to verify FFmpeg and pydub integration
"""

import os
import tempfile
import subprocess
from audio_processor import configure_ffmpeg_path, configure_pydub, FFMPEG_PATH

def test_ffmpeg_direct():
    """Test FFmpeg directly"""
    print("Testing FFmpeg directly...")
    try:
        result = subprocess.run([FFMPEG_PATH, '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg works directly")
            return True
        else:
            print("✗ FFmpeg direct test failed")
            return False
    except Exception as e:
        print(f"✗ FFmpeg direct test failed: {e}")
        return False

def test_pydub_ffmpeg():
    """Test pydub with FFmpeg"""
    print("Testing pydub with FFmpeg...")
    try:
        # Configure pydub
        if not configure_pydub():
            print("✗ Failed to configure pydub")
            return False
        
        from pydub import AudioSegment
        
        # Create a simple test audio file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        test_wav_path = temp_file.name
        temp_file.close()
        
        # Create a simple sine wave using numpy
        import numpy as np
        sample_rate = 16000
        duration = 1
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save as WAV
        import wave
        with wave.open(test_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Test pydub loading
        audio_segment = AudioSegment.from_file(test_wav_path)
        print(f"✓ Pydub loaded audio: {len(audio_segment)} ms")
        
        # Test pydub export
        output_path = test_wav_path.replace(".wav", "_converted.wav")
        audio_segment.export(output_path, format="wav")
        
        if os.path.exists(output_path):
            print("✓ Pydub export successful")
            os.remove(output_path)
            os.remove(test_wav_path)
            return True
        else:
            print("✗ Pydub export failed")
            return False
            
    except Exception as e:
        print(f"✗ Pydub test failed: {e}")
        return False

def main():
    print("FFmpeg and Pydub Integration Test")
    print("=" * 40)
    
    # Test 1: FFmpeg direct
    ffmpeg_ok = test_ffmpeg_direct()
    
    # Test 2: Pydub with FFmpeg
    pydub_ok = test_pydub_ffmpeg()
    
    print("\n" + "=" * 40)
    if ffmpeg_ok and pydub_ok:
        print("✓ All tests passed! FFmpeg and pydub are working correctly.")
    else:
        print("✗ Some tests failed. Check the configuration.")
    
    print(f"FFmpeg path: {FFMPEG_PATH}")

if __name__ == "__main__":
    main() 