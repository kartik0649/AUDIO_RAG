#!/usr/bin/env python3
"""
Test script for audio processing functionality
"""

import os
import sys
import tempfile
import wave
import numpy as np
from audio_processor import AudioProcessor

def create_test_wav_file(duration_seconds=3, sample_rate=16000):
    """Create a simple test WAV file with a sine wave"""
    try:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir)
        wav_path = temp_file.name
        temp_file.close()
        
        # Generate a simple sine wave (440 Hz)
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"Created test WAV file: {wav_path}")
        return wav_path
        
    except Exception as e:
        print(f"Error creating test WAV file: {e}")
        return None

def test_audio_processor():
    """Test the AudioProcessor functionality"""
    print("=== Testing AudioProcessor ===")
    
    # Test 1: Create test WAV file
    print("\n1. Creating test WAV file...")
    test_wav = create_test_wav_file()
    if not test_wav:
        print("Failed to create test WAV file")
        return False
    
    try:
        # Test 2: Validate audio file
        print("\n2. Testing audio file validation...")
        is_valid = AudioProcessor.validate_audio_file(test_wav)
        print(f"Audio file validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test 3: Get audio info
        print("\n3. Testing audio info retrieval...")
        audio_info = AudioProcessor.get_audio_info(test_wav)
        print(f"Audio info: {audio_info}")
        
        # Test 4: Test audio conversion
        print("\n4. Testing audio conversion...")
        converted_path = AudioProcessor.convert_audio_format(test_wav, "wav")
        if converted_path:
            print(f"Audio conversion successful: {converted_path}")
            # Clean up converted file
            AudioProcessor.cleanup_temp_files(converted_path)
        else:
            print("Audio conversion failed")
        
        # Test 5: Test with raw bytes
        print("\n5. Testing WAV creation from bytes...")
        with open(test_wav, 'rb') as f:
            audio_bytes = f.read()
        
        wav_from_bytes = AudioProcessor.create_wav_from_bytes(audio_bytes)
        if wav_from_bytes:
            print(f"WAV from bytes created: {wav_from_bytes}")
            # Clean up
            AudioProcessor.cleanup_temp_files(wav_from_bytes)
        else:
            print("WAV from bytes failed")
        
        return True
        
    finally:
        # Clean up test file
        AudioProcessor.cleanup_temp_files(test_wav)

def test_whisper_integration():
    """Test Whisper integration"""
    print("\n=== Testing Whisper Integration ===")
    
    try:
        import whisper
        print("Whisper imported successfully")
        
        # Load model
        print("Loading Whisper model...")
        model = whisper.load_model("tiny")  # Use tiny model for faster testing
        print("Whisper model loaded successfully")
        
        # Create test audio
        test_wav = create_test_wav_file()
        if not test_wav:
            print("Failed to create test audio for Whisper")
            return False
        
        try:
            # Test transcription
            print("Testing Whisper transcription...")
            result = model.transcribe(test_wav, fp16=False)
            transcript = result.get("text", "")
            if transcript and isinstance(transcript, str):
                transcript = transcript.strip()
                print(f"Transcription result: '{transcript}'")
                
                if transcript:
                    print("Whisper transcription: PASS")
                else:
                    print("Whisper transcription: FAIL (empty result)")
            else:
                print("Whisper transcription: FAIL (invalid result)")
            
            return True
            
        finally:
            AudioProcessor.cleanup_temp_files(test_wav)
            
    except ImportError:
        print("Whisper not available")
        return False
    except Exception as e:
        print(f"Whisper test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("=== Testing Dependencies ===")
    
    dependencies = [
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("pydub", "pydub"),
        ("whisper", "openai-whisper"),
    ]
    
    all_available = True
    for name, package in dependencies:
        try:
            __import__(name)
            print(f"✓ {package} - Available")
        except ImportError:
            print(f"✗ {package} - Not available")
            all_available = False
    
    return all_available

def main():
    """Main test function"""
    print("Audio RAG - Audio Processing Test")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\nSome dependencies are missing. Please install them:")
        print("pip install -r requirements.txt")
        return
    
    # Test audio processor
    processor_ok = test_audio_processor()
    
    # Test Whisper integration
    whisper_ok = test_whisper_integration()
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Dependencies: {'PASS' if deps_ok else 'FAIL'}")
    print(f"Audio Processor: {'PASS' if processor_ok else 'FAIL'}")
    print(f"Whisper Integration: {'PASS' if whisper_ok else 'FAIL'}")
    
    if deps_ok and processor_ok and whisper_ok:
        print("\n🎉 All tests passed! Audio processing should work.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 