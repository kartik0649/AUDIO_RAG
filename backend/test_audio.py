import requests
import os
import tempfile
import wave
import numpy as np

def create_test_audio():
    """Create a simple test audio file"""
    # Create a simple sine wave
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz (A note)
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        # Read the file as bytes
        with open(tmp_file.name, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up
        os.unlink(tmp_file.name)
        
        return audio_bytes

def test_backend():
    """Test the backend with audio data"""
    url = "http://localhost:8001/query"
    
    # Create test audio
    audio_data = create_test_audio()
    print(f"Created test audio of size: {len(audio_data)} bytes")
    
    # Send request
    try:
        response = requests.post(
            url,
            data=audio_data,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Transcript: {result.get('transcript', 'No transcript')}")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Total latency: {result.get('total_latency', 0):.2f}s")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_backend() 