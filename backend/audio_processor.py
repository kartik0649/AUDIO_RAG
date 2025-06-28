import os
import tempfile
import pathlib
import wave
import struct
import numpy as np
from typing import Optional, Tuple, Union
import logging
import subprocess
import base64

# Set FFmpeg environment variables early to avoid pydub warnings
os.environ["FFMPEG_BINARY"] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = "C:\\ffmpeg\\bin\\ffprobe.exe"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FFmpeg configuration
FFMPEG_PATH = None

def configure_ffmpeg_path():
    """Configure FFmpeg path for the system"""
    global FFMPEG_PATH
    
    # Common FFmpeg installation paths on Windows
    possible_paths = [
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
        "ffmpeg"  # Try PATH
    ]
    
    for path in possible_paths:
        try:
            if path == "ffmpeg":
                # Test if ffmpeg is in PATH
                result = subprocess.run([path, "-version"], 
                                      capture_output=True, text=True, timeout=5)
            else:
                # Test full path
                result = subprocess.run([path, "-version"], 
                                      capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                FFMPEG_PATH = path
                logger.info(f"FFmpeg found at: {path}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            continue
    
    logger.warning("FFmpeg not found in common locations")
    return False

# Configure FFmpeg path on import
configure_ffmpeg_path()

# Set environment variables globally for pydub
if FFMPEG_PATH and FFMPEG_PATH != "ffmpeg":
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
    # Also set ffprobe path
    ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    if os.path.exists(ffprobe_path):
        os.environ["FFPROBE_BINARY"] = ffprobe_path

def configure_pydub():
    """Configure pydub to use the correct FFmpeg paths"""
    try:
        # Import pydub only when needed
        from pydub import AudioSegment
        
        # Force pydub to reload its configuration
        if FFMPEG_PATH and FFMPEG_PATH != "ffmpeg":
            AudioSegment.converter = FFMPEG_PATH
            logger.info(f"Configured pydub converter: {FFMPEG_PATH}")
            
            # Also set ffprobe path
            ffprobe_path = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
            if os.path.exists(ffprobe_path):
                os.environ["FFPROBE_BINARY"] = ffprobe_path
                logger.info(f"Configured ffprobe path: {ffprobe_path}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to configure pydub: {e}")
        return False

def decode_audio_data(audio_data: Union[bytes, str]) -> bytes:
    """
    Decode audio data that might be base64 encoded
    """
    if isinstance(audio_data, str):
        # Try to decode base64
        try:
            # Remove data URL prefix if present
            if audio_data.startswith('data:audio/'):
                audio_data = audio_data.split(',', 1)[1]
            
            decoded_data = base64.b64decode(audio_data)
            logger.info("Successfully decoded base64 audio data")
            return decoded_data
        except Exception as e:
            logger.warning(f"Failed to decode as base64: {e}")
            # Return as bytes if it's already a string
            return audio_data.encode('utf-8')
    else:
        # Already bytes
        return audio_data

class AudioProcessor:
    """Audio processing utilities that work without FFmpeg dependency"""
    
    @staticmethod
    def create_wav_from_bytes(audio_data: bytes, sample_rate: int = 16000, channels: int = 1) -> str:
        """
        Create a WAV file from raw audio bytes
        """
        try:
            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir)
            wav_path = temp_file.name
            temp_file.close()
            
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Write WAV file
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
            
            logger.info(f"Created WAV file: {wav_path}")
            return wav_path
            
        except Exception as e:
            logger.error(f"Error creating WAV from bytes: {e}")
            raise
    
    @staticmethod
    def convert_audio_format(input_path: str, output_format: str = "wav") -> Optional[str]:
        """
        Convert audio file to WAV format using direct FFmpeg first, then fall back to librosa/pydub
        """
        try:
            # If input is not WAV, try direct FFmpeg conversion first
            if not input_path.lower().endswith('.wav'):
                logger.info(f"Trying direct FFmpeg conversion for: {input_path}")
                ffmpeg_output = convert_with_ffmpeg_direct(input_path, output_format)
                if ffmpeg_output and os.path.exists(ffmpeg_output) and os.path.getsize(ffmpeg_output) > 0:
                    logger.info(f"Direct FFmpeg conversion succeeded: {ffmpeg_output}")
                    return ffmpeg_output
                else:
                    logger.warning("Direct FFmpeg conversion failed or produced empty file")

            # Try librosa (no FFmpeg dependency)
            try:
                import librosa
                import soundfile as sf
                
                logger.info(f"Converting audio with librosa: {input_path}")
                audio_array, sample_rate = librosa.load(input_path, sr=16000)
                
                # Create output path
                output_path = str(pathlib.Path(input_path).parent / f"converted_{pathlib.Path(input_path).stem}.{output_format}")
                
                # Save as WAV
                sf.write(output_path, audio_array, sample_rate)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Audio converted successfully with librosa: {output_path}")
                    return output_path
                else:
                    logger.warning("Librosa conversion failed - file not created or empty")
                    
            except Exception as e:
                logger.warning(f"Librosa conversion failed: {e}")
            
            # Try pydub as fallback (requires FFmpeg but handles more formats)
            try:
                # Configure pydub before importing
                if not configure_pydub():
                    logger.warning("Failed to configure pydub, skipping pydub conversion")
                    return None
                
                from pydub import AudioSegment
                
                logger.info(f"Converting audio with pydub: {input_path}")
                
                audio_segment = AudioSegment.from_file(input_path)
                
                output_path = str(pathlib.Path(input_path).parent / f"converted_pydub_{pathlib.Path(input_path).stem}.{output_format}")
                audio_segment.export(output_path, format=output_format)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Audio converted successfully with pydub: {output_path}")
                    return output_path
                else:
                    logger.warning("Pydub conversion failed - file not created or empty")
                    
            except Exception as e:
                logger.warning(f"Pydub conversion failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """
        Validate that an audio file exists and has content
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"File is empty: {file_path}")
                return False
            
            logger.info(f"Audio file validated: {file_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            return False
    
    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """
        Get basic information about an audio file
        """
        try:
            info = {
                "path": file_path,
                "exists": os.path.exists(file_path),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "format": pathlib.Path(file_path).suffix.lower()
            }
            
            # Try to get more detailed info with wave module
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    info.update({
                        "channels": wav_file.getnchannels(),
                        "sample_width": wav_file.getsampwidth(),
                        "frame_rate": wav_file.getframerate(),
                        "frames": wav_file.getnframes(),
                        "duration": wav_file.getnframes() / wav_file.getframerate()
                    })
            except Exception:
                # Not a WAV file or can't read with wave module
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def cleanup_temp_files(*file_paths: str):
        """
        Clean up temporary files
        """
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error removing temp file {file_path}: {e}")

def detect_audio_format(audio_data: bytes) -> str:
    """
    Detect audio format from file header
    """
    if len(audio_data) < 12:
        return "unknown"
    
    # Check for WAV format
    if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        return "wav"
    
    # Check for MP3 format
    if audio_data[:3] == b'ID3' or (audio_data[:2] == b'\xff\xfb' or audio_data[:2] == b'\xff\xf3'):
        return "mp3"
    
    # Check for M4A/AAC format
    if audio_data[:4] == b'ftyp':
        return "m4a"
    
    # Check for OGG format
    if audio_data[:4] == b'OggS':
        return "ogg"
    
    # Check for FLAC format
    if audio_data[:4] == b'fLaC':
        return "flac"
    
    return "unknown"

def process_audio_for_whisper(audio_data: Union[bytes, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Process audio data for Whisper transcription
    Returns: (primary_path, converted_path)
    """
    temp_path = None
    converted_path = None
    
    try:
        # Decode audio data if needed
        decoded_audio = decode_audio_data(audio_data)
        
        # Detect audio format
        audio_format = detect_audio_format(decoded_audio)
        logger.info(f"Detected audio format: {audio_format}")
        
        # Create temporary file with appropriate extension
        temp_dir = tempfile.gettempdir()
        if audio_format != "unknown":
            suffix = f".{audio_format}"
        else:
            suffix = ".audio"
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)
        temp_path = temp_file.name
        temp_file.write(decoded_audio)
        temp_file.close()
        
        # Ensure proper path format for Windows
        temp_path = str(pathlib.Path(temp_path).resolve())
        
        # Validate the file
        if not AudioProcessor.validate_audio_file(temp_path):
            raise ValueError("Invalid audio file created")
        
        logger.info(f"Created temporary audio file: {temp_path}")
        
        # If it's already WAV, return as is
        if audio_format == "wav":
            logger.info("Audio data is already WAV format")
            return temp_path, None
        
        # Try to convert to WAV
        logger.info(f"Converting {audio_format} to WAV format...")
        converted_path = AudioProcessor.convert_audio_format(temp_path, "wav")
        
        if converted_path and AudioProcessor.validate_audio_file(converted_path):
            logger.info(f"Successfully converted audio to WAV: {converted_path}")
            return temp_path, converted_path
        else:
            logger.warning("Audio conversion failed, will try with original file")
            return temp_path, None
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        # Cleanup on error
        cleanup_paths = [path for path in [temp_path, converted_path] if path is not None]
        if cleanup_paths:
            AudioProcessor.cleanup_temp_files(*cleanup_paths)
        raise 

def convert_with_ffmpeg_direct(input_path: str, output_format: str = "wav") -> Optional[str]:
    """
    Convert audio file directly using FFmpeg command line
    """
    try:
        if not FFMPEG_PATH:
            logger.warning("FFmpeg not available for direct conversion")
            return None
        
        # Create output path
        output_path = str(pathlib.Path(input_path).parent / f"converted_ffmpeg_{pathlib.Path(input_path).stem}.{output_format}")
        
        # Build FFmpeg command
        cmd = [
            FFMPEG_PATH,
            '-i', input_path,
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',          # 16kHz sample rate
            '-ac', '1',              # Mono
            '-y',                    # Overwrite output file
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"FFmpeg conversion successful: {output_path}")
                return output_path
            else:
                logger.warning("FFmpeg conversion failed - output file not created or empty")
        else:
            logger.warning(f"FFmpeg conversion failed: {result.stderr}")
        
        return None
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return None
    except Exception as e:
        logger.error(f"FFmpeg conversion error: {e}")
        return None 