#!/usr/bin/env python3
"""
Script to add FFmpeg to system PATH permanently on Windows
"""

import os
import subprocess
import winreg
from pathlib import Path

def add_ffmpeg_to_path():
    """Add FFmpeg to system PATH permanently"""
    ffmpeg_path = "C:\\ffmpeg\\bin"
    
    if not os.path.exists(ffmpeg_path):
        print(f"FFmpeg path not found: {ffmpeg_path}")
        return False
    
    try:
        # Open the registry key for system PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment", 
                           0, winreg.KEY_READ | winreg.KEY_WRITE)
        
        # Get current PATH
        current_path, _ = winreg.QueryValueEx(key, "Path")
        
        # Check if FFmpeg is already in PATH
        if ffmpeg_path in current_path:
            print("FFmpeg is already in system PATH")
            return True
        
        # Add FFmpeg to PATH
        new_path = current_path + ";" + ffmpeg_path
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
        
        print(f"Added {ffmpeg_path} to system PATH")
        print("You may need to restart your terminal/IDE for changes to take effect")
        
        return True
        
    except Exception as e:
        print(f"Error adding FFmpeg to PATH: {e}")
        print("You may need to run this script as administrator")
        return False
    finally:
        try:
            winreg.CloseKey(key)
        except:
            pass

def test_ffmpeg():
    """Test if FFmpeg is accessible"""
    try:
        result = subprocess.run(['C:\\ffmpeg\\bin\\ffmpeg.exe', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ FFmpeg is working correctly")
            return True
        else:
            print("✗ FFmpeg test failed")
            return False
    except Exception as e:
        print(f"✗ FFmpeg test failed: {e}")
        return False

if __name__ == "__main__":
    print("FFmpeg PATH Setup")
    print("=" * 20)
    
    if test_ffmpeg():
        print("\nAdding FFmpeg to system PATH...")
        if add_ffmpeg_to_path():
            print("✓ Setup completed successfully!")
        else:
            print("✗ Setup failed. Try running as administrator.")
    else:
        print("✗ FFmpeg not found at C:\\ffmpeg\\bin\\ffmpeg.exe")
        print("Please install FFmpeg first.") 