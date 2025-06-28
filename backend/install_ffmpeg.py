#!/usr/bin/env python3
"""
FFmpeg installation helper for Windows
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path

def check_ffmpeg():
    """Check if FFmpeg is already installed"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ FFmpeg is already installed!")
            print(f"Version: {result.stdout.split('ffmpeg version')[1].split()[0]}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("✗ FFmpeg is not installed or not in PATH")
    return False

def install_with_chocolatey():
    """Install FFmpeg using Chocolatey"""
    print("Attempting to install FFmpeg with Chocolatey...")
    
    try:
        # Check if Chocolatey is installed
        result = subprocess.run(['choco', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Chocolatey not found. Installing Chocolatey first...")
            
            # Install Chocolatey
            install_script = """
            Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
                f.write(install_script)
                script_path = f.name
            
            try:
                subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path], 
                             check=True)
                print("Chocolatey installed successfully")
            finally:
                os.unlink(script_path)
        
        # Install FFmpeg
        print("Installing FFmpeg with Chocolatey...")
        subprocess.run(['choco', 'install', 'ffmpeg', '-y'], check=True)
        
        # Verify installation
        if check_ffmpeg():
            print("✓ FFmpeg installed successfully with Chocolatey!")
            return True
        else:
            print("✗ FFmpeg installation failed with Chocolatey")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Chocolatey installation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during Chocolatey installation: {e}")
        return False

def install_with_scoop():
    """Install FFmpeg using Scoop"""
    print("Attempting to install FFmpeg with Scoop...")
    
    try:
        # Check if Scoop is installed
        result = subprocess.run(['scoop', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Scoop not found. Installing Scoop first...")
            
            # Install Scoop
            install_script = """
            Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
            irm get.scoop.sh | iex
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False) as f:
                f.write(install_script)
                script_path = f.name
            
            try:
                subprocess.run(['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path], 
                             check=True)
                print("Scoop installed successfully")
            finally:
                os.unlink(script_path)
        
        # Install FFmpeg
        print("Installing FFmpeg with Scoop...")
        subprocess.run(['scoop', 'install', 'ffmpeg'], check=True)
        
        # Verify installation
        if check_ffmpeg():
            print("✓ FFmpeg installed successfully with Scoop!")
            return True
        else:
            print("✗ FFmpeg installation failed with Scoop")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Scoop installation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error during Scoop installation: {e}")
        return False

def install_manual():
    """Manual FFmpeg installation"""
    print("Manual FFmpeg installation...")
    print("\nPlease follow these steps:")
    print("1. Go to https://ffmpeg.org/download.html")
    print("2. Click on 'Windows builds'")
    print("3. Download the latest 'essentials' build")
    print("4. Extract the zip file")
    print("5. Copy the extracted folder to C:\\ffmpeg")
    print("6. Add C:\\ffmpeg\\bin to your system PATH")
    print("7. Restart your terminal")
    print("\nAfter installation, run this script again to verify.")
    
    input("\nPress Enter when you have completed the installation...")
    
    if check_ffmpeg():
        print("✓ FFmpeg installed successfully!")
        return True
    else:
        print("✗ FFmpeg not found. Please check your installation.")
        return False

def main():
    """Main installation function"""
    print("FFmpeg Installation Helper for Windows")
    print("=" * 40)
    
    # Check if already installed
    if check_ffmpeg():
        return
    
    print("\nFFmpeg is required for audio processing in the Audio RAG system.")
    print("Choose an installation method:")
    print("1. Chocolatey (recommended)")
    print("2. Scoop")
    print("3. Manual installation")
    print("4. Skip installation")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            if install_with_chocolatey():
                break
            else:
                print("Chocolatey installation failed. Try another method.")
        elif choice == "2":
            if install_with_scoop():
                break
            else:
                print("Scoop installation failed. Try another method.")
        elif choice == "3":
            if install_manual():
                break
        elif choice == "4":
            print("Skipping FFmpeg installation.")
            print("Note: Some audio formats may not work without FFmpeg.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print("\nInstallation complete!")
    print("You can now run the Audio RAG system.")

if __name__ == "__main__":
    main() 