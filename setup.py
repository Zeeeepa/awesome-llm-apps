#!/usr/bin/env python3
"""
Setup script for Voice Computer Assistant

This script helps with the installation and setup of the voice computer assistant.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_python_dependencies():
    """Install Python dependencies from requirements.txt"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def install_system_dependencies():
    """Install system-specific dependencies"""
    system = platform.system()
    print(f"🔧 Installing system dependencies for {system}...")
    
    if system == "Darwin":  # macOS
        print("Installing macOS dependencies...")
        try:
            # Check if Homebrew is installed
            subprocess.check_call(["which", "brew"], stdout=subprocess.DEVNULL)
            print("  ✅ Homebrew found")
            
            # Install portaudio for pyaudio
            subprocess.check_call(["brew", "install", "portaudio"])
            print("  ✅ PortAudio installed")
            
            return True
        except subprocess.CalledProcessError:
            print("  ⚠️ Homebrew not found. Please install Homebrew first:")
            print("    /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
    
    elif system == "Linux":
        print("Installing Linux dependencies...")
        try:
            # Try to detect the package manager
            if subprocess.call(["which", "apt-get"], stdout=subprocess.DEVNULL) == 0:
                # Ubuntu/Debian
                subprocess.check_call([
                    "sudo", "apt-get", "update"
                ])
                subprocess.check_call([
                    "sudo", "apt-get", "install", "-y",
                    "portaudio19-dev", "python3-pyaudio", "tesseract-ocr",
                    "python3-dev", "build-essential"
                ])
                print("  ✅ Ubuntu/Debian dependencies installed")
                return True
            
            elif subprocess.call(["which", "yum"], stdout=subprocess.DEVNULL) == 0:
                # CentOS/RHEL
                subprocess.check_call([
                    "sudo", "yum", "install", "-y",
                    "portaudio-devel", "tesseract", "python3-devel", "gcc"
                ])
                print("  ✅ CentOS/RHEL dependencies installed")
                return True
            
            else:
                print("  ⚠️ Unknown Linux distribution. Please install manually:")
                print("    - portaudio development libraries")
                print("    - tesseract-ocr")
                print("    - python3 development headers")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install Linux dependencies: {e}")
            return False
    
    elif system == "Windows":
        print("Windows setup instructions:")
        print("  1. Install Tesseract OCR from:")
        print("     https://github.com/UB-Mannheim/tesseract/wiki")
        print("  2. Add Tesseract to your PATH environment variable")
        print("  3. Python dependencies should install automatically")
        print("  ✅ Please follow the manual steps above")
        return True
    
    else:
        print(f"  ⚠️ Unsupported system: {system}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        Path.home() / ".voice_assistant",
        Path.home() / ".voice_assistant" / "audio",
        Path.home() / ".voice_assistant" / "screenshots",
        Path.home() / ".voice_assistant" / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def create_env_file():
    """Create a sample .env file"""
    print("📝 Creating sample .env file...")
    
    env_content = """# Voice Computer Assistant Configuration
# Copy this file to .env and fill in your actual values

# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Logging configuration
VOICE_ASSISTANT_LOG_LEVEL=INFO

# Optional: Custom directories
VOICE_ASSISTANT_TEMP_DIR=/tmp/voice_assistant
VOICE_ASSISTANT_AUDIO_DIR=/tmp/voice_assistant/audio
VOICE_ASSISTANT_SCREENSHOTS_DIR=/tmp/voice_assistant/screenshots

# Optional: Safety settings
VOICE_ASSISTANT_SAFETY_MODE=high
"""
    
    env_file = Path(".env.example")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"  ✅ Created: {env_file}")
    print("  📝 Copy .env.example to .env and add your OpenAI API key")
    
    return True

def test_installation():
    """Test if the installation was successful"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import streamlit
        print("  ✅ Streamlit import successful")
        
        import openai
        print("  ✅ OpenAI import successful")
        
        try:
            import cv2
            print("  ✅ OpenCV import successful")
        except ImportError:
            print("  ⚠️ OpenCV import failed (optional)")
        
        try:
            import pyautogui
            print("  ✅ PyAutoGUI import successful")
        except ImportError:
            print("  ⚠️ PyAutoGUI import failed")
        
        try:
            import PIL
            print("  ✅ Pillow import successful")
        except ImportError:
            print("  ⚠️ Pillow import failed")
        
        print("✅ Installation test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🎤💻 Voice Computer Assistant Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing Python dependencies", install_python_dependencies),
        ("Installing system dependencies", install_system_dependencies),
        ("Creating directories", create_directories),
        ("Creating environment file", create_env_file),
        ("Testing installation", test_installation)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"❌ {step_name} failed")
                break
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            results.append((step_name, False))
            break
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Setup Results:")
    print("=" * 50)
    
    for step_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step_name}: {status}")
    
    if all(result[1] for result in results):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run the demo: python demo.py")
        print("4. Start the app: streamlit run voice_computer_assistant.py")
    else:
        print("\n⚠️ Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()

