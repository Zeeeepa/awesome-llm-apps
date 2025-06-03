#!/usr/bin/env python3
"""
Setup script for DeepSeek Voice Assistant

This script helps with the installation and configuration of the DeepSeek voice assistant.
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

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "deepseek_requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create a sample .env file for API keys"""
    print("📝 Creating sample .env file...")
    
    env_content = """# DeepSeek Voice Assistant Configuration
# Copy this file to .env and fill in your actual API keys

# DeepSeek API Key (required)
# Get your API key from: https://platform.deepseek.com/
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI API Key (required for text-to-speech)
# Get your API key from: https://platform.openai.com/
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
"""
    
    env_file = Path(".env.deepseek.example")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"✅ Created: {env_file}")
    print("📝 Copy .env.deepseek.example to .env and add your API keys")
    
    return True

def install_audio_dependencies():
    """Install audio dependencies based on the operating system"""
    system = platform.system()
    print(f"🔧 Installing audio dependencies for {system}...")
    
    if system == "Darwin":  # macOS
        print("Installing macOS audio dependencies...")
        try:
            # Check if Homebrew is installed
            subprocess.check_call(["which", "brew"], stdout=subprocess.DEVNULL)
            print("  ✅ Homebrew found")
            
            # Install portaudio for sounddevice
            subprocess.check_call(["brew", "install", "portaudio"])
            print("  ✅ PortAudio installed")
            
            return True
        except subprocess.CalledProcessError:
            print("  ⚠️ Homebrew not found. Please install Homebrew first:")
            print("    /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
    
    elif system == "Linux":
        print("Installing Linux audio dependencies...")
        try:
            if subprocess.call(["which", "apt-get"], stdout=subprocess.DEVNULL) == 0:
                # Ubuntu/Debian
                subprocess.check_call([
                    "sudo", "apt-get", "update"
                ])
                subprocess.check_call([
                    "sudo", "apt-get", "install", "-y",
                    "portaudio19-dev", "python3-dev", "build-essential"
                ])
                print("  ✅ Ubuntu/Debian audio dependencies installed")
                return True
            else:
                print("  ⚠️ Please install portaudio development libraries manually")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install Linux dependencies: {e}")
            return False
    
    elif system == "Windows":
        print("Windows audio setup:")
        print("  ✅ Audio dependencies should install automatically with pip")
        print("  📝 If you encounter issues, try installing Microsoft Visual C++ Build Tools")
        return True
    
    else:
        print(f"  ⚠️ Unsupported system: {system}")
        return False

def test_installation():
    """Test if the installation was successful"""
    print("🧪 Testing installation...")
    
    try:
        # Test core imports
        import streamlit
        print("  ✅ Streamlit import successful")
        
        import requests
        print("  ✅ Requests import successful")
        
        import openai
        print("  ✅ OpenAI import successful")
        
        # Test audio imports (optional)
        try:
            import sounddevice
            import soundfile
            import numpy
            print("  ✅ Audio libraries import successful")
        except ImportError:
            print("  ⚠️ Audio libraries import failed (optional)")
        
        print("✅ Installation test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Installation test failed: {e}")
        return False

def create_launch_script():
    """Create a convenient launch script"""
    print("📝 Creating launch script...")
    
    # Create launch script for different platforms
    if platform.system() == "Windows":
        script_content = """@echo off
echo Starting DeepSeek Voice Assistant...
streamlit run deepseek_voice_assistant.py
pause
"""
        script_file = Path("launch_deepseek_assistant.bat")
    else:
        script_content = """#!/bin/bash
echo "Starting DeepSeek Voice Assistant..."
streamlit run deepseek_voice_assistant.py
"""
        script_file = Path("launch_deepseek_assistant.sh")
    
    with open(script_file, "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod(script_file, 0o755)
    
    print(f"✅ Created launch script: {script_file}")
    return True

def main():
    """Main setup function"""
    print("🤖🎤 DeepSeek Voice Assistant Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing Python dependencies", install_dependencies),
        ("Installing audio dependencies", install_audio_dependencies),
        ("Creating environment file", create_env_file),
        ("Creating launch script", create_launch_script),
        ("Testing installation", test_installation)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success and step_name in ["Checking Python version", "Installing Python dependencies"]:
                print(f"❌ {step_name} failed - stopping setup")
                break
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            results.append((step_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Setup Results:")
    print("=" * 50)
    
    for step_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step_name}: {status}")
    
    # Check if core setup succeeded
    core_steps = ["Checking Python version", "Installing Python dependencies"]
    core_success = all(
        result[1] for result in results 
        if result[0] in core_steps
    )
    
    if core_success:
        print("\n🎉 Core setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.deepseek.example to .env")
        print("2. Add your DeepSeek API key to .env")
        print("3. Add your OpenAI API key to .env (for TTS)")
        print("4. Run the assistant:")
        
        if platform.system() == "Windows":
            print("   - Double-click launch_deepseek_assistant.bat")
        else:
            print("   - Run: ./launch_deepseek_assistant.sh")
        
        print("   - Or manually: streamlit run deepseek_voice_assistant.py")
        
        print("\n📚 API Key Resources:")
        print("   - DeepSeek: https://platform.deepseek.com/")
        print("   - OpenAI: https://platform.openai.com/")
        
    else:
        print("\n⚠️ Core setup failed. Please resolve the issues above.")
        print("You may need to:")
        print("- Update Python to version 3.8 or higher")
        print("- Check your internet connection")
        print("- Install pip if not available")

if __name__ == "__main__":
    main()

