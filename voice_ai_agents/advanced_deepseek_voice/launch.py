#!/usr/bin/env python3
"""
Advanced DeepSeek Voice Agent Launcher
Quick start script for the enhanced voice agent
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'openai', 'speech_recognition', 
        'sounddevice', 'soundfile', 'numpy', 
        'librosa', 'noisereduce', 'plotly'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def main():
    """Launch the advanced voice agent"""
    print("🎙️ Advanced DeepSeek Voice Agent Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "advanced_voice_agent.py"
    
    if not app_file.exists():
        print("❌ Error: advanced_voice_agent.py not found!")
        sys.exit(1)
    
    print("🚀 Starting Advanced DeepSeek Voice Agent...")
    print("🌟 Enhanced features: Emotion detection, noise reduction, multi-language")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--server.maxUploadSize", "200",
            "--theme.base", "dark"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Advanced DeepSeek Voice Agent...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

