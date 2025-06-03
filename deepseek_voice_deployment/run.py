#!/usr/bin/env python3
"""
Quick launcher for DeepSeek Voice Agent
Alternative entry point that can be run directly
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "deepseek_voice_app.py"
    
    if not app_file.exists():
        print("❌ Error: deepseek_voice_app.py not found!")
        sys.exit(1)
    
    print("🚀 Starting DeepSeek Voice Agent...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping DeepSeek Voice Agent...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

