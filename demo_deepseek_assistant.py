#!/usr/bin/env python3
"""
Demo script for DeepSeek Voice Assistant

This script demonstrates the key capabilities of the DeepSeek voice assistant
without requiring the full Streamlit interface.
"""

import os
import sys
import asyncio
import requests
import json
from pathlib import Path

# Check if required libraries are available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# DeepSeek API configuration
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]

class DeepSeekDemo:
    """Demo class for DeepSeek Voice Assistant"""
    
    def __init__(self):
        self.conversation_history = []
    
    def test_deepseek_api(self, api_key: str, model: str = "deepseek-chat") -> bool:
        """Test DeepSeek API connectivity"""
        print(f"🧪 Testing DeepSeek API with model: {model}")
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello! Please respond with 'API test successful' if you can hear me."}
                ],
                "temperature": 0.1,
                "max_tokens": 50
            }
            
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                print(f"  ✅ API Response: {response_text}")
                return True
            else:
                print(f"  ❌ API Error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"  ❌ API Test Failed: {str(e)}")
            return False
    
    def test_openai_tts(self, api_key: str, voice: str = "nova") -> bool:
        """Test OpenAI TTS functionality"""
        print(f"🔊 Testing OpenAI TTS with voice: {voice}")
        
        if not OPENAI_AVAILABLE:
            print("  ❌ OpenAI library not available")
            return False
        
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input="This is a test of the text-to-speech functionality.",
                speed=1.0
            )
            
            # Save to temporary file
            temp_file = Path("test_tts_output.mp3")
            response.stream_to_file(str(temp_file))
            
            print(f"  ✅ TTS test successful! Audio saved to: {temp_file}")
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"  ❌ TTS Test Failed: {str(e)}")
            return False
    
    def test_audio_recording(self) -> bool:
        """Test audio recording capabilities"""
        print("🎤 Testing audio recording capabilities")
        
        if not AUDIO_AVAILABLE:
            print("  ❌ Audio libraries not available")
            print("  📝 Install with: pip install sounddevice soundfile numpy")
            return False
        
        try:
            # Test basic audio device access
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                print("  ❌ No audio input devices found")
                return False
            
            print(f"  ✅ Found {len(input_devices)} audio input device(s)")
            print(f"  📱 Default input device: {sd.query_devices(kind='input')['name']}")
            
            # Test short recording (without actually recording)
            sample_rate = 16000
            duration = 0.1  # Very short test
            
            try:
                # Test if we can create a recording stream
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32):
                    print("  ✅ Audio recording stream test successful")
                return True
                
            except Exception as stream_error:
                print(f"  ❌ Audio stream test failed: {stream_error}")
                return False
                
        except Exception as e:
            print(f"  ❌ Audio test failed: {str(e)}")
            return False
    
    def demo_conversation(self, deepseek_api_key: str, model: str = "deepseek-chat") -> bool:
        """Demo a conversation with DeepSeek"""
        print(f"💬 Demo conversation with {model}")
        
        test_messages = [
            "Hello! Can you introduce yourself?",
            "What are your main capabilities?",
            "Can you help me write a simple Python function to calculate fibonacci numbers?"
        ]
        
        try:
            for i, message in enumerate(test_messages, 1):
                print(f"\n  📤 Message {i}: {message}")
                
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": message})
                
                # Prepare API request
                headers = {
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model,
                    "messages": self.conversation_history.copy(),
                    "temperature": 0.7,
                    "max_tokens": 500
                }
                
                response = requests.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    
                    # Add to conversation history
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    
                    print(f"  📥 Response {i}: {response_text[:200]}...")
                    
                else:
                    print(f"  ❌ API Error: {response.status_code}")
                    return False
            
            print("  ✅ Conversation demo completed successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ Conversation demo failed: {str(e)}")
            return False
    
    def demo_model_comparison(self, deepseek_api_key: str) -> bool:
        """Demo different DeepSeek models"""
        print("🧠 Testing different DeepSeek models")
        
        test_prompt = "Write a simple 'Hello World' program in Python and explain what it does."
        
        for model in DEEPSEEK_MODELS:
            print(f"\n  🔄 Testing {model}...")
            
            try:
                headers = {
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300
                }
                
                response = requests.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result["choices"][0]["message"]["content"]
                    print(f"    ✅ {model}: {response_text[:100]}...")
                else:
                    print(f"    ❌ {model}: API Error {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ {model}: {str(e)}")
        
        return True

def main():
    """Main demo function"""
    print("🤖🎤 DeepSeek Voice Assistant Demo")
    print("=" * 50)
    
    # Check for API keys
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY environment variable not found")
        print("   Please set your DeepSeek API key:")
        print("   export DEEPSEEK_API_KEY='your_api_key_here'")
        print("   Or create a .env file with DEEPSEEK_API_KEY=your_api_key_here")
        return
    
    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY environment variable not found")
        print("   TTS functionality will be skipped")
    
    # Initialize demo
    demo = DeepSeekDemo()
    
    # Run tests
    tests = [
        ("DeepSeek API Connectivity", lambda: demo.test_deepseek_api(deepseek_api_key)),
        ("Audio Recording Capabilities", demo.test_audio_recording),
        ("Conversation Demo", lambda: demo.demo_conversation(deepseek_api_key)),
        ("Model Comparison", lambda: demo.demo_model_comparison(deepseek_api_key))
    ]
    
    # Add TTS test if OpenAI key is available
    if openai_api_key:
        tests.insert(1, ("OpenAI TTS", lambda: demo.test_openai_tts(openai_api_key)))
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Demo Results Summary:")
    print("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The DeepSeek voice assistant is ready to use.")
        print("\nTo start the full application, run:")
        print("  streamlit run deepseek_voice_assistant.py")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("Make sure all dependencies are installed and API keys are configured.")
    
    print("\n📚 Next Steps:")
    print("1. Set up your API keys in a .env file")
    print("2. Install any missing dependencies")
    print("3. Run the setup script: python setup_deepseek_assistant.py")
    print("4. Launch the assistant: streamlit run deepseek_voice_assistant.py")

if __name__ == "__main__":
    main()

