#!/usr/bin/env python3
"""
DeepSeek Voice Assistant with Enhanced UI

A voice-enabled assistant that uses DeepSeek API for AI responses with toggleable
voice input/output controls and model selection.
"""

import streamlit as st
import asyncio
import os
import tempfile
import uuid
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import requests
import base64

# Audio processing imports
try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.warning("Audio libraries not available. Install with: pip install sounddevice soundfile numpy")

# OpenAI for TTS (keeping this for voice output)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="DeepSeek Voice Assistant",
    page_icon="🤖🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DeepSeek API configuration
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner"
]

# Voice options for TTS
VOICE_OPTIONS = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]

class AudioRecorder:
    """Simple audio recorder for voice input"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.channels = 1
        
    def start_recording(self):
        """Start recording audio"""
        if not AUDIO_AVAILABLE:
            return False
            
        self.is_recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            dtype=np.float32
        )
        self.stream.start()
        return True
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not AUDIO_AVAILABLE or not self.is_recording:
            return None
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        if self.audio_data:
            # Combine all audio chunks
            audio_array = np.concatenate(self.audio_data, axis=0)
            return audio_array
        return None

class DeepSeekVoiceAssistant:
    """Main voice assistant class"""
    
    def __init__(self):
        self.conversation_history = []
        self.audio_recorder = AudioRecorder()
        self.temp_dir = Path(tempfile.gettempdir()) / "deepseek_voice_assistant"
        self.temp_dir.mkdir(exist_ok=True)
    
    def call_deepseek_api(self, messages: List[Dict], model: str, api_key: str) -> Optional[str]:
        """Call DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
                "stream": False
            }
            
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                st.error(f"DeepSeek API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error calling DeepSeek API: {str(e)}")
            return None
    
    def text_to_speech(self, text: str, voice: str, openai_api_key: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS"""
        if not OPENAI_AVAILABLE:
            return None
            
        try:
            client = OpenAI(api_key=openai_api_key)
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=1.0
            )
            
            # Save to temporary file
            audio_file_path = self.temp_dir / f"tts_{uuid.uuid4()}.mp3"
            response.stream_to_file(str(audio_file_path))
            
            return str(audio_file_path)
            
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
            return None
    
    def speech_to_text_simple(self, audio_data: np.ndarray) -> str:
        """Simple speech to text (placeholder - would need actual STT service)"""
        # This is a placeholder - in a real implementation you'd use:
        # - OpenAI Whisper API
        # - Google Speech-to-Text
        # - Azure Speech Services
        # - Or local Whisper model
        
        # For demo purposes, return a placeholder
        return "[Speech recognized - integrate with STT service]"

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "deepseek_api_key": "",
        "openai_api_key": "",
        "selected_model": "deepseek-chat",
        "selected_voice": "nova",
        "microphone_listen": False,
        "read_response": False,
        "conversation_history": [],
        "current_response": "",
        "recording": False,
        "assistant": None,
        "last_audio_path": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_config():
    """Configure sidebar with API keys and settings"""
    with st.sidebar:
        st.title("🤖 DeepSeek Voice Assistant")
        st.markdown("---")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        
        st.session_state.deepseek_api_key = st.text_input(
            "DeepSeek API Key",
            value=st.session_state.deepseek_api_key,
            type="password",
            help="Get your API key from https://platform.deepseek.com/"
        )
        
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key (for TTS)",
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for text-to-speech functionality"
        )
        
        # Model Selection
        st.markdown("### 🧠 Model Selection")
        st.session_state.selected_model = st.selectbox(
            "DeepSeek Model",
            options=DEEPSEEK_MODELS,
            index=DEEPSEEK_MODELS.index(st.session_state.selected_model),
            help="Choose the DeepSeek model for responses"
        )
        
        # Voice Settings
        st.markdown("### 🎙️ Voice Settings")
        st.session_state.selected_voice = st.selectbox(
            "TTS Voice",
            options=VOICE_OPTIONS,
            index=VOICE_OPTIONS.index(st.session_state.selected_voice),
            help="Choose voice for text-to-speech"
        )
        
        # Toggle Controls
        st.markdown("### 🎛️ Controls")
        
        # Microphone Listen Toggle
        st.session_state.microphone_listen = st.toggle(
            "🎤 Microphone Listen",
            value=st.session_state.microphone_listen,
            help="Enable voice input"
        )
        
        # Read Response Toggle
        st.session_state.read_response = st.toggle(
            "🔊 Read Response",
            value=st.session_state.read_response,
            help="Enable text-to-speech for responses"
        )
        
        # Status indicators
        st.markdown("### 📊 Status")
        
        # API Key status
        deepseek_status = "✅" if st.session_state.deepseek_api_key else "❌"
        openai_status = "✅" if st.session_state.openai_api_key else "❌"
        
        st.markdown(f"DeepSeek API: {deepseek_status}")
        st.markdown(f"OpenAI API: {openai_status}")
        st.markdown(f"Audio Available: {'✅' if AUDIO_AVAILABLE else '❌'}")
        
        # Recording status
        if st.session_state.recording:
            st.markdown("🔴 **Recording...**")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.current_response = ""
            st.rerun()

def main_interface():
    """Main application interface"""
    st.title("🤖🎤 DeepSeek Voice Assistant")
    st.markdown("Interact with DeepSeek AI using voice commands and get spoken responses!")
    
    # Check if required APIs are configured
    if not st.session_state.deepseek_api_key:
        st.warning("⚠️ Please configure your DeepSeek API key in the sidebar to get started.")
        return
    
    # Initialize assistant if not already done
    if st.session_state.assistant is None:
        st.session_state.assistant = DeepSeekVoiceAssistant()
    
    # Main interface layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Voice Input Section
        st.markdown("### 🎤 Voice Input")
        
        if st.session_state.microphone_listen and AUDIO_AVAILABLE:
            # Voice input controls
            voice_col1, voice_col2 = st.columns(2)
            
            with voice_col1:
                if st.button("🎤 Start Recording", disabled=st.session_state.recording):
                    st.session_state.recording = True
                    st.session_state.assistant.audio_recorder.start_recording()
                    st.rerun()
            
            with voice_col2:
                if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
                    audio_data = st.session_state.assistant.audio_recorder.stop_recording()
                    st.session_state.recording = False
                    
                    if audio_data is not None:
                        # Convert audio to text (placeholder)
                        text_input = st.session_state.assistant.speech_to_text_simple(audio_data)
                        st.success(f"Recognized: {text_input}")
                        
                        # Process the voice input
                        process_user_input(text_input)
                    
                    st.rerun()
        
        elif st.session_state.microphone_listen and not AUDIO_AVAILABLE:
            st.error("Audio libraries not available. Please install: pip install sounddevice soundfile numpy")
        
        # Text Input Section
        st.markdown("### ⌨️ Text Input")
        
        # Text input area
        user_input = st.text_area(
            "Type your message:",
            placeholder="Ask DeepSeek anything...",
            height=100,
            key="text_input"
        )
        
        # Send button
        if st.button("📤 Send Message", type="primary"):
            if user_input.strip():
                process_user_input(user_input.strip())
                # Clear the text input
                st.session_state.text_input = ""
                st.rerun()
        
        # Current Response Display
        if st.session_state.current_response:
            st.markdown("### 🤖 DeepSeek Response")
            
            # Display response in a nice container
            with st.container():
                st.markdown(st.session_state.current_response)
            
            # Audio playback if available
            if st.session_state.read_response and st.session_state.last_audio_path:
                st.markdown("### 🔊 Audio Response")
                
                try:
                    with open(st.session_state.last_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name=f"deepseek_response_{int(time.time())}.mp3",
                            mime="audio/mp3"
                        )
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
    
    with col2:
        # Conversation History
        st.markdown("### 💬 Conversation History")
        
        if st.session_state.conversation_history:
            # Display conversation in reverse order (newest first)
            for i, entry in enumerate(reversed(st.session_state.conversation_history[-10:])):
                with st.expander(f"{entry['role'].title()} - {entry['timestamp']}", expanded=(i == 0)):
                    st.markdown(entry['content'])
        else:
            st.info("No conversation history yet. Start by sending a message!")
        
        # Settings Summary
        st.markdown("### ⚙️ Current Settings")
        st.markdown(f"**Model:** {st.session_state.selected_model}")
        st.markdown(f"**Voice:** {st.session_state.selected_voice}")
        st.markdown(f"**Microphone:** {'🟢 ON' if st.session_state.microphone_listen else '🔴 OFF'}")
        st.markdown(f"**Read Response:** {'🟢 ON' if st.session_state.read_response else '🔴 OFF'}")

def process_user_input(user_input: str):
    """Process user input and get response from DeepSeek"""
    if not user_input.strip():
        return
    
    # Add user message to conversation history
    user_entry = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.conversation_history.append(user_entry)
    
    # Prepare messages for DeepSeek API
    messages = []
    
    # Add conversation history (last 10 messages to stay within limits)
    for entry in st.session_state.conversation_history[-10:]:
        messages.append({
            "role": entry["role"],
            "content": entry["content"]
        })
    
    # Show processing indicator
    with st.spinner("🤖 DeepSeek is thinking..."):
        # Call DeepSeek API
        response = st.session_state.assistant.call_deepseek_api(
            messages=messages,
            model=st.session_state.selected_model,
            api_key=st.session_state.deepseek_api_key
        )
    
    if response:
        # Add assistant response to conversation history
        assistant_entry = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.conversation_history.append(assistant_entry)
        
        # Set current response
        st.session_state.current_response = response
        
        # Generate speech if read response is enabled
        if st.session_state.read_response and st.session_state.openai_api_key and OPENAI_AVAILABLE:
            with st.spinner("🔊 Generating speech..."):
                audio_path = st.session_state.assistant.text_to_speech(
                    text=response,
                    voice=st.session_state.selected_voice,
                    openai_api_key=st.session_state.openai_api_key
                )
                
                if audio_path:
                    st.session_state.last_audio_path = audio_path
                    st.success("🔊 Audio response generated!")
        
        st.success("✅ Response received!")
    else:
        st.error("❌ Failed to get response from DeepSeek API")

def main():
    """Main application entry point"""
    init_session_state()
    sidebar_config()
    main_interface()

if __name__ == "__main__":
    main()

