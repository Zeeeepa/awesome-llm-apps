#!/usr/bin/env python3
"""
DeepSeek Voice Agent - Single Deployment Package
A consolidated voice-to-voice AI assistant using DeepSeek models

This is a complete deployment package that combines:
- Voice input (speech-to-text)
- DeepSeek AI processing
- Voice output (text-to-speech)
- All features in a single file for easy deployment
"""

import os
import sys
import json
import tempfile
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-coder", 
    "deepseek-reasoner",
    "deepseek-r1",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-qwen-14b",
    "deepseek-r1-distill-qwen-7b",
    "deepseek-r1-distill-qwen-1.5b"
]

TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

class DeepSeekVoiceAgent:
    """Complete voice-to-voice AI agent with DeepSeek integration"""
    
    def __init__(self):
        self.deepseek_client = None
        self.openai_client = None
        self.recognizer = None
        self.microphone = None
        self.conversation_history = []
        self.setup_complete = False
        
    def initialize_clients(self, deepseek_key: str, openai_key: str) -> bool:
        """Initialize API clients"""
        try:
            if deepseek_key:
                self.deepseek_client = OpenAI(
                    api_key=deepseek_key,
                    base_url="https://api.deepseek.com"
                )
            
            if openai_key:
                self.openai_client = OpenAI(api_key=openai_key)
            
            if self.deepseek_client and self.openai_client:
                self.setup_complete = True
                return True
        except Exception as e:
            st.error(f"Error initializing clients: {str(e)}")
        return False
    
    def setup_speech_recognition(self) -> bool:
        """Initialize speech recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            return True
        except Exception as e:
            st.error(f"Error setting up speech recognition: {str(e)}")
            return False
    
    def listen_for_speech(self, timeout: int = 5) -> Optional[str]:
        """Capture speech input"""
        if not self.recognizer or not self.microphone:
            return None
        
        try:
            with self.microphone as source:
                st.info("🎤 Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            st.info("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.warning("🤷 Could not understand the speech. Please try again.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
        
        return None
    
    async def get_deepseek_response(self, message: str, model: str) -> str:
        """Get response from DeepSeek"""
        if not self.deepseek_client:
            return "DeepSeek client not initialized"
        
        try:
            # Prepare messages with conversation history
            messages = []
            for entry in self.conversation_history[-10:]:  # Last 10 messages
                messages.append({"role": entry["role"], "content": entry["content"]})
            messages.append({"role": "user", "content": message})
            
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error getting DeepSeek response: {str(e)}"
    
    async def text_to_speech(self, text: str, voice: str) -> Optional[str]:
        """Convert text to speech"""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=1.0
            )
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")
            
            with open(audio_path, "wb") as f:
                f.write(response.content)
            
            return audio_path
            
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })

def init_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        "agent": DeepSeekVoiceAgent(),
        "deepseek_api_key": "",
        "openai_api_key": "",
        "selected_model": "deepseek-chat",
        "selected_voice": "nova",
        "microphone_enabled": True,
        "read_response_enabled": True,
        "setup_complete": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="DeepSeek Voice Agent",
        page_icon="🎙️",
        layout="wide"
    )
    
    init_session_state()
    agent = st.session_state.agent
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("---")
        
        # API Keys
        st.markdown("### 🔑 API Settings")
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.deepseek_api_key,
            help="Get your key from https://platform.deepseek.com/"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.openai_api_key,
            help="Get your key from https://platform.openai.com/"
        )
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "DeepSeek Model",
            options=DEEPSEEK_MODELS,
            index=DEEPSEEK_MODELS.index(st.session_state.selected_model)
        )
        
        # Voice Settings
        st.markdown("### 🎤 Voice Settings")
        selected_voice = st.selectbox(
            "TTS Voice",
            options=TTS_VOICES,
            index=TTS_VOICES.index(st.session_state.selected_voice)
        )
        
        # Feature Toggles
        st.markdown("### ⚙️ Controls")
        microphone_enabled = st.toggle(
            "🎙️ Microphone Listen",
            value=st.session_state.microphone_enabled
        )
        
        read_response_enabled = st.toggle(
            "🔊 Read Response",
            value=st.session_state.read_response_enabled
        )
        
        st.markdown("---")
        
        # Initialize System
        if st.button("🚀 Initialize System", type="primary"):
            if deepseek_key and openai_key:
                with st.spinner("Setting up system..."):
                    # Initialize clients
                    clients_ok = agent.initialize_clients(deepseek_key, openai_key)
                    speech_ok = agent.setup_speech_recognition()
                    
                    if clients_ok and speech_ok:
                        st.session_state.setup_complete = True
                        st.session_state.deepseek_api_key = deepseek_key
                        st.session_state.openai_api_key = openai_key
                        st.session_state.selected_model = selected_model
                        st.session_state.selected_voice = selected_voice
                        st.session_state.microphone_enabled = microphone_enabled
                        st.session_state.read_response_enabled = read_response_enabled
                        st.success("✅ System initialized!")
                        st.rerun()
                    else:
                        st.error("❌ Initialization failed!")
            else:
                st.error("Please enter both API keys!")
        
        if st.button("🗑️ Clear History"):
            agent.conversation_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Main Interface
    st.title("🎙️ DeepSeek Voice Agent")
    st.markdown("""
    **Voice-to-Voice AI Assistant** - Talk to DeepSeek models using your voice!
    
    **Features:**
    - 🗣️ Voice input with speech recognition
    - 🤖 DeepSeek AI model processing  
    - 🎧 Voice output with text-to-speech
    - 💬 Conversation history
    - ⚙️ Toggle controls for all features
    """)
    
    if not st.session_state.setup_complete:
        st.info("👈 Configure your API keys and initialize the system first!")
        return
    
    # Input Section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_input(
            "💬 Type your message:",
            placeholder="Ask me anything...",
            key="text_input"
        )
    
    with col2:
        if st.session_state.microphone_enabled:
            if st.button("🎤 Voice Input", type="secondary"):
                voice_input = agent.listen_for_speech()
                if voice_input:
                    st.session_state.text_input = voice_input
                    st.rerun()
        else:
            st.button("🎤 Voice Input", disabled=True)
    
    # Process Input
    if text_input:
        # Add user message
        agent.add_to_history("user", text_input)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(text_input)
            st.caption(f"🕒 {agent.conversation_history[-1]['timestamp']}")
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = asyncio.run(agent.get_deepseek_response(
                    text_input, 
                    st.session_state.selected_model
                ))
            
            st.markdown(response)
            
            # Add to history
            agent.add_to_history("assistant", response)
            
            # Generate speech if enabled
            if st.session_state.read_response_enabled:
                with st.spinner("🔊 Generating speech..."):
                    audio_path = asyncio.run(agent.text_to_speech(
                        response,
                        st.session_state.selected_voice
                    ))
                    
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/mp3")
            
            st.caption(f"🕒 {agent.conversation_history[-1]['timestamp']}")
        
        # Clear input
        st.session_state.text_input = ""
        st.rerun()
    
    # Display Conversation History
    if agent.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for entry in agent.conversation_history[:-2]:  # Exclude the current exchange
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                st.caption(f"🕒 {entry['timestamp']}")
    
    # System Status
    with st.expander("ℹ️ System Status"):
        st.markdown(f"""
        **Configuration:**
        - Model: `{st.session_state.selected_model}`
        - Voice: `{st.session_state.selected_voice}`
        - Microphone: {'✅ Enabled' if st.session_state.microphone_enabled else '❌ Disabled'}
        - Read Response: {'✅ Enabled' if st.session_state.read_response_enabled else '❌ Disabled'}
        - Messages: `{len(agent.conversation_history)}`
        - DeepSeek: {'✅ Connected' if agent.deepseek_client else '❌ Not Connected'}
        - OpenAI: {'✅ Connected' if agent.openai_client else '❌ Not Connected'}
        """)

if __name__ == "__main__":
    main()

