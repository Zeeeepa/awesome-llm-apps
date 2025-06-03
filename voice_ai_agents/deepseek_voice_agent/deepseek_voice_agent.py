#!/usr/bin/env python3
"""
DeepSeek Voice Agent - A voice-enabled AI assistant using DeepSeek models

This application provides a Streamlit-based interface for interacting with DeepSeek models
using voice input and output capabilities. Features include:
- DeepSeek API integration with model selection
- Speech-to-text input with microphone toggle
- Text-to-speech output with voice selection and toggle
- Real-time conversation interface
- Session management and conversation history
"""

import os
import sys
import json
import tempfile
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import pyttsx3
import threading
import time
from datetime import datetime
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
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

def init_session_state():
    """Initialize Streamlit session state with default values"""
    defaults = {
        "deepseek_api_key": "",
        "openai_api_key": "",  # For TTS
        "selected_model": "deepseek-chat",
        "selected_voice": "nova",
        "microphone_enabled": True,
        "read_response_enabled": True,
        "conversation_history": [],
        "deepseek_client": None,
        "openai_client": None,
        "is_listening": False,
        "is_speaking": False,
        "recognizer": None,
        "microphone": None,
        "setup_complete": False,
        "audio_playing": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_speech_components():
    """Initialize speech recognition and TTS components"""
    try:
        # Initialize speech recognition
        if st.session_state.recognizer is None:
            st.session_state.recognizer = sr.Recognizer()
            st.session_state.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with st.session_state.microphone as source:
                st.session_state.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Initialize TTS engine
        if st.session_state.tts_engine is None:
            st.session_state.tts_engine = pyttsx3.init()
            
        return True
    except Exception as e:
        st.error(f"Error setting up speech components: {str(e)}")
        return False

def setup_api_clients():
    """Initialize API clients for DeepSeek and OpenAI"""
    try:
        if st.session_state.deepseek_api_key:
            # DeepSeek client
            st.session_state.deepseek_client = OpenAI(
                api_key=st.session_state.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            
        if st.session_state.openai_api_key:
            # OpenAI client for TTS
            st.session_state.openai_client = OpenAI(
                api_key=st.session_state.openai_api_key
            )
            
        if st.session_state.deepseek_client and st.session_state.openai_client:
            st.session_state.setup_complete = True
            return True
    except Exception as e:
        st.error(f"Error setting up API clients: {str(e)}")
        return False
    
    return False

def sidebar_config():
    """Create sidebar configuration interface"""
    with st.sidebar:
        st.title("🔧 Configuration")
        st.markdown("---")
        
        # API Configuration
        st.markdown("### 🔑 API Settings")
        st.session_state.deepseek_api_key = st.text_input(
            "DeepSeek API Key",
            value=st.session_state.deepseek_api_key,
            type="password",
            help="Enter your DeepSeek API key from https://platform.deepseek.com/"
        )
        
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key for text-to-speech functionality"
        )
        
        # Model Selection
        st.markdown("### 🤖 Model Selection")
        st.session_state.selected_model = st.selectbox(
            "Choose DeepSeek Model",
            options=DEEPSEEK_MODELS,
            index=DEEPSEEK_MODELS.index(st.session_state.selected_model),
            help="Select the DeepSeek model for conversation"
        )
        
        st.markdown("---")
        
        # Voice Settings
        st.markdown("### 🎤 Voice Settings")
        st.session_state.selected_voice = st.selectbox(
            "TTS Voice",
            options=TTS_VOICES,
            index=TTS_VOICES.index(st.session_state.selected_voice),
            help="Choose the voice for text-to-speech output"
        )
        
        # Feature Toggles
        st.markdown("### ⚙️ Feature Controls")
        st.session_state.microphone_enabled = st.toggle(
            "🎙️ Microphone Listen",
            value=st.session_state.microphone_enabled,
            help="Enable/disable voice input via microphone"
        )
        
        st.session_state.read_response_enabled = st.toggle(
            "🔊 Read Response",
            value=st.session_state.read_response_enabled,
            help="Enable/disable text-to-speech for responses"
        )
        
        st.markdown("---")
        
        # System Controls
        if st.button("🚀 Initialize System", type="primary"):
            if st.session_state.deepseek_api_key and st.session_state.openai_api_key:
                with st.spinner("Setting up system..."):
                    speech_setup = setup_speech_components()
                    api_setup = setup_api_clients()
                    
                    if speech_setup and api_setup:
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ System initialization failed!")
            else:
                st.error("Please enter both DeepSeek and OpenAI API keys!")
        
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.success("Conversation history cleared!")
            st.rerun()

def listen_for_speech():
    """Capture speech input from microphone"""
    if not st.session_state.microphone_enabled or not st.session_state.setup_complete:
        return None
    
    try:
        with st.session_state.microphone as source:
            st.info("🎤 Listening... Speak now!")
            # Listen for audio with timeout
            audio = st.session_state.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
        st.info("🔄 Processing speech...")
        # Use Google's speech recognition
        text = st.session_state.recognizer.recognize_google(audio)
        return text
        
    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("🤷 Could not understand the speech. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Speech recognition error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during speech recognition: {str(e)}")
        return None

async def get_deepseek_response(message: str) -> str:
    """Get response from DeepSeek API"""
    try:
        # Prepare conversation history for context
        messages = []
        
        # Add conversation history (last 10 messages for context)
        for entry in st.session_state.conversation_history[-10:]:
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call DeepSeek API
        response = st.session_state.deepseek_client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error getting response from DeepSeek: {str(e)}"

async def text_to_speech(text: str):
    """Convert text to speech using OpenAI TTS"""
    if not st.session_state.read_response_enabled or not st.session_state.openai_client:
        return None
    
    try:
        # Generate speech using OpenAI TTS
        response = st.session_state.openai_client.audio.speech.create(
            model="tts-1",
            voice=st.session_state.selected_voice,
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

def display_conversation():
    """Display conversation history"""
    st.markdown("### 💬 Conversation")
    
    if not st.session_state.conversation_history:
        st.info("Start a conversation by typing a message or using voice input!")
        return
    
    # Display conversation in chat format
    for entry in st.session_state.conversation_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            
            # Show timestamp
            if "timestamp" in entry:
                st.caption(f"🕒 {entry['timestamp']}")
            
            # Show audio player for assistant responses if available
            if entry["role"] == "assistant" and "audio_path" in entry and entry["audio_path"]:
                if os.path.exists(entry["audio_path"]):
                    st.audio(entry["audio_path"], format="audio/mp3")

async def process_user_input(user_input: str):
    """Process user input and generate response"""
    if not user_input.strip():
        return
    
    # Add user message to history
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"🕒 {timestamp}")
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = await get_deepseek_response(user_input)
        
        st.markdown(response)
        
        # Generate audio if enabled
        audio_path = None
        if st.session_state.read_response_enabled:
            with st.spinner("🔊 Generating speech..."):
                audio_path = await text_to_speech(response)
                
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/mp3")
        
        # Add assistant response to history
        response_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": response_timestamp,
            "audio_path": audio_path
        })
        
        st.caption(f"🕒 {response_timestamp}")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="DeepSeek Voice Agent",
        page_icon="🎙���",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Create sidebar configuration
    sidebar_config()
    
    # Main interface
    st.title("🎙️ DeepSeek Voice Agent")
    st.markdown("""
    Welcome to the DeepSeek Voice Agent! This application allows you to:
    - 🗣️ Interact with DeepSeek models using voice or text
    - 🎧 Listen to AI responses with text-to-speech
    - 🔄 Toggle voice features on/off as needed
    - 💾 Maintain conversation history
    
    **Get started:** Configure your API key and settings in the sidebar, then initialize the system!
    """)
    
    if not st.session_state.setup_complete:
        st.info("👈 Please configure and initialize the system using the sidebar first!")
        return
    
    # Input methods
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input
        text_input = st.text_input(
            "💬 Type your message:",
            placeholder="Ask me anything...",
            key="text_input"
        )
    
    with col2:
        # Voice input button
        if st.session_state.microphone_enabled:
            if st.button("🎤 Voice Input", type="secondary"):
                voice_input = listen_for_speech()
                if voice_input:
                    st.session_state.text_input = voice_input
                    st.rerun()
        else:
            st.button("🎤 Voice Input", disabled=True, help="Enable microphone in sidebar")
    
    # Process input
    if text_input:
        asyncio.run(process_user_input(text_input))
        # Clear input after processing
        st.session_state.text_input = ""
        st.rerun()
    
    # Display conversation
    display_conversation()
    
    # Status information
    with st.expander("ℹ️ System Status", expanded=False):
        st.markdown(f"""
        **Configuration:**
        - Model: `{st.session_state.selected_model}`
        - Voice: `{st.session_state.selected_voice}`
        - Microphone: {'✅ Enabled' if st.session_state.microphone_enabled else '❌ Disabled'}
        - Read Response: {'✅ Enabled' if st.session_state.read_response_enabled else '❌ Disabled'}
        - Messages in History: `{len(st.session_state.conversation_history)}`
        - DeepSeek API: {'✅ Connected' if st.session_state.deepseek_client else '❌ Not Connected'}
        - OpenAI API: {'✅ Connected' if st.session_state.openai_client else '❌ Not Connected'}
        """)

if __name__ == "__main__":
    main()
