#!/usr/bin/env python3
"""
Enhanced Voice Assistant - Unified Implementation

This combines the best features from all voice assistant implementations:
- Multi-agent architecture for complex tasks
- DeepSeek integration for conversational AI
- Advanced audio processing with RealtimeSTT
- Conversation persistence and management
- Enhanced UI with comprehensive controls
"""

import streamlit as st
import asyncio
import os
import sys
import json
import yaml
import uuid
import tempfile
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
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

# Advanced audio processing
try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIME_STT_AVAILABLE = True
except ImportError:
    REALTIME_STT_AVAILABLE = False

# OpenAI for TTS and Whisper
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Rich console for better output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Computer vision and automation (optional)
try:
    import cv2
    import pyautogui
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Enhanced Voice Assistant",
    page_icon="🤖🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration constants
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
VOICE_OPTIONS = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
STT_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v2"]
TRIGGER_WORDS = ["assistant", "hey", "computer", "ai"]

class ConversationManager:
    """Manages conversation history and persistence"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())[:8]
        self.output_dir = Path("conversations")
        self.output_dir.mkdir(exist_ok=True)
        self.conversation_file = self.output_dir / f"{self.conversation_id}.yml"
        self.history = self.load_conversation()
    
    def load_conversation(self) -> List[Dict[str, Any]]:
        """Load conversation history from file"""
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, "r") as f:
                    history = yaml.safe_load(f) or []
                return history
            except Exception as e:
                st.error(f"Error loading conversation: {e}")
                return []
        return []
    
    def save_conversation(self):
        """Save conversation history to file"""
        try:
            with open(self.conversation_file, "w") as f:
                yaml.dump(self.history, f, default_flow_style=False)
        except Exception as e:
            st.error(f"Error saving conversation: {e}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(message)
        self.save_conversation()
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for context"""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        self.save_conversation()

class AdvancedAudioProcessor:
    """Advanced audio processing with multiple STT options"""
    
    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.channels = 1
        self.realtime_recorder = None
        
        # Initialize RealtimeSTT if available
        if REALTIME_STT_AVAILABLE:
            self.setup_realtime_stt()
    
    def setup_realtime_stt(self, model: str = "small.en"):
        """Setup RealtimeSTT recorder"""
        try:
            self.realtime_recorder = AudioToTextRecorder(
                model=model,
                language="en",
                compute_type="float32",
                post_speech_silence_duration=0.8,
                beam_size=5,
                spinner=False,
                print_transcription_time=False,
                enable_realtime_transcription=True,
                realtime_model_type="tiny.en",
                realtime_processing_pause=0.4,
            )
        except Exception as e:
            st.error(f"Error setting up RealtimeSTT: {e}")
            self.realtime_recorder = None
    
    def start_recording(self) -> bool:
        """Start recording audio"""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
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
            
        except Exception as e:
            st.error(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
        
        try:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            
            if self.audio_data:
                return np.concatenate(self.audio_data, axis=0)
            return None
            
        except Exception as e:
            st.error(f"Error stopping recording: {e}")
            return None
    
    def transcribe_with_whisper(self, audio_data: np.ndarray, api_key: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper API"""
        if not OPENAI_AVAILABLE:
            return None
        
        try:
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio_data, self.sample_rate)
            
            # Transcribe with OpenAI
            client = OpenAI(api_key=api_key)
            with open(temp_file.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            # Clean up
            os.unlink(temp_file.name)
            
            return transcript.text
            
        except Exception as e:
            st.error(f"Error transcribing with Whisper: {e}")
            return None
    
    def transcribe_with_realtime_stt(self) -> Optional[str]:
        """Transcribe using RealtimeSTT"""
        if not self.realtime_recorder:
            return None
        
        try:
            # This would be implemented with RealtimeSTT's async methods
            # For now, return a placeholder
            return "[RealtimeSTT transcription would go here]"
        except Exception as e:
            st.error(f"Error with RealtimeSTT: {e}")
            return None

class MultiModalAssistant:
    """Main assistant class with multiple capabilities"""
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.audio_processor = AdvancedAudioProcessor()
        self.temp_dir = Path(tempfile.gettempdir()) / "enhanced_voice_assistant"
        self.temp_dir.mkdir(exist_ok=True)
    
    def compress_response_for_speech(self, text: str, api_key: str) -> str:
        """Compress response for better TTS output"""
        if not OPENAI_AVAILABLE:
            return text
        
        try:
            client = OpenAI(api_key=api_key)
            
            prompt = """
You are an assistant that makes long technical responses more concise for voice output.
Your task is to rephrase the following text to be shorter and more conversational,
while preserving all key information. Focus only on the most important details.
Be brief but clear, as this will be spoken aloud.

IMPORTANT HANDLING FOR CODE BLOCKS:
- Do not include full code blocks in your response
- Instead, briefly mention "I've created code for X" or "Here's a script that does Y"
- For large code blocks, just say something like "I've written a Python function that handles user authentication"
- DO NOT attempt to read out the actual code syntax
- Only describe what the code does in 1 sentence maximum

Original text:
{text}

Return only the compressed text, without any explanation or introduction.
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.format(text=text)}],
                temperature=0.1,
                max_tokens=1024,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error compressing response: {e}")
            return text
    
    def text_to_speech(self, text: str, voice: str, api_key: str) -> Optional[str]:
        """Convert text to speech using OpenAI TTS"""
        if not OPENAI_AVAILABLE:
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=1.0
            )
            
            # Save to file
            audio_file_path = self.temp_dir / f"tts_{uuid.uuid4()}.mp3"
            response.stream_to_file(str(audio_file_path))
            
            return str(audio_file_path)
            
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")
            return None
    
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
    
    def call_openai_api(self, messages: List[Dict], model: str, api_key: str) -> Optional[str]:
        """Call OpenAI API"""
        if not OPENAI_AVAILABLE:
            return None
        
        try:
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
            return None
    
    def detect_trigger_words(self, text: str) -> bool:
        """Check if text contains trigger words"""
        text_lower = text.lower()
        return any(trigger.lower() in text_lower for trigger in TRIGGER_WORDS)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "deepseek_api_key": "",
        "openai_api_key": "",
        "selected_model": "deepseek-chat",
        "selected_voice": "nova",
        "selected_stt_model": "small.en",
        "ai_provider": "deepseek",
        "microphone_listen": False,
        "read_response": False,
        "use_trigger_words": True,
        "use_realtime_stt": False,
        "conversation_history": [],
        "current_response": "",
        "recording": False,
        "assistant": None,
        "last_audio_path": None,
        "conversation_id": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_config():
    """Configure sidebar with comprehensive settings"""
    with st.sidebar:
        st.title("🤖🎤 Enhanced Voice Assistant")
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
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for TTS and Whisper STT"
        )
        
        # AI Provider Selection
        st.markdown("### 🧠 AI Provider")
        st.session_state.ai_provider = st.selectbox(
            "AI Provider",
            options=["deepseek", "openai"],
            index=0 if st.session_state.ai_provider == "deepseek" else 1,
            help="Choose the AI provider for responses"
        )
        
        # Model Selection
        if st.session_state.ai_provider == "deepseek":
            st.session_state.selected_model = st.selectbox(
                "DeepSeek Model",
                options=DEEPSEEK_MODELS,
                index=DEEPSEEK_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in DEEPSEEK_MODELS else 0,
                help="Choose the DeepSeek model"
            )
        else:
            st.session_state.selected_model = st.selectbox(
                "OpenAI Model",
                options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                index=0,
                help="Choose the OpenAI model"
            )
        
        # Voice Settings
        st.markdown("### 🎙️ Voice Settings")
        st.session_state.selected_voice = st.selectbox(
            "TTS Voice",
            options=VOICE_OPTIONS,
            index=VOICE_OPTIONS.index(st.session_state.selected_voice),
            help="Choose voice for text-to-speech"
        )
        
        # STT Settings
        st.markdown("### 🎧 Speech Recognition")
        
        if REALTIME_STT_AVAILABLE:
            st.session_state.use_realtime_stt = st.checkbox(
                "Use RealtimeSTT",
                value=st.session_state.use_realtime_stt,
                help="Use RealtimeSTT for advanced speech recognition"
            )
            
            if st.session_state.use_realtime_stt:
                st.session_state.selected_stt_model = st.selectbox(
                    "STT Model",
                    options=STT_MODELS,
                    index=STT_MODELS.index(st.session_state.selected_stt_model),
                    help="Choose speech recognition model"
                )
        
        # Control Settings
        st.markdown("### 🎛️ Controls")
        
        st.session_state.microphone_listen = st.toggle(
            "🎤 Microphone Listen",
            value=st.session_state.microphone_listen,
            help="Enable voice input"
        )
        
        st.session_state.read_response = st.toggle(
            "🔊 Read Response",
            value=st.session_state.read_response,
            help="Enable text-to-speech for responses"
        )
        
        st.session_state.use_trigger_words = st.toggle(
            "🎯 Use Trigger Words",
            value=st.session_state.use_trigger_words,
            help=f"Require trigger words: {', '.join(TRIGGER_WORDS)}"
        )
        
        # Status indicators
        st.markdown("### 📊 Status")
        
        # API Key status
        deepseek_status = "✅" if st.session_state.deepseek_api_key else "❌"
        openai_status = "✅" if st.session_state.openai_api_key else "❌"
        
        st.markdown(f"DeepSeek API: {deepseek_status}")
        st.markdown(f"OpenAI API: {openai_status}")
        st.markdown(f"Audio Available: {'✅' if AUDIO_AVAILABLE else '❌'}")
        st.markdown(f"RealtimeSTT: {'✅' if REALTIME_STT_AVAILABLE else '❌'}")
        st.markdown(f"Automation: {'✅' if AUTOMATION_AVAILABLE else '❌'}")
        
        # Recording status
        if st.session_state.recording:
            st.markdown("🔴 **Recording...**")
        
        # Conversation management
        st.markdown("### 💬 Conversation")
        
        if st.button("🗑️ Clear Conversation"):
            if st.session_state.assistant:
                st.session_state.assistant.conversation_manager.clear_history()
            st.session_state.conversation_history = []
            st.session_state.current_response = ""
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            if st.session_state.assistant:
                st.session_state.assistant.conversation_manager.save_conversation()
                st.success("Conversation saved!")

def main_interface():
    """Main application interface"""
    st.title("🤖🎤 Enhanced Voice Assistant")
    st.markdown("Advanced voice assistant with multi-modal capabilities and comprehensive AI integration!")
    
    # Check configuration
    required_key = st.session_state.deepseek_api_key if st.session_state.ai_provider == "deepseek" else st.session_state.openai_api_key
    if not required_key:
        st.warning(f"⚠️ Please configure your {st.session_state.ai_provider.title()} API key in the sidebar to get started.")
        return
    
    # Initialize assistant
    if st.session_state.assistant is None:
        st.session_state.assistant = MultiModalAssistant()
    
    # Main interface layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Voice Input Section
        st.markdown("### 🎤 Voice Input")
        
        if st.session_state.microphone_listen and AUDIO_AVAILABLE:
            voice_col1, voice_col2, voice_col3 = st.columns(3)
            
            with voice_col1:
                if st.button("🎤 Start Recording", disabled=st.session_state.recording):
                    if st.session_state.assistant.audio_processor.start_recording():
                        st.session_state.recording = True
                        st.rerun()
            
            with voice_col2:
                if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
                    audio_data = st.session_state.assistant.audio_processor.stop_recording()
                    st.session_state.recording = False
                    
                    if audio_data is not None:
                        # Transcribe audio
                        if st.session_state.use_realtime_stt and REALTIME_STT_AVAILABLE:
                            text_input = st.session_state.assistant.audio_processor.transcribe_with_realtime_stt()
                        else:
                            text_input = st.session_state.assistant.audio_processor.transcribe_with_whisper(
                                audio_data, st.session_state.openai_api_key
                            )
                        
                        if text_input:
                            st.success(f"Recognized: {text_input}")
                            process_user_input(text_input)
                        else:
                            st.error("Failed to transcribe audio")
                    
                    st.rerun()
            
            with voice_col3:
                if st.session_state.recording:
                    st.markdown("🔴 **Recording...**")
                else:
                    st.markdown("⚪ Ready")
        
        elif st.session_state.microphone_listen and not AUDIO_AVAILABLE:
            st.error("Audio libraries not available. Please install: pip install sounddevice soundfile numpy")
        
        # Text Input Section
        st.markdown("### ⌨️ Text Input")
        
        user_input = st.text_area(
            "Type your message:",
            placeholder=f"Ask the AI anything... {f'(Include trigger words: {TRIGGER_WORDS})' if st.session_state.use_trigger_words else ''}",
            height=100,
            key="text_input"
        )
        
        if st.button("📤 Send Message", type="primary"):
            if user_input.strip():
                process_user_input(user_input.strip())
                st.session_state.text_input = ""
                st.rerun()
        
        # Current Response Display
        if st.session_state.current_response:
            st.markdown("### 🤖 AI Response")
            
            with st.container():
                st.markdown(st.session_state.current_response)
            
            # Audio playback if available
            if st.session_state.read_response and st.session_state.last_audio_path:
                st.markdown("### 🔊 Audio Response")
                
                try:
                    with open(st.session_state.last_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                        
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name=f"response_{int(time.time())}.mp3",
                            mime="audio/mp3"
                        )
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
    
    with col2:
        # Conversation History
        st.markdown("### 💬 Conversation History")
        
        if st.session_state.assistant and st.session_state.assistant.conversation_manager.history:
            history = st.session_state.assistant.conversation_manager.get_recent_messages(10)
            
            for i, entry in enumerate(reversed(history)):
                with st.expander(f"{entry['role'].title()} - {entry.get('timestamp', 'Unknown')[:19]}", expanded=(i == 0)):
                    st.markdown(entry['content'])
        else:
            st.info("No conversation history yet. Start by sending a message!")
        
        # Settings Summary
        st.markdown("### ⚙️ Current Settings")
        st.markdown(f"**Provider:** {st.session_state.ai_provider.title()}")
        st.markdown(f"**Model:** {st.session_state.selected_model}")
        st.markdown(f"**Voice:** {st.session_state.selected_voice}")
        st.markdown(f"**STT:** {'RealtimeSTT' if st.session_state.use_realtime_stt else 'Whisper'}")
        st.markdown(f"**Microphone:** {'🟢 ON' if st.session_state.microphone_listen else '🔴 OFF'}")
        st.markdown(f"**Read Response:** {'🟢 ON' if st.session_state.read_response else '🔴 OFF'}")
        st.markdown(f"**Trigger Words:** {'🟢 ON' if st.session_state.use_trigger_words else '🔴 OFF'}")

def process_user_input(user_input: str):
    """Process user input and get AI response"""
    if not user_input.strip():
        return
    
    # Check trigger words if enabled
    if st.session_state.use_trigger_words:
        if not st.session_state.assistant.detect_trigger_words(user_input):
            st.warning(f"Please include one of these trigger words: {', '.join(TRIGGER_WORDS)}")
            return
    
    # Add user message to conversation
    st.session_state.assistant.conversation_manager.add_message("user", user_input)
    
    # Prepare messages for API
    recent_messages = st.session_state.assistant.conversation_manager.get_recent_messages(10)
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in recent_messages]
    
    # Get AI response
    with st.spinner(f"🤖 {st.session_state.ai_provider.title()} is thinking..."):
        if st.session_state.ai_provider == "deepseek":
            response = st.session_state.assistant.call_deepseek_api(
                messages=messages,
                model=st.session_state.selected_model,
                api_key=st.session_state.deepseek_api_key
            )
        else:
            response = st.session_state.assistant.call_openai_api(
                messages=messages,
                model=st.session_state.selected_model,
                api_key=st.session_state.openai_api_key
            )
    
    if response:
        # Add assistant response to conversation
        st.session_state.assistant.conversation_manager.add_message("assistant", response)
        st.session_state.current_response = response
        
        # Generate speech if enabled
        if st.session_state.read_response and st.session_state.openai_api_key and OPENAI_AVAILABLE:
            with st.spinner("🔊 Generating speech..."):
                # Compress response for better TTS
                compressed_response = st.session_state.assistant.compress_response_for_speech(
                    response, st.session_state.openai_api_key
                )
                
                audio_path = st.session_state.assistant.text_to_speech(
                    text=compressed_response,
                    voice=st.session_state.selected_voice,
                    api_key=st.session_state.openai_api_key
                )
                
                if audio_path:
                    st.session_state.last_audio_path = audio_path
                    st.success("🔊 Audio response generated!")
        
        st.success("✅ Response received!")
    else:
        st.error("❌ Failed to get response from AI")

def main():
    """Main application entry point"""
    init_session_state()
    sidebar_config()
    main_interface()

if __name__ == "__main__":
    main()

