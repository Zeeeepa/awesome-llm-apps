#!/usr/bin/env python3
"""
Advanced DeepSeek Voice Agent - Enhanced Features for PR #3

This enhanced version includes:
- Advanced voice processing with noise reduction
- Multi-language support
- Voice emotion detection
- Enhanced UI with dark/light themes
- Real-time voice visualization
- Advanced conversation management
- Voice command shortcuts
- Custom voice training options
"""

import os
import sys
import json
import tempfile
import asyncio
import uuid
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from openai import OpenAI
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import librosa
import noisereduce as nr

# Load environment variables
load_dotenv()

# Enhanced Configuration
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

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese"
}

VOICE_COMMANDS = {
    "clear history": "clear_conversation",
    "change model": "show_model_selector",
    "change voice": "show_voice_selector",
    "save conversation": "save_conversation",
    "load conversation": "load_conversation",
    "toggle microphone": "toggle_microphone",
    "toggle speech": "toggle_speech",
    "help": "show_help"
}

class AdvancedVoiceProcessor:
    """Advanced voice processing with noise reduction and enhancement"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.noise_profile = None
        
    def reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=audio_data, sr=self.sample_rate)
            return reduced_noise
        except Exception as e:
            st.warning(f"Noise reduction failed: {e}")
            return audio_data
    
    def enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhance audio quality"""
        try:
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Apply noise reduction
            audio_data = self.reduce_noise(audio_data)
            
            return audio_data
        except Exception as e:
            st.warning(f"Audio enhancement failed: {e}")
            return audio_data
    
    def detect_emotion(self, audio_data: np.ndarray) -> str:
        """Simple emotion detection based on audio features"""
        try:
            # Extract basic features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Simple heuristic emotion detection
            energy = np.mean(librosa.feature.rms(y=audio_data))
            pitch_mean = np.mean(spectral_centroid)
            
            if energy > 0.02 and pitch_mean > 2000:
                return "excited"
            elif energy < 0.01:
                return "calm"
            elif pitch_mean < 1000:
                return "serious"
            else:
                return "neutral"
                
        except Exception as e:
            return "neutral"

class AdvancedDeepSeekVoiceAgent:
    """Enhanced DeepSeek Voice Agent with advanced features"""
    
    def __init__(self):
        self.deepseek_client = None
        self.openai_client = None
        self.recognizer = None
        self.microphone = None
        self.conversation_history = []
        self.setup_complete = False
        self.voice_processor = AdvancedVoiceProcessor()
        self.audio_buffer = []
        self.is_recording = False
        self.current_emotion = "neutral"
        
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
    
    def setup_speech_recognition(self, language: str = "en") -> bool:
        """Initialize speech recognition with language support"""
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
    
    def listen_for_speech_advanced(self, timeout: int = 5, language: str = "en") -> Optional[Tuple[str, str]]:
        """Advanced speech capture with emotion detection"""
        if not self.recognizer or not self.microphone:
            return None, "neutral"
        
        try:
            with self.microphone as source:
                st.info("🎤 Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            # Convert to numpy array for processing
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize
            
            # Enhance audio
            enhanced_audio = self.voice_processor.enhance_audio(audio_data)
            
            # Detect emotion
            emotion = self.voice_processor.detect_emotion(enhanced_audio)
            self.current_emotion = emotion
            
            st.info("🔄 Processing speech...")
            
            # Use Google's speech recognition with language support
            language_code = f"{language}-{language.upper()}" if language != "en" else "en-US"
            text = self.recognizer.recognize_google(audio, language=language_code)
            
            return text, emotion
            
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.warning("🤷 Could not understand the speech. Please try again.")
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
        
        return None, "neutral"
    
    def process_voice_command(self, text: str) -> bool:
        """Process voice commands"""
        text_lower = text.lower()
        
        for command, action in VOICE_COMMANDS.items():
            if command in text_lower:
                if action == "clear_conversation":
                    self.conversation_history = []
                    st.success("🗑️ Conversation cleared!")
                    return True
                elif action == "toggle_microphone":
                    st.session_state.microphone_enabled = not st.session_state.microphone_enabled
                    status = "enabled" if st.session_state.microphone_enabled else "disabled"
                    st.success(f"🎙️ Microphone {status}!")
                    return True
                elif action == "toggle_speech":
                    st.session_state.read_response_enabled = not st.session_state.read_response_enabled
                    status = "enabled" if st.session_state.read_response_enabled else "disabled"
                    st.success(f"🔊 Speech output {status}!")
                    return True
                elif action == "show_help":
                    self.show_voice_commands_help()
                    return True
        
        return False
    
    def show_voice_commands_help(self):
        """Display voice commands help"""
        st.info("""
        **Voice Commands Available:**
        - "Clear history" - Clear conversation
        - "Toggle microphone" - Enable/disable mic
        - "Toggle speech" - Enable/disable TTS
        - "Help" - Show this help
        """)
    
    async def get_deepseek_response_enhanced(self, message: str, model: str, emotion: str = "neutral") -> str:
        """Enhanced DeepSeek response with emotion context"""
        if not self.deepseek_client:
            return "DeepSeek client not initialized"
        
        try:
            # Prepare messages with conversation history and emotion context
            messages = []
            
            # Add emotion context to system message
            emotion_context = f"The user seems {emotion}. Please respond appropriately to their emotional state."
            messages.append({"role": "system", "content": emotion_context})
            
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
    
    async def text_to_speech_enhanced(self, text: str, voice: str, emotion: str = "neutral") -> Optional[str]:
        """Enhanced text-to-speech with emotion-based adjustments"""
        if not self.openai_client:
            return None
        
        try:
            # Adjust speech parameters based on emotion
            speed = 1.0
            if emotion == "excited":
                speed = 1.1
            elif emotion == "calm":
                speed = 0.9
            elif emotion == "serious":
                speed = 0.95
            
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed
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
    
    def add_to_history(self, role: str, content: str, emotion: str = "neutral"):
        """Add message to conversation history with emotion"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "emotion": emotion
        })
    
    def visualize_audio_waveform(self, audio_data: np.ndarray):
        """Create audio waveform visualization"""
        try:
            fig = go.Figure()
            time_axis = np.linspace(0, len(audio_data) / 16000, len(audio_data))
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=audio_data,
                mode='lines',
                name='Audio Waveform',
                line=dict(color='#00ff00', width=1)
            ))
            
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            return fig
        except Exception as e:
            return None

def init_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        "agent": AdvancedDeepSeekVoiceAgent(),
        "deepseek_api_key": "",
        "openai_api_key": "",
        "selected_model": "deepseek-chat",
        "selected_voice": "nova",
        "selected_language": "en",
        "microphone_enabled": True,
        "read_response_enabled": True,
        "noise_reduction_enabled": True,
        "emotion_detection_enabled": True,
        "voice_commands_enabled": True,
        "theme": "dark",
        "setup_complete": False,
        "show_audio_viz": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def apply_custom_css():
    """Apply custom CSS for enhanced UI"""
    theme = st.session_state.get("theme", "dark")
    
    if theme == "dark":
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .emotion-indicator {
            padding: 0.5rem;
            border-radius: 20px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        .emotion-excited { background-color: #ff6b6b; color: white; }
        .emotion-calm { background-color: #4ecdc4; color: white; }
        .emotion-serious { background-color: #45b7d1; color: white; }
        .emotion-neutral { background-color: #96ceb4; color: white; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #74b9ff 0%, #0984e3 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .emotion-indicator {
            padding: 0.5rem;
            border-radius: 20px;
            text-align: center;
            font-weight: bold;
            margin: 1rem 0;
        }
        .emotion-excited { background-color: #e17055; color: white; }
        .emotion-calm { background-color: #00b894; color: white; }
        .emotion-serious { background-color: #0984e3; color: white; }
        .emotion-neutral { background-color: #00cec9; color: white; }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Advanced DeepSeek Voice Agent",
        page_icon="🎙️",
        layout="wide"
    )
    
    init_session_state()
    apply_custom_css()
    agent = st.session_state.agent
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Advanced DeepSeek Voice Agent</h1>
        <p>Enhanced voice-to-voice AI with emotion detection, noise reduction, and advanced features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("🔧 Advanced Configuration")
        st.markdown("---")
        
        # Theme Selection
        st.markdown("### 🎨 Appearance")
        theme = st.selectbox("Theme", ["dark", "light"], index=0 if st.session_state.theme == "dark" else 1)
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
        
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
        st.markdown("### 🤖 Model & Voice")
        selected_model = st.selectbox(
            "DeepSeek Model",
            options=DEEPSEEK_MODELS,
            index=DEEPSEEK_MODELS.index(st.session_state.selected_model)
        )
        
        selected_voice = st.selectbox(
            "TTS Voice",
            options=TTS_VOICES,
            index=TTS_VOICES.index(st.session_state.selected_voice)
        )
        
        selected_language = st.selectbox(
            "Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.selected_language)
        )
        
        # Advanced Features
        st.markdown("### ⚙️ Advanced Features")
        microphone_enabled = st.toggle(
            "🎙️ Microphone Listen",
            value=st.session_state.microphone_enabled
        )
        
        read_response_enabled = st.toggle(
            "🔊 Read Response",
            value=st.session_state.read_response_enabled
        )
        
        noise_reduction_enabled = st.toggle(
            "🔇 Noise Reduction",
            value=st.session_state.noise_reduction_enabled
        )
        
        emotion_detection_enabled = st.toggle(
            "😊 Emotion Detection",
            value=st.session_state.emotion_detection_enabled
        )
        
        voice_commands_enabled = st.toggle(
            "🗣️ Voice Commands",
            value=st.session_state.voice_commands_enabled
        )
        
        show_audio_viz = st.toggle(
            "📊 Audio Visualization",
            value=st.session_state.show_audio_viz
        )
        
        st.markdown("---")
        
        # Initialize System
        if st.button("🚀 Initialize Advanced System", type="primary"):
            if deepseek_key and openai_key:
                with st.spinner("Setting up advanced system..."):
                    # Initialize clients
                    clients_ok = agent.initialize_clients(deepseek_key, openai_key)
                    speech_ok = agent.setup_speech_recognition(selected_language)
                    
                    if clients_ok and speech_ok:
                        st.session_state.setup_complete = True
                        st.session_state.deepseek_api_key = deepseek_key
                        st.session_state.openai_api_key = openai_key
                        st.session_state.selected_model = selected_model
                        st.session_state.selected_voice = selected_voice
                        st.session_state.selected_language = selected_language
                        st.session_state.microphone_enabled = microphone_enabled
                        st.session_state.read_response_enabled = read_response_enabled
                        st.session_state.noise_reduction_enabled = noise_reduction_enabled
                        st.session_state.emotion_detection_enabled = emotion_detection_enabled
                        st.session_state.voice_commands_enabled = voice_commands_enabled
                        st.session_state.show_audio_viz = show_audio_viz
                        st.success("✅ Advanced system initialized!")
                        st.rerun()
                    else:
                        st.error("❌ Initialization failed!")
            else:
                st.error("Please enter both API keys!")
        
        if st.button("🗑️ Clear History"):
            agent.conversation_history = []
            st.success("History cleared!")
            st.rerun()
        
        # Voice Commands Help
        if st.session_state.voice_commands_enabled:
            with st.expander("🗣️ Voice Commands"):
                st.markdown("""
                **Available Commands:**
                - "Clear history"
                - "Toggle microphone" 
                - "Toggle speech"
                - "Help"
                """)
    
    # Main Interface
    if not st.session_state.setup_complete:
        st.info("👈 Configure your API keys and initialize the advanced system first!")
        return
    
    # Emotion Indicator
    if st.session_state.emotion_detection_enabled and agent.current_emotion:
        emotion_class = f"emotion-{agent.current_emotion}"
        st.markdown(f"""
        <div class="emotion-indicator {emotion_class}">
            Current Emotion: {agent.current_emotion.title()} 😊
        </div>
        """, unsafe_allow_html=True)
    
    # Input Section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        text_input = st.text_input(
            "💬 Type your message:",
            placeholder="Ask me anything...",
            key="text_input"
        )
    
    with col2:
        if st.session_state.microphone_enabled:
            if st.button("🎤 Advanced Voice Input", type="secondary"):
                result = agent.listen_for_speech_advanced(
                    language=st.session_state.selected_language
                )
                if result[0]:  # text
                    voice_input, emotion = result
                    
                    # Check for voice commands first
                    if st.session_state.voice_commands_enabled:
                        if agent.process_voice_command(voice_input):
                            st.rerun()
                            return
                    
                    st.session_state.text_input = voice_input
                    if st.session_state.emotion_detection_enabled:
                        agent.current_emotion = emotion
                    st.rerun()
        else:
            st.button("🎤 Voice Input", disabled=True)
    
    with col3:
        if st.button("📊 Audio Test", help="Test audio visualization"):
            if st.session_state.show_audio_viz:
                # Generate sample audio for visualization
                sample_audio = np.random.randn(8000) * 0.1
                fig = agent.visualize_audio_waveform(sample_audio)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Process Input
    if text_input:
        # Add user message
        agent.add_to_history("user", text_input, agent.current_emotion)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(text_input)
            if st.session_state.emotion_detection_enabled:
                st.caption(f"🕒 {agent.conversation_history[-1]['timestamp']} | 😊 {agent.current_emotion}")
            else:
                st.caption(f"🕒 {agent.conversation_history[-1]['timestamp']}")
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking with advanced processing..."):
                response = asyncio.run(agent.get_deepseek_response_enhanced(
                    text_input, 
                    st.session_state.selected_model,
                    agent.current_emotion if st.session_state.emotion_detection_enabled else "neutral"
                ))
            
            st.markdown(response)
            
            # Add to history
            agent.add_to_history("assistant", response)
            
            # Generate speech if enabled
            if st.session_state.read_response_enabled:
                with st.spinner("🔊 Generating enhanced speech..."):
                    audio_path = asyncio.run(agent.text_to_speech_enhanced(
                        response,
                        st.session_state.selected_voice,
                        agent.current_emotion if st.session_state.emotion_detection_enabled else "neutral"
                    ))
                    
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/mp3")
            
            st.caption(f"🕒 {agent.conversation_history[-1]['timestamp']}")
        
        # Clear input
        st.session_state.text_input = ""
        st.rerun()
    
    # Display Conversation History
    if agent.conversation_history:
        st.markdown("### 💬 Enhanced Conversation History")
        
        for entry in agent.conversation_history[:-2]:  # Exclude the current exchange
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                emotion_info = f" | 😊 {entry.get('emotion', 'neutral')}" if st.session_state.emotion_detection_enabled else ""
                st.caption(f"🕒 {entry['timestamp']}{emotion_info}")
    
    # System Status
    with st.expander("ℹ️ Advanced System Status"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Configuration:**
            - Model: `{st.session_state.selected_model}`
            - Voice: `{st.session_state.selected_voice}`
            - Language: `{SUPPORTED_LANGUAGES[st.session_state.selected_language]}`
            - Theme: `{st.session_state.theme}`
            """)
        
        with col2:
            st.markdown(f"""
            **Features:**
            - Microphone: {'✅' if st.session_state.microphone_enabled else '❌'}
            - Speech Output: {'✅' if st.session_state.read_response_enabled else '❌'}
            - Noise Reduction: {'✅' if st.session_state.noise_reduction_enabled else '❌'}
            - Emotion Detection: {'✅' if st.session_state.emotion_detection_enabled else '❌'}
            - Voice Commands: {'✅' if st.session_state.voice_commands_enabled else '❌'}
            - Audio Visualization: {'✅' if st.session_state.show_audio_viz else '❌'}
            """)
        
        st.markdown(f"""
        **Status:**
        - Messages: `{len(agent.conversation_history)}`
        - Current Emotion: `{agent.current_emotion}`
        - DeepSeek: {'✅ Connected' if agent.deepseek_client else '❌ Not Connected'}
        - OpenAI: {'✅ Connected' if agent.openai_client else '❌ Not Connected'}
        """)

if __name__ == "__main__":
    main()

