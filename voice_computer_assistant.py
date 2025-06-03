import streamlit as st
import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import threading
import queue
import time

# Import our custom agents and utilities
from agents.coordinator_agent import CoordinatorAgent
from agents.voice_agent import VoiceAgent
from agents.vision_agent import VisionAgent
from agents.action_agent import ActionAgent
from utils.audio_utils import AudioRecorder, AudioPlayer
from utils.safety_controls import SafetyManager
from config.settings import AppConfig

# Page configuration
st.set_page_config(
    page_title="Voice Computer Assistant",
    page_icon="🎤💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "initialized": False,
        "openai_api_key": "",
        "selected_voice": "nova",
        "safety_mode": "high",
        "coordinator": None,
        "voice_agent": None,
        "vision_agent": None,
        "action_agent": None,
        "audio_recorder": None,
        "audio_player": None,
        "safety_manager": None,
        "recording": False,
        "processing": False,
        "action_history": [],
        "current_screen": None,
        "last_response": "",
        "conversation_history": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_config():
    """Configure sidebar with settings and controls"""
    with st.sidebar:
        st.title("🎤💻 Voice Computer Assistant")
        st.markdown("---")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for voice processing and AI reasoning"
        )
        
        # Voice Settings
        st.markdown("### 🎙️ Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
            help="Choose the voice for audio responses"
        )
        
        # Safety Settings
        st.markdown("### 🛡️ Safety Settings")
        st.session_state.safety_mode = st.selectbox(
            "Safety Mode",
            options=["high", "medium", "low"],
            index=["high", "medium", "low"].index(st.session_state.safety_mode),
            help="High: Confirm all actions, Medium: Confirm sensitive actions, Low: Minimal confirmation"
        )
        
        # System Status
        st.markdown("### 📊 System Status")
        if st.session_state.initialized:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Initialized")
        
        if st.session_state.recording:
            st.info("🎤 Recording...")
        
        if st.session_state.processing:
            st.info("🔄 Processing...")

async def initialize_system():
    """Initialize all system components"""
    if not st.session_state.openai_api_key:
        st.error("Please provide OpenAI API Key in the sidebar")
        return False
    
    try:
        # Initialize configuration
        config = AppConfig(
            openai_api_key=st.session_state.openai_api_key,
            selected_voice=st.session_state.selected_voice,
            safety_mode=st.session_state.safety_mode
        )
        
        # Initialize safety manager
        st.session_state.safety_manager = SafetyManager(config.safety_mode)
        
        # Initialize agents
        st.session_state.voice_agent = VoiceAgent(config)
        st.session_state.vision_agent = VisionAgent(config)
        st.session_state.action_agent = ActionAgent(config, st.session_state.safety_manager)
        
        # Initialize coordinator
        st.session_state.coordinator = CoordinatorAgent(
            voice_agent=st.session_state.voice_agent,
            vision_agent=st.session_state.vision_agent,
            action_agent=st.session_state.action_agent,
            config=config
        )
        
        # Initialize audio components
        st.session_state.audio_recorder = AudioRecorder()
        st.session_state.audio_player = AudioPlayer()
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def main_interface():
    """Main application interface"""
    st.title("🎤💻 Voice Computer Assistant")
    st.markdown("Control your computer with voice commands! Speak naturally and I'll help you navigate, click, type, and interact with your screen.")
    
    # Initialize system if not done
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing system components..."):
                success = asyncio.run(initialize_system())
                if success:
                    st.success("System initialized successfully!")
                    st.rerun()
        return
    
    # Main interface layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Voice Input Section
        st.markdown("### 🎤 Voice Input")
        
        # Voice recording controls
        record_col1, record_col2, record_col3 = st.columns([1, 1, 1])
        
        with record_col1:
            if st.button("🎤 Start Recording", disabled=st.session_state.recording):
                start_recording()
        
        with record_col2:
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
                stop_recording()
        
        with record_col3:
            if st.button("🛑 Emergency Stop"):
                emergency_stop()
        
        # Text Input Alternative
        st.markdown("### ⌨️ Text Input")
        text_input = st.text_area(
            "Or type your command:",
            placeholder="e.g., 'Open Chrome and navigate to Google', 'Click the save button', 'Type hello world'"
        )
        
        if st.button("📤 Send Text Command") and text_input:
            process_command(text_input, is_voice=False)
        
        # Response Display
        st.markdown("### 💬 Assistant Response")
        if st.session_state.last_response:
            st.info(st.session_state.last_response)
            
            # Audio playback if available
            if hasattr(st.session_state, 'last_audio_path') and st.session_state.last_audio_path:
                st.audio(st.session_state.last_audio_path, format="audio/mp3")
    
    with col2:
        # Screen Preview
        st.markdown("### 🖥️ Screen Preview")
        if st.button("📸 Capture Screen"):
            capture_screen_preview()
        
        if st.session_state.current_screen:
            st.image(st.session_state.current_screen, caption="Current Screen", use_column_width=True)
        
        # Action History
        st.markdown("### 📋 Action History")
        if st.session_state.action_history:
            for i, action in enumerate(reversed(st.session_state.action_history[-5:])):
                with st.expander(f"Action {len(st.session_state.action_history) - i}"):
                    st.write(f"**Command:** {action.get('command', 'N/A')}")
                    st.write(f"**Action:** {action.get('action_type', 'N/A')}")
                    st.write(f"**Status:** {action.get('status', 'N/A')}")
                    st.write(f"**Time:** {action.get('timestamp', 'N/A')}")

def start_recording():
    """Start voice recording"""
    st.session_state.recording = True
    # Implementation will be in audio_utils.py
    st.session_state.audio_recorder.start_recording()
    st.rerun()

def stop_recording():
    """Stop voice recording and process"""
    st.session_state.recording = False
    audio_data = st.session_state.audio_recorder.stop_recording()
    
    if audio_data:
        # Convert audio to text and process
        with st.spinner("Processing voice command..."):
            text_command = asyncio.run(st.session_state.voice_agent.speech_to_text(audio_data))
            if text_command:
                process_command(text_command, is_voice=True)
    
    st.rerun()

def process_command(command: str, is_voice: bool = False):
    """Process a command through the coordinator agent"""
    st.session_state.processing = True
    
    try:
        # Add to conversation history
        st.session_state.conversation_history.append({
            "type": "user",
            "content": command,
            "timestamp": datetime.now(),
            "is_voice": is_voice
        })
        
        # Process through coordinator
        result = asyncio.run(st.session_state.coordinator.process_command(command))
        
        # Update session state with results
        st.session_state.last_response = result.get("response", "")
        
        if result.get("audio_path"):
            st.session_state.last_audio_path = result["audio_path"]
        
        # Add to action history
        st.session_state.action_history.append({
            "command": command,
            "action_type": result.get("action_type", "unknown"),
            "status": result.get("status", "completed"),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "details": result.get("details", {})
        })
        
        # Add assistant response to conversation history
        st.session_state.conversation_history.append({
            "type": "assistant",
            "content": st.session_state.last_response,
            "timestamp": datetime.now(),
            "action_performed": result.get("action_performed", False)
        })
        
    except Exception as e:
        st.error(f"Error processing command: {str(e)}")
        st.session_state.last_response = f"Sorry, I encountered an error: {str(e)}"
    
    finally:
        st.session_state.processing = False
        st.rerun()

def capture_screen_preview():
    """Capture current screen for preview"""
    try:
        screenshot = st.session_state.vision_agent.capture_screen()
        st.session_state.current_screen = screenshot
        st.rerun()
    except Exception as e:
        st.error(f"Failed to capture screen: {str(e)}")

def emergency_stop():
    """Emergency stop all operations"""
    st.session_state.recording = False
    st.session_state.processing = False
    
    if st.session_state.action_agent:
        st.session_state.action_agent.emergency_stop()
    
    st.warning("🛑 Emergency stop activated. All operations halted.")
    st.rerun()

def run_async(func, *args, **kwargs):
    """Helper to run async functions in Streamlit"""
    try:
        return asyncio.run(func(*args, **kwargs))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(func(*args, **kwargs))

if __name__ == "__main__":
    init_session_state()
    sidebar_config()
    main_interface()

