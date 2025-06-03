"""
Voice Agent for handling speech-to-text and text-to-speech operations
"""
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import io
import wave

from openai import AsyncOpenAI
from config.settings import AppConfig, VoiceConfig

class VoiceAgent:
    """Handles all voice-related operations including STT and TTS"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.voice_config = VoiceConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_assistant" / "audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def speech_to_text(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """Convert speech audio to text using OpenAI Whisper"""
        try:
            # Save audio data to temporary file
            audio_file_path = self.temp_dir / f"input_{uuid.uuid4()}.wav"
            
            with open(audio_file_path, "wb") as f:
                f.write(audio_data)
            
            # Transcribe using OpenAI Whisper
            with open(audio_file_path, "rb") as audio_file:
                transcript = await self.client.audio.transcriptions.create(
                    model=self.voice_config.STT_MODEL,
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            
            # Clean up temporary file
            audio_file_path.unlink(missing_ok=True)
            
            return transcript.strip() if transcript else None
            
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return None
    
    async def text_to_speech(self, text: str, voice: Optional[str] = None) -> Optional[str]:
        """Convert text to speech using OpenAI TTS"""
        try:
            selected_voice = voice or self.config.selected_voice
            
            # Generate speech
            response = await self.client.audio.speech.create(
                model=self.voice_config.TTS_MODEL,
                voice=selected_voice,
                input=text,
                speed=self.voice_config.TTS_SPEED,
                instructions=self.voice_config.TTS_INSTRUCTIONS
            )
            
            # Save to temporary file
            audio_file_path = self.temp_dir / f"output_{uuid.uuid4()}.mp3"
            
            with open(audio_file_path, "wb") as f:
                f.write(response.content)
            
            return str(audio_file_path)
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return None
    
    async def text_to_speech_with_instructions(self, text: str, instructions: str, voice: Optional[str] = None) -> Optional[str]:
        """Convert text to speech with custom instructions for tone/style"""
        try:
            selected_voice = voice or self.config.selected_voice
            
            # Generate speech with custom instructions
            response = await self.client.audio.speech.create(
                model=self.voice_config.TTS_MODEL,
                voice=selected_voice,
                input=text,
                speed=self.voice_config.TTS_SPEED,
                instructions=instructions
            )
            
            # Save to temporary file
            audio_file_path = self.temp_dir / f"output_{uuid.uuid4()}.mp3"
            
            with open(audio_file_path, "wb") as f:
                f.write(response.content)
            
            return str(audio_file_path)
            
        except Exception as e:
            print(f"Error in text-to-speech with instructions: {e}")
            return None
    
    async def process_voice_command(self, audio_data: bytes) -> Dict[str, Any]:
        """Process a complete voice command from audio to text"""
        try:
            # Convert speech to text
            text_command = await self.speech_to_text(audio_data)
            
            if not text_command:
                return {
                    "success": False,
                    "error": "Could not understand the audio",
                    "text": None
                }
            
            # Analyze the command for intent and parameters
            intent_analysis = await self._analyze_command_intent(text_command)
            
            return {
                "success": True,
                "text": text_command,
                "intent": intent_analysis.get("intent"),
                "parameters": intent_analysis.get("parameters", {}),
                "confidence": intent_analysis.get("confidence", 0.0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": None
            }
    
    async def _analyze_command_intent(self, text: str) -> Dict[str, Any]:
        """Analyze command text to extract intent and parameters"""
        try:
            # Use OpenAI to analyze the command intent
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a computer command analyzer. Analyze user commands and extract:
                        1. Intent (click, type, open, navigate, scroll, etc.)
                        2. Target (what to interact with)
                        3. Parameters (coordinates, text to type, etc.)
                        4. Confidence (0.0-1.0)
                        
                        Respond in JSON format:
                        {
                            "intent": "action_type",
                            "target": "element_description",
                            "parameters": {"key": "value"},
                            "confidence": 0.95
                        }
                        
                        Common intents: click, double_click, right_click, type, open_application, 
                        navigate_to, scroll, drag, select, copy, paste, save, close, minimize, maximize"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this command: '{text}'"
                    }
                ],
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error analyzing command intent: {e}")
            return {
                "intent": "unknown",
                "target": text,
                "parameters": {},
                "confidence": 0.0
            }
    
    async def generate_response(self, action_result: Dict[str, Any], original_command: str) -> str:
        """Generate a natural language response about the action performed"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful computer assistant. Generate brief, natural responses 
                        about actions you've performed. Be conversational and friendly. Keep responses short 
                        (1-2 sentences). Examples:
                        - "I clicked the save button for you."
                        - "I opened Chrome and navigated to Google."
                        - "I typed your message in the text field."
                        - "I couldn't find that button on the screen."
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Original command: "{original_command}"
                        Action result: {action_result}
                        
                        Generate a brief response about what happened.
                        """
                    }
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I tried to help, but something went wrong."
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary audio files"""
        try:
            import time
            current_time = time.time()
            
            for file_path in self.temp_dir.glob("*.mp3"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    file_path.unlink(missing_ok=True)
                    
            for file_path in self.temp_dir.glob("*.wav"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    file_path.unlink(missing_ok=True)
                    
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        return self.config.voice_options
    
    def set_voice(self, voice: str):
        """Set the current TTS voice"""
        if voice in self.config.voice_options:
            self.config.selected_voice = voice
        else:
            raise ValueError(f"Voice '{voice}' not available. Available voices: {self.config.voice_options}")
    
    async def test_voice_setup(self) -> Dict[str, Any]:
        """Test the voice setup and return status"""
        try:
            # Test TTS
            test_audio = await self.text_to_speech("Voice system test successful.")
            
            return {
                "success": True,
                "tts_working": test_audio is not None,
                "selected_voice": self.config.selected_voice,
                "available_voices": self.config.voice_options
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tts_working": False
            }

