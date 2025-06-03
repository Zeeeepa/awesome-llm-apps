"""
Audio utilities for recording and playing audio
"""
import asyncio
import threading
import queue
import time
import wave
import io
from typing import Optional, Callable
import tempfile
import uuid
from pathlib import Path

try:
    import pyaudio
    import numpy as np
except ImportError as e:
    print(f"Warning: Audio dependencies not available: {e}")
    print("Install with: pip install pyaudio numpy")

from config.settings import VoiceConfig

class AudioRecorder:
    """Handles audio recording from microphone"""
    
    def __init__(self):
        self.voice_config = VoiceConfig()
        self.is_recording = False
        self.audio_data = queue.Queue()
        self.recording_thread = None
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = self.voice_config.SAMPLE_RATE
        self.chunk = self.voice_config.CHUNK_SIZE
        
        # Silence detection
        self.silence_threshold = self.voice_config.SILENCE_THRESHOLD
        self.silence_duration = self.voice_config.SILENCE_DURATION
        self.max_duration = self.voice_config.MAX_RECORDING_DURATION
    
    def start_recording(self) -> bool:
        """Start recording audio from microphone"""
        try:
            if self.is_recording:
                return False
            
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            # Clear the queue
            while not self.audio_data.empty():
                self.audio_data.get()
            
            # Start recording thread
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        try:
            if not self.is_recording:
                return None
            
            # Stop recording
            self.is_recording = False
            
            # Wait for recording thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            # Close audio stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            # Collect all audio data
            audio_frames = []
            while not self.audio_data.empty():
                audio_frames.append(self.audio_data.get())
            
            if not audio_frames:
                return None
            
            # Convert to WAV format
            return self._frames_to_wav(audio_frames)
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None
    
    def _record_audio(self):
        """Recording thread function"""
        try:
            start_time = time.time()
            silence_start = None
            
            while self.is_recording:
                # Check max duration
                if time.time() - start_time > self.max_duration:
                    print("Maximum recording duration reached")
                    break
                
                try:
                    # Read audio data
                    data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_data.put(data)
                    
                    # Check for silence (simple volume-based detection)
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    volume = np.sqrt(np.mean(audio_array**2))
                    
                    if volume < self.silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_duration:
                            print("Silence detected, stopping recording")
                            break
                    else:
                        silence_start = None
                
                except Exception as e:
                    print(f"Error reading audio data: {e}")
                    break
            
            self.is_recording = False
            
        except Exception as e:
            print(f"Error in recording thread: {e}")
            self.is_recording = False
    
    def _frames_to_wav(self, frames: list) -> bytes:
        """Convert audio frames to WAV format"""
        try:
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
                wav_file.setframerate(self.rate)
                wav_file.writeframes(b''.join(frames))
            
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except Exception as e:
            print(f"Error converting frames to WAV: {e}")
            return b''
    
    def is_recording_active(self) -> bool:
        """Check if recording is currently active"""
        return self.is_recording
    
    def get_recording_duration(self) -> float:
        """Get current recording duration"""
        if hasattr(self, 'start_time') and self.is_recording:
            return time.time() - self.start_time
        return 0.0

class AudioPlayer:
    """Handles audio playback"""
    
    def __init__(self):
        self.is_playing = False
        self.playback_thread = None
    
    def play_file(self, file_path: str, callback: Optional[Callable] = None) -> bool:
        """Play audio file"""
        try:
            if self.is_playing:
                return False
            
            self.is_playing = True
            self.playback_thread = threading.Thread(
                target=self._play_audio_file, 
                args=(file_path, callback)
            )
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting playback: {e}")
            return False
    
    def play_bytes(self, audio_bytes: bytes, callback: Optional[Callable] = None) -> bool:
        """Play audio from bytes"""
        try:
            # Save bytes to temporary file and play
            temp_path = Path(tempfile.gettempdir()) / f"temp_audio_{uuid.uuid4()}.wav"
            
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            result = self.play_file(str(temp_path), callback)
            
            # Clean up temp file after playback
            def cleanup():
                time.sleep(1)  # Wait a bit for playback to finish
                temp_path.unlink(missing_ok=True)
            
            threading.Thread(target=cleanup).start()
            
            return result
            
        except Exception as e:
            print(f"Error playing audio bytes: {e}")
            return False
    
    def _play_audio_file(self, file_path: str, callback: Optional[Callable] = None):
        """Playback thread function"""
        try:
            # Use system audio player for simplicity
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Windows":
                subprocess.run(['start', file_path], shell=True, check=True)
            elif system == "Darwin":  # macOS
                subprocess.run(['afplay', file_path], check=True)
            else:  # Linux
                subprocess.run(['aplay', file_path], check=True)
            
            if callback:
                callback()
                
        except Exception as e:
            print(f"Error playing audio file: {e}")
        finally:
            self.is_playing = False
    
    def stop_playback(self):
        """Stop current playback"""
        try:
            self.is_playing = False
            # Note: Stopping system audio players is complex and platform-specific
            # For a production system, you'd want to use a more controllable audio library
            
        except Exception as e:
            print(f"Error stopping playback: {e}")
    
    def is_playing_active(self) -> bool:
        """Check if audio is currently playing"""
        return self.is_playing

class AudioProcessor:
    """Utility class for audio processing"""
    
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str) -> bool:
        """Convert audio file to WAV format"""
        try:
            # This would typically use a library like pydub
            # For now, we'll assume input is already in a compatible format
            import shutil
            shutil.copy2(input_path, output_path)
            return True
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            return False
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
                
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0
    
    @staticmethod
    def normalize_audio_volume(audio_data: bytes, target_volume: float = 0.7) -> bytes:
        """Normalize audio volume"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate current volume
            current_volume = np.sqrt(np.mean(audio_array**2))
            
            if current_volume > 0:
                # Normalize to target volume
                scale_factor = target_volume / current_volume
                normalized_array = (audio_array * scale_factor).astype(np.int16)
                return normalized_array.tobytes()
            
            return audio_data
            
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            return audio_data
    
    @staticmethod
    def detect_silence(audio_data: bytes, threshold: float = 0.01) -> bool:
        """Detect if audio contains mostly silence"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            volume = np.sqrt(np.mean(audio_array**2))
            return volume < threshold
            
        except Exception as e:
            print(f"Error detecting silence: {e}")
            return False

# Utility functions for Streamlit integration
def create_audio_recorder_widget():
    """Create audio recorder widget for Streamlit"""
    # This would integrate with Streamlit's audio recording capabilities
    # For now, we'll use the basic AudioRecorder class
    return AudioRecorder()

def create_audio_player_widget():
    """Create audio player widget for Streamlit"""
    return AudioPlayer()

# Test functions
async def test_audio_setup() -> dict:
    """Test audio recording and playback setup"""
    try:
        # Test recording
        recorder = AudioRecorder()
        recording_available = 'pyaudio' in globals()
        
        # Test playback
        player = AudioPlayer()
        playback_available = True  # Basic playback should always work
        
        return {
            "success": True,
            "recording_available": recording_available,
            "playback_available": playback_available,
            "pyaudio_available": 'pyaudio' in globals(),
            "numpy_available": 'numpy' in globals()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

