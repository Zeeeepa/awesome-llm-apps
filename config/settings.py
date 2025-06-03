"""
Configuration settings for the Voice Computer Assistant
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AppConfig:
    """Main application configuration"""
    openai_api_key: str
    selected_voice: str = "nova"
    safety_mode: str = "high"
    
    # Voice settings
    voice_options: List[str] = None
    speech_recognition_language: str = "en-US"
    
    # Screen capture settings
    screen_capture_quality: int = 85
    ui_detection_confidence: float = 0.8
    
    # Action settings
    action_delay: float = 0.5  # Delay between actions in seconds
    confirmation_timeout: int = 30  # Seconds to wait for user confirmation
    
    # Safety settings
    restricted_applications: List[str] = None
    sensitive_actions: List[str] = None
    
    def __post_init__(self):
        if self.voice_options is None:
            self.voice_options = [
                "alloy", "ash", "ballad", "coral", "echo", 
                "fable", "onyx", "nova", "sage", "shimmer", "verse"
            ]
        
        if self.restricted_applications is None:
            self.restricted_applications = [
                "Terminal", "Command Prompt", "PowerShell", 
                "System Preferences", "Control Panel", "Registry Editor"
            ]
        
        if self.sensitive_actions is None:
            self.sensitive_actions = [
                "delete", "remove", "uninstall", "format", 
                "shutdown", "restart", "logout", "sudo", "admin"
            ]
    
    @property
    def safety_config(self) -> Dict:
        """Get safety configuration based on safety mode"""
        configs = {
            "high": {
                "require_confirmation": True,
                "confirm_all_actions": True,
                "allow_system_actions": False,
                "allow_file_operations": False,
                "max_actions_per_command": 3
            },
            "medium": {
                "require_confirmation": True,
                "confirm_all_actions": False,
                "allow_system_actions": False,
                "allow_file_operations": True,
                "max_actions_per_command": 5
            },
            "low": {
                "require_confirmation": False,
                "confirm_all_actions": False,
                "allow_system_actions": True,
                "allow_file_operations": True,
                "max_actions_per_command": 10
            }
        }
        return configs.get(self.safety_mode, configs["high"])
    
    def is_action_allowed(self, action_type: str, target: str = "") -> bool:
        """Check if an action is allowed based on current safety settings"""
        safety_config = self.safety_config
        
        # Check for sensitive actions
        if any(sensitive in action_type.lower() or sensitive in target.lower() 
               for sensitive in self.sensitive_actions):
            return safety_config.get("allow_system_actions", False)
        
        # Check for file operations
        if action_type.lower() in ["file_open", "file_save", "file_delete"]:
            return safety_config.get("allow_file_operations", True)
        
        return True
    
    def requires_confirmation(self, action_type: str, target: str = "") -> bool:
        """Check if an action requires user confirmation"""
        safety_config = self.safety_config
        
        if safety_config.get("confirm_all_actions", False):
            return True
        
        if not safety_config.get("require_confirmation", True):
            return False
        
        # Always confirm sensitive actions
        if any(sensitive in action_type.lower() or sensitive in target.lower() 
               for sensitive in self.sensitive_actions):
            return True
        
        # Confirm system-level actions
        if action_type.lower() in ["system_command", "application_launch", "file_delete"]:
            return True
        
        return False

class VoiceConfig:
    """Voice-specific configuration"""
    
    # OpenAI TTS settings
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_SPEED = 1.0
    TTS_INSTRUCTIONS = """
    You are a helpful computer assistant. Speak clearly and naturally, 
    as if you're helping a friend with their computer. Use a friendly, 
    professional tone. Keep responses concise but informative.
    """
    
    # Speech recognition settings
    STT_MODEL = "whisper-1"
    AUDIO_FORMAT = "mp3"
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    
    # Recording settings
    MAX_RECORDING_DURATION = 30  # seconds
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 2  # seconds of silence to auto-stop

class VisionConfig:
    """Computer vision configuration"""
    
    # Screen capture settings
    SCREENSHOT_FORMAT = "PNG"
    SCREENSHOT_QUALITY = 95
    
    # UI element detection
    OCR_CONFIDENCE = 0.7
    BUTTON_DETECTION_CONFIDENCE = 0.8
    TEXT_FIELD_DETECTION_CONFIDENCE = 0.7
    
    # Image processing
    RESIZE_FACTOR = 0.5  # Resize images for faster processing
    BLUR_KERNEL_SIZE = 3
    
    # Element types to detect
    UI_ELEMENTS = [
        "button", "text_field", "dropdown", "checkbox", 
        "radio_button", "link", "menu", "icon", "window"
    ]

class ActionConfig:
    """Action execution configuration"""
    
    # Mouse settings
    CLICK_DURATION = 0.1
    DOUBLE_CLICK_INTERVAL = 0.3
    DRAG_SPEED = 1.0
    
    # Keyboard settings
    TYPE_INTERVAL = 0.05  # Delay between keystrokes
    KEY_PRESS_DURATION = 0.1
    
    # Scroll settings
    SCROLL_SPEED = 3
    SCROLL_PAUSE = 0.2
    
    # Window management
    WINDOW_SWITCH_DELAY = 0.5
    APPLICATION_LAUNCH_TIMEOUT = 10
    
    # Safety delays
    ACTION_CONFIRMATION_DELAY = 1.0
    BETWEEN_ACTION_DELAY = 0.5

# Environment variable defaults
DEFAULT_ENV_VARS = {
    "OPENAI_API_KEY": "",
    "VOICE_ASSISTANT_LOG_LEVEL": "INFO",
    "VOICE_ASSISTANT_TEMP_DIR": "/tmp/voice_assistant",
    "VOICE_ASSISTANT_AUDIO_DIR": "/tmp/voice_assistant/audio",
    "VOICE_ASSISTANT_SCREENSHOTS_DIR": "/tmp/voice_assistant/screenshots"
}

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with default fallback"""
    return os.getenv(key, DEFAULT_ENV_VARS.get(key, default))

def create_temp_directories():
    """Create necessary temporary directories"""
    import tempfile
    from pathlib import Path
    
    temp_dir = Path(get_env_var("VOICE_ASSISTANT_TEMP_DIR", tempfile.gettempdir())) / "voice_assistant"
    audio_dir = temp_dir / "audio"
    screenshots_dir = temp_dir / "screenshots"
    
    for directory in [temp_dir, audio_dir, screenshots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    return {
        "temp_dir": str(temp_dir),
        "audio_dir": str(audio_dir),
        "screenshots_dir": str(screenshots_dir)
    }

