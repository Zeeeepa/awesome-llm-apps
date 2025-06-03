"""
Voice Computer Assistant Utilities Package

This package contains utility functions and classes for audio processing,
safety controls, and other supporting functionality.
"""

from .audio_utils import AudioRecorder, AudioPlayer, AudioProcessor
from .safety_controls import SafetyManager, PermissionManager

__all__ = [
    'AudioRecorder',
    'AudioPlayer', 
    'AudioProcessor',
    'SafetyManager',
    'PermissionManager'
]

