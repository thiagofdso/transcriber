"""
Transcription Library - A pluggable Python library for audio and video transcription.

This library provides a modular architecture for transcribing audio and video files
using various AI models including Distil-Whisper, Faster-Whisper, and Gemini.
"""

from .core.manager import TranscriptionManager
from .core.interfaces import ITranscriptionProvider, TranscriptionResult
from .core.config import settings

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "TranscriptionManager",
    "ITranscriptionProvider",
    "TranscriptionResult",
    "settings"
]