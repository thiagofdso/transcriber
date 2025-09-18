"""
Core module for the transcription library.

Contains the main interfaces, utilities, and configuration.
"""

from .interfaces import ITranscriptionProvider, TranscriptionResult
from .manager import TranscriptionManager
from .config import settings
from .utils import get_file_hash, get_audio_duration

__all__ = [
    "ITranscriptionProvider",
    "TranscriptionResult",
    "TranscriptionManager",
    "settings",
    "get_file_hash",
    "get_audio_duration"
]