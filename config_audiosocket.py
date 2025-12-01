"""AudioSocket Voicebot Configuration"""

import os
from pathlib import Path
from enum import Enum


# Temp directory for audio files
TEMP_DIR = Path("/home/aiadmin/netovo_voicebot/kokora/temp")


class AudioSocketConfig:
    """AudioSocket TCP server configuration"""
    HOST = "127.0.0.1"  # Listen address
    PORT = 9092         # TCP port for AudioSocket

    # Audio format (matches AudioSocket protocol)
    SAMPLE_RATE = 8000  # Hz (telephony standard)
    SAMPLE_WIDTH = 2    # bytes (16-bit)
    CHANNELS = 1        # mono
    FRAME_SIZE = 320    # bytes (20ms @ 8kHz)
    FRAME_DURATION_MS = 20


class AudioConfig:
    """Audio processing configuration (REUSED from ARI)"""
    # Vosk ASR requires 16kHz
    VOSK_SAMPLE_RATE = 16000
    VOSK_MODEL_PATH = Path("/home/aiadmin/netovo_voicebot/kokora/vosk-model-en-us-0.22")

    # VAD configuration
    VAD_SAMPLE_RATE = 8000  # Matches AudioSocket
    VAD_FRAME_DURATION_MS = 20
    VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive


class TurnTakingConfig:
    """Turn-taking and interruption configuration (REUSED)"""
    INTERRUPTION_ENABLED = True
    INTERRUPTION_ENERGY_THRESHOLD = 300
    INTERRUPTION_CONSECUTIVE_FRAMES = 3  # 60ms of speech to interrupt


class LLMConfig:
    """LLM configuration (REUSED)"""
    MODEL_NAME = "llama3.2"
    BASE_URL = "http://localhost:11434"
    TIMEOUT = 30


class ConversationState(str, Enum):
    """Conversation states (REUSED)"""
    IDLE = "IDLE"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AI_SPEAKING = "AI_SPEAKING"
