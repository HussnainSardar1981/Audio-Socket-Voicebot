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
    """Audio processing configuration - EASY TO TUNE"""

    # ===== Vosk ASR Configuration =====
    VOSK_SAMPLE_RATE = 16000
    VOSK_MODEL_PATH = Path("/home/aiadmin/netovo_voicebot/kokora/vosk-model-en-us-0.22")

    # ===== VAD Configuration - TUNE THESE =====
    VAD_SAMPLE_RATE = 8000           # Matches AudioSocket (don't change)
    VAD_FRAME_DURATION_MS = 20       # Frame size in milliseconds (don't change)
    VAD_AGGRESSIVENESS = 0           # 0-3 (0=least sensitive, 3=most sensitive)
                                      # Increase if VAD misses speech
                                      # Decrease if VAD detects too much noise

    # ===== Speech Detection Timeouts - TUNE THESE =====
    SILENCE_FRAMES_TO_END_SPEECH = 25  # 500ms silence to end speech (25 frames @ 20ms)
                                        # Increase for slower speakers
                                        # Decrease for faster response

    MAX_SPEECH_FRAMES = 1000            # 20 seconds max speech (1000 frames @ 20ms)
                                         # Increase to allow longer user speech
                                         # Decrease for faster cutoff

    MIN_SPEECH_BYTES = 1600             # 100ms minimum speech required (1600 bytes @ 8kHz)
                                         # Increase to filter out very short sounds


class KokoroConfig:
    """Kokoro TTS Configuration - EASY TO TUNE"""
    VOICE = "af_heart"           # Default voice (af_heart, af_sky, af_bella, af_nova, etc.)
    SPEED = 0.92                 # Speech speed (0.8-1.2, lower=slower, higher=faster)
    SAMPLE_RATE = 24000          # Kokoro native rate (don't change)
    TARGET_SAMPLE_RATE = 8000    # AudioSocket rate (don't change)


class TurnTakingConfig:
    """Turn-taking and interruption configuration - TUNE THESE"""
    # ===== Interruption Detection =====
    INTERRUPTION_ENABLED = True                   # Enable/disable interruption
    INTERRUPTION_ENERGY_THRESHOLD = 300          # RMS energy to detect interruption
                                                  # Increase if false interruptions
                                                  # Decrease if not detecting interruptions

    INTERRUPTION_CONSECUTIVE_FRAMES = 3          # 60ms of speech to confirm interruption (3 frames @ 20ms)
                                                  # Increase to reduce false interruptions
                                                  # Decrease for faster interruption response

    # ===== VAD Pre-filtering =====
    VAD_ENERGY_THRESHOLD = 50                    # RMS energy to filter channel noise before VAD
                                                  # Increase if detecting too much background noise
                                                  # Decrease if missing quiet speech


class LLMConfig:
    """LLM configuration (REUSED)"""
    MODEL_NAME = "llama3.1:8b"
    BASE_URL = "http://localhost:11434"
    TIMEOUT = 30


class RAGConfig:
    """RAG (Retrieval Augmented Generation) configuration"""
    # Enable/disable RAG enhancement
    ENABLED = True

    # Server root path - where customers/ folder is located
    # ChromaDB database is stored DIRECTLY in customers/{customer_id}/ as chroma.sqlite3
    # NOT in a separate chroma_db/ folder
    SERVER_ROOT = "/home/aiadmin/netovo_voicebot/kokora/audiosocket"

    # Customer ID for knowledge base lookup (will be set dynamically per call)
    # Set to None to disable RAG, or set to specific customer ID
    CUSTOMER_ID = None

    # Number of relevant chunks to retrieve
    N_RESULTS = 3

    # Minimum relevance score (0-1) to include in context
    # Higher = more selective, only very relevant chunks
    # Lower = more inclusive, even loosely relevant chunks
    MIN_RELEVANCE = 0.5

    # Include sources in response (for debugging/transparency)
    INCLUDE_SOURCES = True

    # Maximum context length to append to LLM prompt (characters)
    MAX_CONTEXT_LENGTH = 2000


class ZabbixConfig:
    """Zabbix alert integration configuration"""
    # Alert server endpoint
    ALERT_SERVER_URL = "http://localhost:9001"

    # DTMF detection settings
    DTMF_SAMPLE_RATE = 8000              # AudioSocket rate (don't change)
    DTMF_FRAME_DURATION_MS = 20          # Frame duration (don't change)
    DTMF_WAIT_TIMEOUT = 30               # Seconds to wait for DTMF response
    DTMF_ENERGY_THRESHOLD = 100.0        # Minimum energy for tone detection
    DTMF_TONE_THRESHOLD = 0.3            # Relative magnitude threshold
    DTMF_MIN_DURATION_MS = 40            # Minimum tone duration (40ms = 2 frames)

    # Alert call detection
    # Alert calls are detected via dialplan passing call_id in UUID field
    ALERT_CALL_ID_PREFIX = "zabbix_alert_"


class ConversationState(str, Enum):
    """Conversation states (REUSED)"""
    IDLE = "IDLE"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING = "PROCESSING"
    AI_SPEAKING = "AI_SPEAKING"
