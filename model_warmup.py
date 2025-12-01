#!/usr/bin/env python3
"""
Model Warmup Module
Loads all models once at startup for instant responses
"""

import time
import logging
import torch
from vosk import Model
from kokoro import KPipeline
import requests
from config_audiosocket import AudioConfig, LLMConfig

logger = logging.getLogger(__name__)


class SharedModels:
    """Shared model instances across all connections"""
    vosk_model = None
    kokoro_pipeline = None
    kokoro_device = None
    models_loaded = False


def load_models():
    """Load all models once at startup (model warmup)"""
    logger.info("=" * 60)
    logger.info("Loading models for AudioSocket Voicebot...")
    logger.info("=" * 60)

    total_start = time.time()

    try:
        # Load Vosk ASR model
        logger.info(f"Loading Vosk model from {AudioConfig.VOSK_MODEL_PATH}...")
        vosk_start = time.time()
        SharedModels.vosk_model = Model(str(AudioConfig.VOSK_MODEL_PATH))
        vosk_time = time.time() - vosk_start
        logger.info(f"Vosk model loaded in {vosk_time:.1f}s")

        # Load Kokoro TTS pipeline
        logger.info("Loading Kokoro TTS pipeline...")
        kokoro_start = time.time()
        SharedModels.kokoro_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available():
            SharedModels.kokoro_pipeline = KPipeline(lang_code='a', device=SharedModels.kokoro_device)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            SharedModels.kokoro_pipeline = KPipeline(lang_code='a')

        kokoro_time = time.time() - kokoro_start
        logger.info(f"Kokoro TTS loaded in {kokoro_time:.1f}s")

        # Warmup Ollama
        logger.info(f"Warming up Ollama ({LLMConfig.MODEL_NAME})...")
        ollama_start = time.time()
        response = requests.post(
            f"{LLMConfig.BASE_URL}/api/generate",
            json={'model': LLMConfig.MODEL_NAME, 'prompt': 'Hello', 'stream': False},
            timeout=30
        )
        response.raise_for_status()
        ollama_time = time.time() - ollama_start
        logger.info(f"Ollama warmed up in {ollama_time:.1f}s")

        SharedModels.models_loaded = True
        total_time = time.time() - total_start

        logger.info("=" * 60)
        logger.info(f"All models loaded in {total_time:.1f}s")
        logger.info("AudioSocket Voicebot ready for INSTANT responses!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        logger.error("Server cannot start without models")
        raise
