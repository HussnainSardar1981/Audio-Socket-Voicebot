"""
AudioSocket Voicebot - Complete Integration
Combines AudioSocket server with VAD, ASR, Kokoro TTS, and Ollama LLM.
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional

# Local imports
from audio_socket_server import AudioSocketServer, AudioSocketConnection
from vad_processor import VADProcessor
from vosk_asr import VoskASR
from audio_utils import resample_8khz_to_16khz
from kokoro_tts_audiosocket import KokoroTTSClient
from ollama_audiosocket import OllamaClient
from config_audiosocket import (
    AudioSocketConfig, AudioConfig, ConversationState,
    TurnTakingConfig, LLMConfig
)

# Imports for model loading
import torch
from kokoro import KPipeline
import requests

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global shared models (loaded once at startup, reused across all connections)
class SharedModels:
    """Shared model instances to avoid loading on every call"""
    vosk_model = None
    kokoro_pipeline = None
    kokoro_device = None
    models_loaded = False


class AudioSocketVoicebot:
    """
    Complete voicebot implementation using AudioSocket.

    Integrates:
    - AudioSocket bidirectional audio
    - VAD for speech detection
    - Vosk ASR for transcription
    - Ollama LLM for responses
    - Kokoro TTS for synthesis
    """

    def __init__(self, connection: AudioSocketConnection):
        self.connection = connection
        self.state = ConversationState.IDLE

        # Initialize components
        self.vad = VADProcessor(
            sample_rate=AudioConfig.VAD_SAMPLE_RATE,
            frame_duration_ms=AudioConfig.VAD_FRAME_DURATION_MS,
            aggressiveness=AudioConfig.VAD_AGGRESSIVENESS
        )

        # Use shared pre-loaded Vosk model
        if not SharedModels.models_loaded:
            logger.error("Models not loaded! Call load_models() before creating AudioSocketVoicebot")
            raise RuntimeError("Shared models not loaded")

        self.asr = VoskASR(
            model=SharedModels.vosk_model,
            sample_rate=AudioConfig.VOSK_SAMPLE_RATE
        )

        # Use new modular TTS and LLM clients
        self.tts = KokoroTTSClient(
            shared_pipeline=SharedModels.kokoro_pipeline,
            shared_device=SharedModels.kokoro_device
        )
        self.llm = OllamaClient()

        # Audio buffers
        self.user_speech_buffer = bytearray()  # 8kHz buffer for user speech
        self.silence_frames = 0
        self.speech_frames = 0

        # Interruption detection
        self.interruption_enabled = TurnTakingConfig.INTERRUPTION_ENABLED
        self.consecutive_speech_frames = 0

        logger.info("AudioSocket Voicebot initialized")

    async def start(self):
        """Start voicebot conversation"""
        try:
            # Set audio callback
            self.connection.on_audio_received = self._on_audio_frame

            # Send greeting
            await self._speak("Hello! How can I help you today?", voice_type="greeting")

            # Keep running until connection closes
            while self.connection.active:
                await asyncio.sleep(0.1)

            logger.info("Voicebot conversation ended")

        except Exception as e:
            logger.error(f"Voicebot error: {e}", exc_info=True)

    def _on_audio_frame(self, pcm_data: bytes):
        """
        Handle incoming audio frame from user.

        Args:
            pcm_data: 320 bytes of int16 LE PCM @ 8kHz
        """
        try:
            # Check for interruption during AI speaking
            if self.state == ConversationState.AI_SPEAKING:
                if self.interruption_enabled:
                    energy = self._calculate_energy(pcm_data)
                    if energy > TurnTakingConfig.INTERRUPTION_ENERGY_THRESHOLD:
                        self.consecutive_speech_frames += 1
                        if self.consecutive_speech_frames >= TurnTakingConfig.INTERRUPTION_CONSECUTIVE_FRAMES:
                            logger.info("User interrupted bot")
                            self._handle_interruption()
                    else:
                        self.consecutive_speech_frames = 0
                return  # Don't process audio during bot speaking

            # Pre-filter: reject low-energy frames (channel noise) before VAD
            energy = self._calculate_energy(pcm_data)
            if energy < TurnTakingConfig.VAD_ENERGY_THRESHOLD:
                is_speech = False
                # Log first few rejections for debugging
                if not hasattr(self, '_noise_logged'):
                    self._noise_logged = 0
                if self._noise_logged < 5:
                    logger.debug(f"Frame rejected: energy {energy:.1f} < threshold {TurnTakingConfig.VAD_ENERGY_THRESHOLD}")
                    self._noise_logged += 1
            else:
                # Energy above threshold - run VAD
                is_speech = self.vad.process_frame(pcm_data)

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0

                # Transition to USER_SPEAKING
                if self.state == ConversationState.IDLE:
                    logger.info("User started speaking")
                    self.state = ConversationState.USER_SPEAKING
                    self.user_speech_buffer.clear()

                # Accumulate speech
                if self.state == ConversationState.USER_SPEAKING:
                    self.user_speech_buffer.extend(pcm_data)

                    # Debug: Log buffer size every 50 frames
                    if self.speech_frames % 50 == 0:
                        logger.debug(f"Speech: {self.speech_frames} frames, buffer: {len(self.user_speech_buffer)} bytes")

                    # Max speech duration: 10 seconds (500 frames @ 20ms)
                    if self.speech_frames >= 500:
                        logger.warning(f"Max speech duration reached ({self.speech_frames} frames), forcing transcription")
                        asyncio.create_task(self._process_user_speech())

            else:
                self.silence_frames += 1

                # Debug: Log silence detection
                if self.state == ConversationState.USER_SPEAKING:
                    if self.silence_frames % 10 == 0:
                        logger.debug(f"Silence: {self.silence_frames} frames, buffer: {len(self.user_speech_buffer)} bytes")

                # End of speech detection (500ms of silence = 25 frames @ 20ms)
                if self.state == ConversationState.USER_SPEAKING and self.silence_frames >= 25:
                    logger.info(f"User finished speaking (silence detected after {self.silence_frames} frames)")
                    asyncio.create_task(self._process_user_speech())

        except Exception as e:
            logger.error(f"Error processing audio frame: {e}", exc_info=True)

    def _calculate_energy(self, pcm_data: bytes) -> float:
        """Calculate audio energy for interruption detection"""
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    def _handle_interruption(self):
        """Handle user interruption during bot speech"""
        # Stop current playback (would need enhancement to actually stop audio stream)
        self.state = ConversationState.IDLE
        self.consecutive_speech_frames = 0
        logger.info("Interruption handled - returning to IDLE")

    async def _process_user_speech(self):
        """Process accumulated user speech"""
        try:
            self.state = ConversationState.PROCESSING

            # Get speech buffer
            speech_8khz = bytes(self.user_speech_buffer)
            self.user_speech_buffer.clear()

            if len(speech_8khz) < 1600:  # Less than 100ms
                logger.warning("Speech too short, ignoring")
                self.state = ConversationState.IDLE
                return

            # Resample 8kHz -> 16kHz for Vosk
            speech_16khz = resample_8khz_to_16khz(speech_8khz)

            # Transcribe with Vosk
            logger.info("Transcribing speech...")
            transcript = await asyncio.to_thread(
                self.asr.transcribe_audio,
                speech_16khz,
                emit_final=True
            )

            if not transcript or transcript.strip() == "":
                logger.warning("Empty transcription")
                self.state = ConversationState.IDLE
                return

            logger.info(f"User said: {transcript}")

            # Generate LLM response
            logger.info("Generating response...")
            response_text = await asyncio.to_thread(
                self.llm.generate_response,
                transcript
            )

            # Speak response
            await self._speak(response_text)

            # Return to IDLE
            self.state = ConversationState.IDLE

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            self.state = ConversationState.IDLE

    async def _speak(self, text: str, voice_type: str = "default"):
        """
        Synthesize and send speech to user.

        Args:
            text: Text to speak
            voice_type: Voice type (empathetic, technical, greeting, default)
        """
        try:
            self.state = ConversationState.AI_SPEAKING
            logger.info(f"Bot speaking: {text}")

            # Synthesize with Kokoro (using af_heart voice with appropriate voice type)
            audio_pcm_8khz = await asyncio.to_thread(
                self.tts.synthesize,
                text,
                voice_type=voice_type
            )

            if not audio_pcm_8khz:
                logger.error("TTS synthesis failed")
                self.state = ConversationState.IDLE
                return

            # Send audio in 320-byte frames (20ms @ 8kHz)
            frame_size = AudioSocketConfig.FRAME_SIZE
            for i in range(0, len(audio_pcm_8khz), frame_size):
                frame = audio_pcm_8khz[i:i + frame_size]

                # Pad last frame if needed
                if len(frame) < frame_size:
                    frame += b'\x00' * (frame_size - len(frame))

                # Send frame
                await self.connection.send_audio(frame)

                # Throttle to real-time (20ms per frame)
                await asyncio.sleep(0.020)

            logger.info("Finished speaking")
            self.state = ConversationState.IDLE

        except Exception as e:
            logger.error(f"Error speaking: {e}", exc_info=True)
            self.state = ConversationState.IDLE


def load_models():
    """Load all models once at startup (model warmup)"""
    import time
    from vosk import Model

    logger.info("=" * 60)
    logger.info("🔥 Loading models for AudioSocket Voicebot...")
    logger.info("=" * 60)

    total_start = time.time()

    try:
        # Load Vosk ASR model
        logger.info(f"Loading Vosk model from {AudioConfig.VOSK_MODEL_PATH}...")
        vosk_start = time.time()
        SharedModels.vosk_model = Model(str(AudioConfig.VOSK_MODEL_PATH))
        vosk_time = time.time() - vosk_start
        logger.info(f"✅ Vosk model loaded in {vosk_time:.1f}s")

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
        logger.info(f"✅ Kokoro TTS loaded in {kokoro_time:.1f}s")

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
        logger.info(f"✅ Ollama warmed up in {ollama_time:.1f}s")

        SharedModels.models_loaded = True
        total_time = time.time() - total_start

        logger.info("=" * 60)
        logger.info(f"✅ All models loaded in {total_time:.1f}s")
        logger.info("🚀 AudioSocket Voicebot ready for INSTANT responses!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}", exc_info=True)
        logger.error("Server cannot start without models")
        raise


async def main():
    """Main entry point"""
    logger.info("Starting AudioSocket Voicebot Server...")

    # Load models once at startup (model warmup)
    load_models()

    # Connection handler
    def on_connection(connection: AudioSocketConnection):
        """Handle new AudioSocket connection"""
        logger.info(f"New call from {connection.peer_address}")

        # Create voicebot for this connection
        voicebot = AudioSocketVoicebot(connection)

        # Start voicebot in background
        asyncio.create_task(voicebot.start())

    # Create and start server
    server = AudioSocketServer(
        host=AudioSocketConfig.HOST,
        port=AudioSocketConfig.PORT,
        on_connection=on_connection
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
