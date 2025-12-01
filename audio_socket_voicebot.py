"""
AudioSocket Voicebot - Complete Integration
Combines AudioSocket server with VAD, ASR, Kokoro TTS, and Ollama LLM.
"""

import asyncio
import logging
import struct
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime

# Local imports
from audio_socket_server import AudioSocketServer, AudioSocketConnection
from vad_processor import VADProcessor
from vosk_asr import VoskASR
from audio_utils import resample_8khz_to_16khz
from config_audiosocket import (
    AudioSocketConfig, AudioConfig, ConversationState,
    TurnTakingConfig, LLMConfig
)

# Kokoro TTS imports
import torch
from kokoro import KPipeline
import soundfile as sf

# Ollama imports
import requests
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KokoroTTS:
    """Kokoro TTS integration for audio generation"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Kokoro TTS using device: {self.device}")

        # Initialize Kokoro pipeline
        if torch.cuda.is_available():
            self.pipeline = KPipeline(lang_code='a', device=self.device)
        else:
            self.pipeline = KPipeline(lang_code='a')

        # Voice mapping from AGI version
        self.voice_mapping = {
            "af_sarah": "af_sarah",
            "af_bella": "af_bella",
            "af_jessica": "af_jessica",
            "af_nova": "af_nova",
            "af_sky": "af_sky",
            "af_heart": "af_heart",
            "af_alloy": "af_alloy"
        }

    def synthesize(self, text: str, voice: str = 'af_sky') -> bytes:
        """
        Synthesize speech from text using Kokoro TTS.
        Uses sox for high-quality 24kHz -> 8kHz resampling (EXACT AGI approach).

        Args:
            text: Text to synthesize
            voice: Voice name (af_sky, af_bella, af_heart, etc.)

        Returns:
            PCM audio data at 8kHz (int16 LE)
        """
        import subprocess
        import uuid
        import time
        import os

        temp_24k = None
        temp_8k = None

        try:
            # Map voice
            kokoro_voice = self.voice_mapping.get(voice, "af_heart")

            # Generate audio using KPipeline (yields chunks at 24kHz)
            generator = self.pipeline(text, voice=kokoro_voice)

            # Collect audio chunks
            audio_chunks = []
            for i, (gs, ps, audio_chunk) in enumerate(generator):
                audio_chunks.append(audio_chunk)

            if not audio_chunks:
                logger.error("No audio generated from Kokoro")
                return b''

            # Concatenate chunks
            full_audio = np.concatenate(audio_chunks)

            # Generate unique temp filenames (MATCH AGI naming)
            unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            temp_24k = f"/tmp/kokoro_temp_{unique_id}.wav"
            temp_8k = f"/tmp/kokoro_tts_{unique_id}.wav"

            # Save at native sample rate (24kHz) - EXACT AGI approach
            sf.write(temp_24k, full_audio, 24000, subtype='PCM_16')

            logger.debug(f"Generated audio: {len(full_audio)} samples at 24000Hz")

            # Convert to 8kHz using sox (EXACT AGI command)
            sox_cmd = [
                'sox', temp_24k,
                '-r', '8000',              # 8kHz for AudioSocket
                '-c', '1',                 # Mono
                '-b', '16',                # 16-bit
                '-e', 'signed-integer',    # PCM
                temp_8k
            ]

            result = subprocess.run(sox_cmd, capture_output=True, text=True, timeout=10)

            # Cleanup temp 24k file immediately
            try:
                os.unlink(temp_24k)
            except:
                pass

            if result.returncode != 0:
                logger.error(f"Sox resampling failed: {result.stderr}")
                return b''

            if not os.path.exists(temp_8k):
                logger.error("Sox output file not created")
                return b''

            # Read WAV file and extract raw PCM data
            audio_8khz, sr = sf.read(temp_8k, dtype='int16')

            # Convert numpy array to bytes
            audio_8khz_bytes = audio_8khz.tobytes()

            file_size = os.path.getsize(temp_8k)
            logger.info(f"Kokoro TTS success: {file_size} bytes WAV, {len(audio_8khz_bytes)} bytes PCM")

            return audio_8khz_bytes

        except Exception as e:
            logger.error(f"Kokoro TTS error: {e}", exc_info=True)
            return b''

        finally:
            # Cleanup temp files
            for temp_file in [temp_24k, temp_8k]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logger.debug(f"Cleanup failed for {temp_file}: {e}")


class OllamaLLM:
    """Ollama LLM integration"""

    def __init__(self):
        self.base_url = LLMConfig.BASE_URL
        self.model = LLMConfig.MODEL_NAME
        self.timeout = LLMConfig.TIMEOUT

        # System prompt
        self.system_prompt = """You are a helpful voice assistant.
Keep responses concise and natural for spoken conversation.
Respond in 1-3 sentences unless more detail is specifically requested."""

        self.conversation_history = []

    def generate_response(self, user_text: str) -> str:
        """
        Generate response from user input.

        Args:
            user_text: User's transcribed speech

        Returns:
            AI response text
        """
        try:
            # Add user message to history
            self.conversation_history.append({
                'role': 'user',
                'content': user_text
            })

            # Build prompt with history
            messages = [{'role': 'system', 'content': self.system_prompt}]
            messages.extend(self.conversation_history[-10:])  # Last 10 messages

            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    'model': self.model,
                    'messages': messages,
                    'stream': False
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            # Extract response
            result = response.json()
            assistant_text = result['message']['content']

            # Add to history
            self.conversation_history.append({
                'role': 'assistant',
                'content': assistant_text
            })

            logger.info(f"LLM response: {assistant_text}")
            return assistant_text

        except Exception as e:
            logger.error(f"Ollama error: {e}", exc_info=True)
            return "I'm sorry, I'm having trouble processing that right now."

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []


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

        self.asr = VoskASR(
            model_path=str(AudioConfig.VOSK_MODEL_PATH),
            sample_rate=AudioConfig.VOSK_SAMPLE_RATE
        )

        self.tts = KokoroTTS()
        self.llm = OllamaLLM()

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
            await self._speak("Hello! How can I help you today?")

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

            # VAD on 8kHz audio
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

            else:
                self.silence_frames += 1

                # End of speech detection (500ms of silence = 25 frames @ 20ms)
                if self.state == ConversationState.USER_SPEAKING and self.silence_frames >= 25:
                    logger.info("User finished speaking (silence detected)")
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

    async def _speak(self, text: str):
        """
        Synthesize and send speech to user.

        Args:
            text: Text to speak
        """
        try:
            self.state = ConversationState.AI_SPEAKING
            logger.info(f"Bot speaking: {text}")

            # Synthesize with Kokoro
            audio_pcm_8khz = await asyncio.to_thread(
                self.tts.synthesize,
                text
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


async def main():
    """Main entry point"""
    logger.info("Starting AudioSocket Voicebot Server...")

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
