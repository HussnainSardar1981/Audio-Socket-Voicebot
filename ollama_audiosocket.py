#!/usr/bin/env python3
"""
Ollama LLM Client for AudioSocket
Fast LLM integration for voice conversation
"""

import logging
import requests
from config_audiosocket import LLMConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama LLM client for AudioSocket voicebot"""

    def __init__(self):
        self.base_url = LLMConfig.BASE_URL
        self.model = LLMConfig.MODEL_NAME
        self.timeout = LLMConfig.TIMEOUT

        # System prompt for voice assistant
        self.system_prompt = """You are a helpful voice assistant.
Keep responses concise and natural for spoken conversation.
Respond in 1-3 sentences unless more detail is specifically requested."""

        self.conversation_history = []

        logger.info(f"Ollama client initialized: {self.model}")

    def generate_response(self, user_text: str) -> str:
        """
        Generate response from user input.

        Args:
            user_text: User's transcribed speech

        Returns:
            AI response text
        """
        try:
            # Build prompt with conversation context
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])

            prompt = f"{self.system_prompt}\n\n{context}\nuser: {user_text}\nassistant:"

            # Call Ollama /api/generate endpoint
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            # Extract response
            result = response.json()
            assistant_text = result['response'].strip()

            # Add to history
            self.conversation_history.append({'role': 'user', 'content': user_text})
            self.conversation_history.append({'role': 'assistant', 'content': assistant_text})

            # Keep only last 10 messages for memory efficiency
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            logger.info(f"LLM response: {assistant_text[:50]}...")
            return assistant_text

        except Exception as e:
            logger.error(f"Ollama error: {e}", exc_info=True)
            return "I'm sorry, I'm having trouble processing that right now."

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.debug("Conversation history reset")
