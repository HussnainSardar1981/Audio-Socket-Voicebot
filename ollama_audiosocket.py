#!/usr/bin/env python3
"""
Ollama LLM Client for AudioSocket with RAG Integration
Fast LLM integration for voice conversation with knowledge base retrieval
"""

import logging
import requests
from typing import Optional
from config_audiosocket import LLMConfig, RAGConfig

logger = logging.getLogger(__name__)

# Try to import RAG client (optional)
try:
    from chroma_rag_client import ChromaRAGClient
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("ChromaRAGClient not available - RAG features disabled")


class OllamaClient:
    """Ollama LLM client for AudioSocket voicebot with optional RAG enhancement"""

    def __init__(self, customer_id: Optional[str] = None):
        self.base_url = LLMConfig.BASE_URL
        self.model = LLMConfig.MODEL_NAME
        self.timeout = LLMConfig.TIMEOUT
        self.customer_id = customer_id or RAGConfig.CUSTOMER_ID

        # System prompt for voice assistant
        self.system_prompt = """You are a helpful voice assistant. Keep responses concise and natural for spoken conversation.
Respond in 1-3 sentences unless more detail is specifically requested.
Use the provided context from the knowledge base to answer questions accurately."""

        self.conversation_history = []

        # Initialize RAG client if enabled and available
        self.rag_client = None
        self.use_rag = False

        if RAGConfig.ENABLED and RAG_AVAILABLE and self.customer_id:
            try:
                # Construct path to customer directory
                # ChromaDB database is stored directly in customers/{customer_id}/
                from pathlib import Path
                customer_dir = str(Path(RAGConfig.SERVER_ROOT) / "customers" / self.customer_id)
                self.rag_client = ChromaRAGClient(customer_dir, self.customer_id)
                self.use_rag = True
                logger.info(f"RAG enabled for customer: {self.customer_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Continuing without RAG.")
                self.use_rag = False
        elif RAGConfig.ENABLED and not RAG_AVAILABLE:
            logger.warning("RAG enabled but ChromaRAGClient not available")
        elif not RAGConfig.ENABLED:
            logger.info("RAG disabled in configuration")

        logger.info(f"Ollama client initialized: {self.model}")

    def generate_response(self, user_text: str) -> str:
        """
        Generate response from user input, optionally enhanced with RAG context.

        Args:
            user_text: User's transcribed speech

        Returns:
            AI response text
        """
        try:
            # Retrieve context from knowledge base if RAG is enabled
            rag_context = ""
            rag_sources = []

            if self.use_rag and self.rag_client:
                try:
                    rag_result = self.rag_client.query(
                        user_text,
                        n_results=RAGConfig.N_RESULTS,
                        min_relevance=RAGConfig.MIN_RELEVANCE
                    )

                    if rag_result['status'] == 'success' and rag_result['context']:
                        rag_context = rag_result['context'][:RAGConfig.MAX_CONTEXT_LENGTH]
                        rag_sources = rag_result['sources']
                        logger.info(f"RAG retrieved {len(rag_sources)} relevant chunks")
                    else:
                        logger.debug(f"RAG query returned no results: {rag_result['status']}")

                except Exception as e:
                    logger.warning(f"RAG query failed: {e}. Continuing without context.")

            # Build prompt with conversation context and RAG context
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            ])

            # Build final prompt
            if rag_context:
                prompt = f"""{self.system_prompt}

KNOWLEDGE BASE CONTEXT:
{rag_context}

CONVERSATION HISTORY:
{conversation_context}

user: {user_text}
assistant:"""
                logger.debug(f"Using RAG-enhanced prompt with {len(rag_sources)} sources")
            else:
                prompt = f"""{self.system_prompt}

{conversation_context}

user: {user_text}
assistant:"""

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

            # Add source attribution if RAG was used and enabled
            if RAGConfig.INCLUDE_SOURCES and rag_sources:
                sources_text = " [Sources: " + ", ".join([
                    f"{s['doc_name']} p.{s['page_num']}"
                    for s in rag_sources[:2]  # Limit to 2 sources for voice
                ]) + "]"
                logger.info(f"Response based on sources: {sources_text}")

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
