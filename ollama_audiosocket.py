#!/usr/bin/env python3
"""
Ollama LLM Client for AudioSocket with RAG Support
Integrates retrieval-augmented generation for customer-specific knowledge
"""

import logging
import requests
from typing import Optional
from config_audiosocket import LLMConfig, RAGConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Ollama LLM client with optional RAG support for voicebot"""

    def __init__(self, customer_id: Optional[str] = None):
        """
        Initialize Ollama client with optional RAG

        Args:
            customer_id: Customer ID for RAG retrieval (e.g., "stuart_dean", "skisafe")
                        If None, RAG is disabled for this client (LLM-only mode)
        """
        self.base_url = LLMConfig.BASE_URL
        self.model = LLMConfig.MODEL_NAME
        self.timeout = LLMConfig.TIMEOUT

        # RAG setup
        self.customer_id = customer_id
        self.rag_enabled = RAGConfig.ENABLED and customer_id is not None
        self.rag_client = None

        # Initialize RAG if enabled
        if self.rag_enabled:
            try:
                self.rag_client = self._init_rag_client()
                logger.info(f"RAG enabled for customer: {customer_id}")
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {e}", exc_info=True)

                if RAGConfig.FALLBACK_ON_ERROR:
                    logger.warning("RAG disabled, falling back to LLM-only mode")
                    self.rag_enabled = False
                else:
                    raise
        else:
            if customer_id:
                logger.info(f"RAG globally disabled (customer: {customer_id})")
            else:
                logger.info("RAG disabled (no customer_id provided)")

        # System prompts
        if self.rag_enabled:
            self.system_prompt = f"""You are Alexis, a helpful voice assistant for {customer_id}.

IMPORTANT INSTRUCTIONS:
1. Answer questions using information from the customer's documents provided below
2. Speak naturally as if you have this knowledge yourself
3. Do NOT mention document names, page numbers, or file paths when speaking
4. If the documents don't contain the answer, say "I don't have that information in your knowledge base"
5. Keep responses concise and natural for phone conversation (1-3 sentences)
6. Do NOT add information from your training data if it conflicts with the documents"""
        else:
            self.system_prompt = """You are Alexis, a helpful voice assistant.
Keep responses concise and natural for spoken conversation.
Respond in 1-3 sentences unless more detail is specifically requested."""

        self.conversation_history = []

        logger.info(f"Ollama client initialized: {self.model} (RAG: {self.rag_enabled})")

    def _init_rag_client(self):
        """
        Lazy initialization of RAG client

        Returns:
            ChromaRAGClient instance

        Raises:
            ImportError: If required RAG dependencies not installed
            RuntimeError: If ChromaDB initialization fails
        """
        try:
            from chroma_rag_client import get_rag_client

            # Get ChromaDB path from config
            chroma_path = RAGConfig.get_chroma_db_path()

            # Get or create singleton RAG client
            rag_client = get_rag_client(
                db_path=chroma_path,
                embedding_model=RAGConfig.EMBEDDING_MODEL
            )

            # Verify customer collection exists
            health = rag_client.health_check(self.customer_id)

            if health['status'] != 'ok':
                logger.warning(f"RAG health check failed: {health.get('message', 'unknown error')}")
                if not RAGConfig.FALLBACK_ON_ERROR:
                    raise RuntimeError(f"RAG health check failed for {self.customer_id}")

            logger.info(f"RAG client initialized: {chroma_path} ({health.get('total_chunks', 0)} chunks)")

            return rag_client

        except ImportError as e:
            logger.error(f"RAG dependencies not installed: {e}")
            raise ImportError("Install RAG dependencies: pip install chromadb sentence-transformers")

        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            raise

    def _retrieve_context(self, user_text: str) -> Optional[str]:
        """
        Retrieve relevant context from RAG

        Args:
            user_text: User's question

        Returns:
            Formatted context string or None if RAG disabled/failed
        """
        if not self.rag_enabled or not self.rag_client:
            return None

        try:
            # Retrieve top-K relevant chunks
            results = self.rag_client.retrieve(
                customer_id=self.customer_id,
                query=user_text,
                top_k=RAGConfig.TOP_K,
                min_similarity=RAGConfig.MIN_SIMILARITY_SCORE
            )

            if not results:
                logger.debug("No relevant documents found for query")
                return None

            # Format context for LLM prompt
            context = self.rag_client.format_context(
                results,
                max_length=RAGConfig.MAX_CONTEXT_LENGTH,
                include_metadata=RAGConfig.INCLUDE_SOURCE_METADATA
            )

            if RAGConfig.LOG_RETRIEVAL:
                logger.info(f"Retrieved {len(results)} chunks, context length: {len(context)} chars")
                logger.debug(f"Top similarity: {results[0]['similarity']:.3f}, "
                           f"doc: {results[0]['metadata'].get('doc_name', 'unknown')}")

            return context

        except Exception as e:
            logger.error(f"RAG retrieval error: {e}", exc_info=True)

            if RAGConfig.FALLBACK_ON_ERROR:
                logger.warning("Falling back to LLM without RAG due to error")
                return None
            else:
                raise

    def generate_response(self, user_text: str) -> str:
        """
        Generate response from user input with optional RAG

        Args:
            user_text: User's transcribed speech

        Returns:
            AI response text
        """
        try:
            # Step 1: Retrieve context from RAG (if enabled)
            rag_context = self._retrieve_context(user_text)

            # Step 2: Build prompt
            if rag_context:
                # RAG-augmented prompt
                prompt = self._build_rag_prompt(user_text, rag_context)
                logger.info("Using RAG-augmented prompt")
            else:
                # Standard prompt without RAG
                prompt = self._build_standard_prompt(user_text)
                logger.info("Using standard prompt (no RAG context)")

            # Step 3: Call Ollama LLM
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

            # Step 4: Extract response
            result = response.json()
            assistant_text = result['response'].strip()

            # Step 5: Update conversation history
            self.conversation_history.append({'role': 'user', 'content': user_text})
            self.conversation_history.append({'role': 'assistant', 'content': assistant_text})

            # Keep only last 10 messages for memory efficiency
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            logger.info(f"LLM response generated: {assistant_text[:50]}...")
            return assistant_text

        except Exception as e:
            logger.error(f"Ollama error: {e}", exc_info=True)
            return "I'm sorry, I'm having trouble processing that right now."

    def _build_rag_prompt(self, user_text: str, rag_context: str) -> str:
        """
        Build RAG-augmented prompt with retrieved context

        Args:
            user_text: User's question
            rag_context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        # Get recent conversation context (reduced to save tokens)
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-3:]  # Last 3 messages (reduced from 5)
        ])

        # Build RAG-augmented prompt
        # Put context BEFORE question (recency bias helps LLM focus on it)
        prompt = f"""{self.system_prompt}

RELEVANT INFORMATION FROM KNOWLEDGE BASE:
{rag_context}

RECENT CONVERSATION:
{conversation_context}

USER QUESTION: {user_text}

ASSISTANT RESPONSE (speak naturally, answer from knowledge base):"""

        return prompt

    def _build_standard_prompt(self, user_text: str) -> str:
        """
        Build standard prompt without RAG context

        Args:
            user_text: User's question

        Returns:
            Formatted prompt string
        """
        # Get more conversation context when not using RAG
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-5:]  # Last 5 messages
        ])

        prompt = f"{self.system_prompt}\n\n{conversation_context}\nuser: {user_text}\nassistant:"

        return prompt

    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.debug("Conversation history reset")
