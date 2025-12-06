#!/usr/bin/env python3
"""
Ollama LLM Client for AudioSocket with RAG Support
Integrates retrieval-augmented generation for customer-specific knowledge
"""

import logging
import requests
import json
import os
from datetime import datetime
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
            self.system_prompt = f"""You are Alexis, a professional customer support assistant for {customer_id}.

CONVERSATION STYLE (CRITICAL):
- Keep responses SHORT (1-2 sentences, max 40 words)
- Ask ONE question at a time to keep conversation flowing
- Speak naturally like a helpful human agent on the phone
- End your turn with a question when appropriate

ANSWERING FROM KNOWLEDGE BASE:
- Use ONLY the information from the documents provided below
- Speak naturally as if you have this knowledge yourself
- NEVER mention "documents", "knowledge base", "page numbers", or file names
- If documents don't have the answer, say "I don't have that specific information. Could you clarify what you're looking for?"

GOOD EXAMPLES (Customer Support):
User: "How do I reset my password?"
You: "You can reset it by clicking 'Forgot Password' on the login page. Would you like me to walk you through the steps?"

User: "What are your business hours?"
You: "We're open Monday to Friday, 9 AM to 5 PM. Is there a specific day you'd like to visit?"

User: "My system keeps crashing"
You: "I can help with that. What error message do you see when it crashes?"

BAD EXAMPLES (Avoid):
- "According to the documentation on page 5..." (too technical)
- "Let me tell you all about our products and services and pricing..." (too long)
- "What's your issue and when did it start and have you tried restarting?" (multiple questions)

WHEN INFORMATION NOT FOUND:
- Don't make up answers
- Politely say you don't have that information
- Ask a clarifying question or offer to help with something else"""
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
            # Note: {customer_id} is a LITERAL folder name, not a placeholder
            chroma_path = RAGConfig.CHROMA_DB_PATH

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

    def _log_unanswered_question(self, question: str):
        """
        Log questions that RAG couldn't answer for knowledge base improvement

        Args:
            question: User's question that couldn't be answered from RAG
        """
        try:
            # Create directory for unanswered questions
            unanswered_dir = f"./unanswered_questions/{self.customer_id}"
            os.makedirs(unanswered_dir, exist_ok=True)

            # Log file path (one file per day)
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = f"{unanswered_dir}/{today}.jsonl"

            # Create log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "customer_id": self.customer_id,
                "question": question,
                "date": today
            }

            # Append to JSONL file (one JSON object per line)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

            logger.info(f"Logged unanswered question for {self.customer_id}: {question}")

        except Exception as e:
            logger.error(f"Failed to log unanswered question: {e}", exc_info=True)

    def generate_response_streaming(self, user_text: str):
        """
        Generate response with streaming (yields sentences as they complete)

        Args:
            user_text: User's transcribed speech

        Yields:
            Complete sentences as they are generated

        This allows TTS to start speaking while LLM is still generating,
        dramatically reducing perceived latency.
        """
        try:
            # Step 1: Retrieve context from RAG (if enabled)
            rag_context = self._retrieve_context(user_text)

            # Step 2: Build prompt
            if rag_context:
                prompt = self._build_rag_prompt(user_text, rag_context)
                logger.info("Using RAG-augmented prompt (streaming)")
            else:
                prompt = self._build_standard_prompt(user_text)
                logger.info("Using standard prompt (streaming, no RAG context)")

                # Log unanswered question if RAG was enabled but no context found
                if self.rag_enabled:
                    self._log_unanswered_question(user_text)

            # Step 3: Call Ollama LLM with STREAMING
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': True,  # Enable streaming
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 50,
                        'stop': ['\n\n', 'User:', 'Human:', 'Assistant:']
                    }
                },
                timeout=self.timeout,
                stream=True  # Enable streaming on requests
            )
            response.raise_for_status()

            # Step 4: Stream tokens and yield complete sentences
            current_sentence = ""
            full_response = ""

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get('response', '')
                        current_sentence += token
                        full_response += token

                        # Check if sentence is complete (ends with . ? ! or newline)
                        if token in ['.', '?', '!'] or '\n' in token:
                            sentence = current_sentence.strip()
                            if sentence:
                                logger.info(f"Yielding sentence: {sentence[:50]}...")
                                yield sentence
                                current_sentence = ""

                        # Check if generation is done
                        if chunk.get('done', False):
                            break

                    except json.JSONDecodeError:
                        continue

            # Yield any remaining text that didn't end with punctuation
            if current_sentence.strip():
                logger.info(f"Yielding final text: {current_sentence[:50]}...")
                yield current_sentence.strip()

            # Step 5: Update conversation history with full response
            self.conversation_history.append({'role': 'user', 'content': user_text})
            self.conversation_history.append({'role': 'assistant', 'content': full_response.strip()})

            # Keep only last 10 messages
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            logger.info(f"Streaming complete: {full_response[:50]}...")

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}", exc_info=True)
            yield "I'm sorry, I'm having trouble processing that right now."

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

                # Log unanswered question if RAG was enabled but no context found
                if self.rag_enabled:
                    self._log_unanswered_question(user_text)

            # Step 3: Call Ollama LLM with token limiting for faster, shorter responses
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 50,  # Limit to ~50 tokens (30-40 words) for natural conversation
                        'stop': ['\n\n', 'User:', 'Human:', 'Assistant:']  # Stop at natural breaks
                    }
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
