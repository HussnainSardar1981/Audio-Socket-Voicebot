#!/usr/bin/env python3
"""
ChromaDB RAG Client for Voice Assistant
Queries ChromaDB vector database for relevant context to enhance LLM responses.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("[WARN] chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] sentence-transformers not installed. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class ChromaRAGClient:
    """Query ChromaDB for RAG-enhanced responses"""

    def __init__(self, customer_dir: str, customer_id: str):
        """
        Initialize ChromaDB RAG client

        Args:
            customer_dir: Path to customer directory (e.g., /home/aiadmin/netovo_voicebot/kokora/audiosocket/customers/skiface)
            customer_id: Customer identifier (e.g., 'skiface', 'stuart_dean')

        Note: ChromaDB database is stored DIRECTLY in the customer folder as chroma.sqlite3
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        self.customer_dir = Path(customer_dir)
        self.customer_id = customer_id
        self.collection_name = customer_id

        # Initialize ChromaDB client pointing to customer directory
        # The database is stored as chroma.sqlite3 directly in the customer folder
        try:
            self.client = chromadb.PersistentClient(path=str(self.customer_dir))
            logger.info(f"ChromaDB client initialized at {self.customer_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"Cannot initialize ChromaDB: {e}")

        # Initialize embedder (must match the one used during indexing)
        try:
            self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
            logger.info("Embedding model loaded: BAAI/bge-base-en-v1.5")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Cannot load embedding model: {e}")

    def query(self, query_text: str, n_results: int = 3, min_relevance: float = 0.5) -> Dict:
        """
        Query the knowledge base for relevant context

        Args:
            query_text: User's question or statement
            n_results: Number of results to return
            min_relevance: Minimum relevance score (0-1) to include results

        Returns:
            Dict with 'status', 'context', 'sources', and 'query' keys
        """
        try:
            # Get collection (will fail if collection doesn't exist)
            try:
                collection = self.client.get_collection(name=self.collection_name)
            except Exception as e:
                logger.warning(f"Collection '{self.collection_name}' not found: {e}")
                return {
                    'status': 'no_collection',
                    'context': '',
                    'sources': [],
                    'query': query_text,
                    'message': f"No knowledge base found for {self.customer_id}"
                }

            # Generate query embedding
            query_embedding = self.embedder.encode(
                query_text,
                convert_to_tensor=False,
                normalize_embeddings=True
            ).tolist()

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Process results
            context_chunks = []
            sources = []

            if results and results['ids'] and len(results['ids']) > 0:
                for i, chunk_id in enumerate(results['ids'][0]):
                    # Calculate relevance (convert distance to relevance)
                    distance = results['distances'][0][i]
                    relevance = max(0, 1 - distance)

                    # Skip if below minimum relevance threshold
                    if relevance < min_relevance:
                        logger.debug(f"Skipping chunk {chunk_id} (relevance {relevance:.2f} < {min_relevance})")
                        continue

                    # Extract text and metadata
                    text = results['documents'][0][i] if results['documents'] else ''
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                    if text:
                        context_chunks.append(text)
                        sources.append({
                            'chunk_id': chunk_id,
                            'doc_name': metadata.get('doc_name', 'unknown'),
                            'page_num': metadata.get('page_num', 'unknown'),
                            'relevance': round(relevance * 100, 1)
                        })

            # Format context
            if context_chunks:
                context = "\n\n".join(context_chunks)
                logger.info(f"Found {len(context_chunks)} relevant chunks for query: {query_text[:50]}...")
            else:
                context = ''
                logger.info(f"No relevant chunks found for query: {query_text[:50]}...")

            return {
                'status': 'success',
                'context': context,
                'sources': sources,
                'query': query_text,
                'chunks_found': len(context_chunks)
            }

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}", exc_info=True)
            return {
                'status': 'error',
                'context': '',
                'sources': [],
                'query': query_text,
                'error': str(e)
            }

    def check_collection_exists(self) -> bool:
        """Check if collection exists for this customer"""
        try:
            self.client.get_collection(name=self.collection_name)
            return True
        except:
            return False

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            return {
                'status': 'ok',
                'name': collection.name,
                'total_chunks': collection.count()
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
