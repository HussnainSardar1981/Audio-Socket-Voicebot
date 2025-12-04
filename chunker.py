"""
Token-Aware Text Chunking for RAG Pipeline
Splits cleaned text into 400-600 token chunks with 20-30% overlap
Generates SHA256 hashes for deduplication
"""

import json
import hashlib
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

# Try to import tokenizers for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[WARN] tiktoken not installed. Using character-based estimation.")
    print("       For accurate token counting: pip install tiktoken")

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CHUNKER - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenCounter:
    """Count tokens using tiktoken or character estimation"""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize token counter"""
        self.use_tiktoken = TIKTOKEN_AVAILABLE
        if self.use_tiktoken:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base encoding
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Character-based estimation: ~4 chars per token
            self.chars_per_token = 4

    def count(self, text: str) -> int:
        """Count tokens in text"""
        if self.use_tiktoken:
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: ~4 characters = 1 token
            return len(text) // self.chars_per_token


class DocumentChunker:
    """Split documents into token-aware chunks with overlap"""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize chunker

        Args:
            chunk_size: Target tokens per chunk (400-600 recommended)
            chunk_overlap: Overlap in tokens (20-30% of chunk_size)
            model: Model for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter(model)

        # Validate parameters
        overlap_percent = (chunk_overlap / chunk_size) * 100
        if overlap_percent < 15 or overlap_percent > 35:
            print(f"[WARN] Overlap is {overlap_percent:.1f}% (recommended 20-30%)")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) < 50:
            return [text] if text.strip() else []

        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.token_counter.count(para)

            # If paragraph alone exceeds chunk size, split further
            if para_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph by sentences
                sentences = para.split('. ')
                for sentence in sentences:
                    if not sentence:
                        continue

                    sent_tokens = self.token_counter.count(sentence + '. ')

                    if sent_tokens > self.chunk_size:
                        # Word-level split as last resort
                        words = sentence.split(' ')
                        sub_chunk = []
                        sub_tokens = 0

                        for word in words:
                            word_tokens = self.token_counter.count(word + ' ')
                            if sub_tokens + word_tokens > self.chunk_size and sub_chunk:
                                chunks.append(' '.join(sub_chunk))
                                sub_chunk = [word]
                                sub_tokens = word_tokens
                            else:
                                sub_chunk.append(word)
                                sub_tokens += word_tokens

                        if sub_chunk:
                            chunks.append(' '.join(sub_chunk))

                    elif current_tokens + sent_tokens <= self.chunk_size:
                        current_chunk.append(sentence + '. ')
                        current_tokens += sent_tokens
                    else:
                        if current_chunk:
                            chunks.append(''.join(current_chunk))
                        current_chunk = [sentence + '. ']
                        current_tokens = sent_tokens

            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_tokens += para_tokens

            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens

        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        # Apply overlap: each chunk includes last N tokens of previous
        if len(chunks) > 1 and self.chunk_overlap > 0:
            chunks_with_overlap = []

            for i, chunk in enumerate(chunks):
                if i == 0:
                    chunks_with_overlap.append(chunk)
                else:
                    # Get overlap from previous chunk
                    prev_chunk = chunks[i - 1]
                    prev_tokens = self.token_counter.count(prev_chunk)

                    # Calculate how much to take from previous
                    overlap_chars = int((self.chunk_overlap / prev_tokens) * len(prev_chunk)) if prev_tokens > 0 else 0
                    overlap_text = prev_chunk[-overlap_chars:] if overlap_chars > 0 else ""

                    # Combine with current
                    combined = overlap_text + chunk
                    chunks_with_overlap.append(combined)

            return chunks_with_overlap

        return chunks

    def chunk_document(
        self,
        customer_id: str,
        doc_name: str,
        pages: List[Dict],
        existing_hashes: Optional[set] = None
    ) -> List[Dict]:
        """
        Chunk a complete document (all pages)

        Args:
            customer_id: Customer identifier
            doc_name: Document name
            pages: List of page dicts with 'page_num' and 'pdf_text_clean' keys
            existing_hashes: Set of existing chunk hashes (for deduplication)

        Returns:
            List of chunk dicts with metadata
        """
        if existing_hashes is None:
            existing_hashes = set()

        chunks = []
        chunk_id = 0

        for page in pages:
            page_num = page.get('page_num', 0)
            text_clean = page.get('pdf_text', '')
            images = page.get('images', [])

            if not text_clean.strip():
                logger.debug(f"Skipping empty page {page_num}")
                continue

            # Split page into chunks
            page_chunks = self.chunk_text(text_clean)

            for chunk_text in page_chunks:
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                # Generate chunk hash for deduplication
                chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

                # Skip if duplicate
                if chunk_hash in existing_hashes:
                    logger.debug(f"Skipping duplicate chunk: {chunk_hash[:8]}...")
                    continue

                chunk_id += 1

                # Count tokens
                token_count = self.token_counter.count(chunk_text)

                # Extract image OCR from page if available
                image_texts = [img.get('ocr_text', '') for img in images if img.get('ocr_text')]
                image_text_combined = ' '.join(image_texts)

                # Create chunk metadata
                chunk_metadata = {
                    'chunk_id': chunk_id,
                    'chunk_hash': chunk_hash,
                    'customer_id': customer_id,
                    'doc_name': doc_name,
                    'page_num': page_num,
                    'text_length': len(chunk_text),
                    'token_count': token_count,
                    'created_at': datetime.now().isoformat(),
                    'has_images': len(images) > 0,
                    'image_text': image_text_combined[:500] if image_text_combined else None
                }

                chunks.append({
                    'metadata': chunk_metadata,
                    'text': chunk_text
                })

                existing_hashes.add(chunk_hash)

        logger.info(f"Chunked {doc_name}: {chunk_id} chunks from {len(pages)} pages")

        return chunks


class RAGChunker:
    """Coordinate chunking for all customer documents"""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
        server_root: Optional[Path] = None
    ):
        """Initialize RAG chunker"""
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.server_root = server_root or Path('/home/aiadmin/netovo_voicebot/audiosockets')

    def chunk_customer_documents(self, customer_id: str) -> Dict:
        """
        Chunk all documents for a customer

        Returns:
            {
                'customer_id': '...',
                'documents': {
                    'doc_name': [chunks...]
                },
                'total_chunks': N,
                'created_at': '...'
            }
        """
        print(f"\n[CHUNK] Processing customer: {customer_id}")
        print("=" * 70)

        # Paths
        extracted_dir = self.server_root / "customers" / customer_id / "extracted"
        kb_metadata_path = self.server_root / "customers" / customer_id / "kb_metadata.json"

        if not extracted_dir.exists():
            logger.error(f"Extracted directory not found: {extracted_dir}")
            return {'status': 'error', 'message': 'No extracted directory'}

        # Load existing chunk hashes for deduplication
        existing_hashes = set()
        if kb_metadata_path.exists():
            with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                kb_metadata = json.load(f)
                for file_info in kb_metadata.get('files', {}).values():
                    existing_hashes.update(file_info.get('chunk_hashes', []))

        # Process each document
        all_chunks = []
        documents_processed = 0

        for doc_dir in sorted(extracted_dir.iterdir()):
            if not doc_dir.is_dir():
                continue

            metadata_path = doc_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata.json in {doc_dir.name}")
                continue

            # Load cleaned metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            doc_name = metadata.get('doc_name')
            pages = metadata.get('pages', [])

            print(f"  Chunking {doc_name}...")

            # Chunk document
            doc_chunks = self.chunker.chunk_document(
                customer_id=customer_id,
                doc_name=doc_name,
                pages=pages,
                existing_hashes=existing_hashes
            )

            print(f"    [OK] Created {len(doc_chunks)} chunks")

            all_chunks.extend(doc_chunks)
            documents_processed += 1

        # Update kb_metadata.json with chunk hashes
        if kb_metadata_path.exists():
            with open(kb_metadata_path, 'r', encoding='utf-8') as f:
                kb_metadata = json.load(f)
        else:
            kb_metadata = {'files': {}}

        for chunk in all_chunks:
            doc_name = chunk['metadata']['doc_name']
            chunk_hash = chunk['metadata']['chunk_hash']

            if doc_name not in kb_metadata['files']:
                kb_metadata['files'][doc_name] = {'chunk_hashes': []}

            if chunk_hash not in kb_metadata['files'][doc_name]['chunk_hashes']:
                kb_metadata['files'][doc_name]['chunk_hashes'].append(chunk_hash)

        with open(kb_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(kb_metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print(f"[SUMMARY] {customer_id}")
        print(f"  Documents processed: {documents_processed}")
        print(f"  Total chunks created: {len(all_chunks)}")
        print(f"  Avg chunk size: {sum(c['metadata']['token_count'] for c in all_chunks) // max(len(all_chunks), 1)} tokens")

        return {
            'customer_id': customer_id,
            'documents_processed': documents_processed,
            'total_chunks': len(all_chunks),
            'chunks': all_chunks,
            'created_at': datetime.now().isoformat()
        }

    def chunk_all_customers(self, customer_ids: Optional[List[str]] = None) -> Dict:
        """Chunk documents for all customers"""
        print("\n" + "=" * 70)
        print("RAG DOCUMENT CHUNKING")
        print("=" * 70)

        results = {}

        # Get customer list
        if customer_ids is None:
            customers_dir = self.server_root / "customers"
            customer_ids = [d.name for d in customers_dir.iterdir() if d.is_dir()]

        for customer_id in customer_ids:
            result = self.chunk_customer_documents(customer_id)
            results[customer_id] = result

        # Final summary
        print("\n" + "=" * 70)
        print("CHUNKING COMPLETE")
        print("=" * 70)

        total_chunks = sum(r.get('total_chunks', 0) for r in results.values())
        print(f"Total customers: {len(results)}")
        print(f"Total chunks: {total_chunks}")

        return results


def main():
    if DOTENV_AVAILABLE:
        load_dotenv()

    server_root = os.getenv('SERVER_ROOT', '/home/aiadmin/netovo_voicebot/audiosockets')

    chunker = RAGChunker(
        chunk_size=600,
        chunk_overlap=150,
        server_root=Path(server_root)
    )

    # Chunk all customers
    results = chunker.chunk_all_customers()

    # Summary
    total_chunks = sum(r.get('total_chunks', 0) for r in results.values())
    return 0 if total_chunks > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
