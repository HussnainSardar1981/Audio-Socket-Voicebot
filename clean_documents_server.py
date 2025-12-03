"""
Production Text Cleaning Script for Server Deployment
Two-stage cleaning pipeline:
  Stage 1: Rigorous programmatic cleanup (boilerplate/whitespace removal)
  Stage 2: LLM semantic fixing (OCR errors, hyphenation)

IMPORTANT NOTE ON IMAGE OCR:
The OCR text from screenshots (UI interfaces) is kept SEPARATE from PDF text.
They are NOT synthesized/merged because:
  1. Screenshots show UI elements, form fields, button labels
  2. PDF text shows procedural instructions and documentation
  3. Keeping them separate allows context-aware RAG retrieval
  4. Users can search for both "what to do" AND "what it looks like"
  5. Synthesizing would lose the distinction between action and visual context

Usage:
  # Clean single document
  python clean_documents_server.py --file /path/to/extraction.json --customer CUSTOMER_ID

  # Clean all documents for one customer
  python clean_documents_server.py --customer CUSTOMER_ID

  # Clean all documents for all customers
  python clean_documents_server.py --all

  # Clean specific customers
  python clean_documents_server.py --customers CUST1 CUST2 CUST3
"""

import json
import sys
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests
import time
import argparse

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[WARN] python-dotenv not installed")

logger = None


def rigorous_pre_cleanup(text: str) -> str:
    """
    Stage 1: Rigorous programmatic cleanup before LLM processing
    Removes boilerplate, excessive whitespace, and structural noise
    This ensures the LLM focuses on semantic fixes, not garbage removal
    """
    if not text or not text.strip():
        return ""

    # Common boilerplate patterns found in PDFs (manually identified - NOT LLM dependent)
    # IMPORTANT: These patterns are carefully crafted to NOT remove solution content
    # especially numbered steps (1., 2., 3., etc.) that are part of instructions
    boilerplate_patterns = [
        # Madison IT company header/footer (completely ignore this)
        r'MADISON\s+TECHNOLO[GC].*?Managed Hosting.*?Services',
        # MTI Support Help Desk contact info (completely ignore)
        r'MTI\s+Support\s+Help\s+Desk\s+T:\s*\+1\s*\(\s*212\s*\)\s*400-7550.*?www\.madisonti\.com',
        # Document prep line (completely ignore)
        r'Prepared\s+By\s+Madison\s+Technology\s+for\s+DeLorenzo',
        # Madison Technology How-To header (completely ignore)
        r'Madison\s+Technology\s*\n\s*DeLorenzo\s+How\s+To\s+Use.*?Rev\s+1a',
        # Page numbers (remove)
        r'Page\s+\d+\s+of\s+\d+',
        # Confidential notice (remove)
        r'Confidential',
        # Copyright (remove)
        r'Copyright.*?\d{4}',
        # Footer with copyright (remove)
        r'Footer.*?©.*?\d{4}',
    ]

    # Remove boilerplate patterns
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize excessive whitespace and indentation
    # Replace 2+ spaces/tabs with single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Clean up multiple consecutive newlines (keep max 2 for paragraphs)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


class TextCleaner:
    """Clean extracted PDF text using LLM for semantic fixing"""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "nous-hermes2"):
        global logger

        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 300

        # Initialize logger if not already done
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - CLEANING - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.ollama_url}")
                available_models = [m.get('name', '').split(':')[0] for m in response.json().get('models', [])]
                logger.info(f"Available models: {available_models}")
            else:
                logger.warning(f"Ollama connection status: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")

    def clean_text_with_llm(self, text: str, max_retries: int = 3) -> str:
        """
        Use LLM to clean extracted text
        Two-stage process:
        1. Rigorous programmatic cleanup (boilerplate, whitespace)
        2. LLM semantic fixing (OCR errors, hyphenation)
        """
        if not text.strip():
            return ""

        # Stage 1: Rigorous pre-cleanup (removes structural noise)
        text = rigorous_pre_cleanup(text)

        if not text.strip():
            return ""

        # Stage 2: LLM-based semantic cleaning
        prompt = (
            "You are a text reconstruction specialist. Fix ONLY actual errors, make ZERO other changes.\n\n"
            "STRICT RULES (follow exactly):\n"
            "1. Fix broken hyphenated words ONLY: 'informa-\\ntion' becomes 'information'\n"
            "2. Fix ONLY clear OCR character substitutions: '1l' to 'll', 'rn' to 'm', 'O' to '0' if numbers expected\n"
            "3. Fix ONLY broken sentences from line breaks (rejoin split words)\n"
            "4. PRESERVE ALL OTHER TEXT EXACTLY AS-IS\n"
            "5. NEVER remove any words or content\n"
            "6. NEVER add explanations or comments\n"
            "7. Output ONLY the corrected text, nothing else\n\n"
            f"TEXT:\n{text}\n\n"
            "CORRECTED TEXT (with ONLY the minimal fixes above):"
        )

        for attempt in range(max_retries):
            try:
                logger.debug(f"Cleaning text with {self.model} (attempt {attempt + 1}/{max_retries})")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.1
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    cleaned_text = response.json().get("response", "").strip()
                    return cleaned_text if cleaned_text else text
                else:
                    logger.warning(f"Ollama returned status {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to clean text after {max_retries} attempts")
                        return text

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout cleaning text (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Text cleaning timed out after {max_retries} attempts")
                    return text
            except Exception as e:
                logger.error(f"Error cleaning text: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return text

        return text

    def clean_extraction_result(self, extraction_json: Dict) -> Dict:
        """
        Clean all text in an extraction result

        IMPORTANT: OCR text from images is cleaned but kept SEPARATE from PDF text
        This preserves the distinction between:
        - PDF text: procedural instructions and documentation
        - OCR text: visual context from interface screenshots
        """
        pdf_name = extraction_json['metadata'].get('pdf_name', 'unknown')
        logger.info(f"Cleaning extraction result for {pdf_name}")

        cleaned_result = extraction_json.copy()
        cleaned_pages = []

        for page_data in extraction_json['pages']:
            cleaned_page = page_data.copy()

            # Clean PDF text
            pdf_text = page_data.get('pdf_text', '').strip()

            if pdf_text:
                logger.debug(f"Cleaning page {page_data['page_num']} PDF text ({len(pdf_text)} chars)")
                cleaned_pdf_text = self.clean_text_with_llm(pdf_text)
                logger.debug(f"Cleaned to {len(cleaned_pdf_text)} chars")
                cleaned_page['pdf_text'] = cleaned_pdf_text
            else:
                cleaned_page['pdf_text'] = ""

            # Clean OCR text from images (but keep separate - don't synthesize)
            cleaned_images = []
            for img_data in page_data.get('images', []):
                cleaned_img = img_data.copy()

                if img_data.get('ocr_text', '').strip():
                    ocr_text = img_data['ocr_text']
                    logger.debug(f"Cleaning image OCR ({len(ocr_text)} chars)")
                    cleaned_ocr_text = self.clean_text_with_llm(ocr_text)
                    logger.debug(f"Cleaned to {len(cleaned_ocr_text)} chars")
                    cleaned_img['ocr_text'] = cleaned_ocr_text
                else:
                    cleaned_img['ocr_text'] = ""

                cleaned_images.append(cleaned_img)

            cleaned_page['images'] = cleaned_images
            cleaned_pages.append(cleaned_page)

        cleaned_result['pages'] = cleaned_pages
        cleaned_result['metadata']['cleaned_at'] = datetime.now().isoformat()
        cleaned_result['metadata']['cleaning_notes'] = (
            "Two-stage cleaning: 1) Boilerplate removal, 2) LLM semantic fixing. "
            "OCR text kept separate from PDF text to preserve visual context distinction."
        )

        return cleaned_result


class CleaningPipeline:
    """Manage text cleaning for customers"""

    def __init__(self, server_root: Path, ollama_url: str = "http://localhost:11434"):
        global logger

        self.server_root = Path(server_root)
        self.ollama_url = ollama_url

        logs_dir = self.server_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - CLEANING - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(str(logs_dir / 'cleaning.log'), encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            logger = logging.getLogger(__name__)

        logger.info(f"Cleaning pipeline initialized with root: {self.server_root}")
        logger.info(f"Ollama URL: {self.ollama_url}")

    def clean_single_file(self, json_path: Path) -> bool:
        """Clean a single extracted JSON file"""

        if not json_path.exists():
            logger.error(f"File not found: {json_path}")
            return False

        try:
            logger.info(f"Cleaning file: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                extraction_result = json.load(f)

            cleaner = TextCleaner(self.ollama_url)
            cleaned_result = cleaner.clean_extraction_result(extraction_result)

            # Save with _cleaned suffix
            cleaned_path = json_path.parent / f"{json_path.stem}_cleaned.json"
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved cleaned result to {cleaned_path}")
            print(f"[OK] Cleaned: {json_path.name} -> {cleaned_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clean {json_path}: {e}")
            print(f"[ERROR] {json_path.name}: {e}")
            return False

    def clean_customer(self, customer_id: str) -> Dict:
        """Clean all extracted PDFs for one customer"""
        customer_dir = self.server_root / "customers" / customer_id
        extracted_dir = customer_dir / customer_id

        if not extracted_dir.exists():
            logger.warning(f"No extracted documents folder for {customer_id}")
            return {
                'customer_id': customer_id,
                'status': 'no_documents',
                'total_documents': 0,
                'cleaned': 0,
                'failed': 0
            }

        extracted_files = list(extracted_dir.glob("*/content.json"))

        cleaned_count = 0
        failed_count = 0

        print(f"\n[CLEAN] Starting cleaning for customer: {customer_id}")
        print("="*70)

        for extracted_json_path in sorted(extracted_files):
            try:
                print(f"  [{extracted_json_path.parent.name}] Cleaning...")

                with open(extracted_json_path, 'r', encoding='utf-8') as f:
                    extraction_result = json.load(f)

                cleaner = TextCleaner(self.ollama_url)
                cleaned_result = cleaner.clean_extraction_result(extraction_result)

                cleaned_json_path = extracted_json_path.parent / "content_cleaned.json"
                with open(cleaned_json_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_result, f, indent=2, ensure_ascii=False)

                print(f"    [OK] {len(cleaned_result['pages'])} pages cleaned")
                cleaned_count += 1

            except Exception as e:
                logger.error(f"Failed to clean {extracted_json_path.parent.name}: {e}")
                print(f"    [ERROR] {e}")
                failed_count += 1

        print("="*70)
        print(f"[SUMMARY] {customer_id}")
        print(f"  Cleaned: {cleaned_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(extracted_files)}")

        return {
            'customer_id': customer_id,
            'status': 'success' if failed_count == 0 else 'partial',
            'total_documents': len(extracted_files),
            'cleaned': cleaned_count,
            'failed': failed_count
        }

    def clean_all_customers(self, customer_filter: Optional[List[str]] = None) -> Dict:
        """Clean extracted documents for all customers"""
        print("\n" + "="*70)
        print("TEXT CLEANING PIPELINE")
        print("="*70)

        customers_dir = self.server_root / "customers"

        if not customers_dir.exists():
            logger.error(f"Customers directory not found: {customers_dir}")
            return {}

        all_customers = []
        for customer_folder in sorted(customers_dir.glob("*")):
            if customer_folder.is_dir():
                extracted_dir = customer_folder / customer_folder.name
                if extracted_dir.exists() and any(extracted_dir.glob("*/content.json")):
                    all_customers.append(customer_folder.name)

        print(f"\n[DISCOVER] Found {len(all_customers)} customers with extracted documents")
        for customer_id in all_customers:
            print(f"  - {customer_id}")

        if customer_filter:
            all_customers = [c for c in all_customers if c in customer_filter]
            print(f"\n[FILTER] Processing {len(all_customers)} selected customers")

        results = {}
        for customer_id in all_customers:
            result = self.clean_customer(customer_id)
            results[customer_id] = result

        # Print final summary
        print("\n" + "="*70)
        print("CLEANING SUMMARY")
        print("="*70)

        total_documents = sum(r['total_documents'] for r in results.values())
        total_cleaned = sum(r['cleaned'] for r in results.values())
        total_failed = sum(r['failed'] for r in results.values())

        for customer_id, result in sorted(results.items()):
            status_icon = "[OK]" if result['status'] == 'success' else "[WARN]"
            print(f"{status_icon} {customer_id:25} {result['cleaned']}/{result['total_documents']} cleaned")

        print("\n" + "-"*70)
        print(f"Total Documents: {total_documents}")
        print(f"Cleaned: {total_cleaned}")
        print(f"Failed: {total_failed}")
        print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Text Cleaning Pipeline for Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean single file
  python clean_documents_server.py --file /path/to/content.json

  # Clean all documents for one customer
  python clean_documents_server.py --customer CUSTOMER_ID

  # Clean all documents
  python clean_documents_server.py --all

  # Clean specific customers
  python clean_documents_server.py --customers CUST1 CUST2
        """
    )

    parser.add_argument('--server-root', default='/home/aiadmin/netovo_voicebot/kokora/audiosocket',
                       help='Server root directory')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='Ollama API URL')
    parser.add_argument('--file', type=Path, help='Clean single extraction file')
    parser.add_argument('--customer', help='Clean all documents for one customer')
    parser.add_argument('--customers', nargs='+', help='Clean specific customers')
    parser.add_argument('--all', action='store_true', help='Clean all customers')

    args = parser.parse_args()

    if DOTENV_AVAILABLE:
        load_dotenv()

    server_root = Path(args.server_root)

    try:
        pipeline = CleaningPipeline(server_root, args.ollama_url)

        if args.file:
            # Clean single file
            success = pipeline.clean_single_file(args.file)
            sys.exit(0 if success else 1)

        elif args.customer:
            # Clean one customer
            result = pipeline.clean_customer(args.customer)
            sys.exit(0 if result['status'] in ['success', 'no_documents'] else 1)

        elif args.customers:
            # Clean specific customers
            results = pipeline.clean_all_customers(customer_filter=args.customers)
            failed = sum(1 for r in results.values() if r['status'] not in ['success', 'no_documents'])
            sys.exit(1 if failed > 0 else 0)

        elif args.all:
            # Clean all customers
            results = pipeline.clean_all_customers()
            failed = sum(1 for r in results.values() if r['status'] not in ['success', 'no_documents'])
            sys.exit(1 if failed > 0 else 0)

        else:
            parser.print_help()
            sys.exit(0)

    except Exception as e:
        logger.error(f"Cleaning pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
