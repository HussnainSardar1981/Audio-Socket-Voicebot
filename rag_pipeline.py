"""
Master RAG Pipeline Orchestrator
Runs all stages: Download -> Extract -> Clean -> Embed -> Index
For one customer, all customers, or a single file
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - RAG_PIPELINE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
SCRIPTS = {
    'download': SCRIPT_DIR / 'downloader.py',
    'extract': SCRIPT_DIR / 'extraction.py',
    'clean': SCRIPT_DIR / 'cleaner.py',
    'embed': SCRIPT_DIR / 'embeddings.py',
    'index': SCRIPT_DIR / 'indexing.py',
}

STAGES = ['download', 'extract', 'clean', 'embed', 'index']


class RAGPipeline:
    """Orchestrate the complete RAG pipeline"""

    def __init__(self, skip_stages=None, only_stages=None):
        """
        Initialize pipeline

        Args:
            skip_stages: List of stages to skip (e.g., ['download', 'extract'])
            only_stages: Only run these stages (e.g., ['embed', 'index'])
        """
        self.skip_stages = skip_stages or []
        self.only_stages = only_stages or []
        self.results = {}

    def get_stages_to_run(self):
        """Determine which stages to run"""
        if self.only_stages:
            return [s for s in STAGES if s in self.only_stages]
        return [s for s in STAGES if s not in self.skip_stages]

    def run_stage(self, stage, args):
        """
        Run a single stage

        Args:
            stage: Stage name (download, extract, clean, embed, index)
            args: List of command line arguments for the stage script

        Returns:
            True if successful, False otherwise
        """
        script = SCRIPTS.get(stage)
        if not script or not script.exists():
            logger.error(f"Script not found: {script}")
            return False

        print(f"\n{'='*70}")
        print(f"RUNNING STAGE: {stage.upper()}")
        print(f"{'='*70}\n")

        try:
            cmd = ['python', str(script)] + args
            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=False)

            self.results[stage] = 'success'
            logger.info(f"Stage {stage} completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            self.results[stage] = 'failed'
            logger.error(f"Stage {stage} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            self.results[stage] = 'error'
            logger.error(f"Stage {stage} error: {e}")
            return False

    def run_all_customers(self, skip_stages=None, only_stages=None):
        """
        Run full pipeline for all customers

        Args:
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        print(f"\n{'='*70}")
        print("RAG PIPELINE: ALL CUSTOMERS")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages: {', '.join(stages)}")

        for stage in stages:
            success = self.run_stage(stage, [])
            if not success:
                logger.warning(f"Stage {stage} failed. Continue anyway? (y/n)")
                if input().lower() != 'y':
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def run_single_customer(self, customer_id, skip_stages=None, only_stages=None):
        """
        Run full pipeline for a single customer

        Args:
            customer_id: Customer identifier
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        print(f"\n{'='*70}")
        print(f"RAG PIPELINE: CUSTOMER {customer_id}")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages for {customer_id}: {', '.join(stages)}")

        for stage in stages:
            args = ['--customer', customer_id]
            success = self.run_stage(stage, args)
            if not success:
                logger.warning(f"Stage {stage} failed for {customer_id}. Continue? (y/n)")
                if input().lower() != 'y':
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def run_single_file(self, file_path, customer_id=None, skip_stages=None, only_stages=None):
        """
        Run full pipeline for a single file

        Args:
            file_path: Path to file (PDF, JSON, etc.)
            customer_id: Customer identifier (optional, extracted from path if not provided)
            skip_stages: Stages to skip
            only_stages: Only run these stages
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        if not customer_id:
            customer_id = file_path.parent.parent.name

        print(f"\n{'='*70}")
        print(f"RAG PIPELINE: SINGLE FILE")
        print(f"File: {file_path}")
        print(f"Customer: {customer_id}")
        print(f"{'='*70}\n")

        if skip_stages:
            self.skip_stages = skip_stages
        if only_stages:
            self.only_stages = only_stages

        stages = self.get_stages_to_run()
        logger.info(f"Running stages for file: {', '.join(stages)}")

        for stage in stages:
            args = ['--file', str(file_path), '--customer', customer_id]

            success = self.run_stage(stage, args)
            if not success:
                logger.warning(f"Stage {stage} failed. Continue? (y/n)")
                if input().lower() != 'y':
                    logger.error("Pipeline aborted")
                    return False

        return self._print_summary()

    def _print_summary(self):
        """Print pipeline execution summary"""
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        for stage, status in self.results.items():
            icon = "✓" if status == 'success' else "✗"
            print(f"{icon} {stage:15} {status}")
        print(f"{'='*70}\n")

        all_success = all(s == 'success' for s in self.results.values())
        return all_success


def main():
    parser = argparse.ArgumentParser(
        description='RAG Pipeline Orchestrator - Run all stages at once',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for all customers
  python rag_pipeline.py --all

  # Run full pipeline for one customer
  python rag_pipeline.py --customer stuart_dean

  # Run full pipeline for a single file
  python rag_pipeline.py --file customers/stuart_dean/Documents/doc.pdf

  # Skip download/extract stages, only clean/embed/index
  python rag_pipeline.py --all --skip download extract

  # Run only embedding and indexing stages
  python rag_pipeline.py --all --only embed index

  # Run full pipeline for specific file with custom customer ID
  python rag_pipeline.py --file path/to/file.pdf --customer-id my_customer

Pipeline Stages (in order):
  1. download  - Download PDFs from SharePoint
  2. extract   - Extract text and images from PDFs
  3. clean     - Clean and fix extracted text
  4. embed     - Generate embeddings for chunks
  5. index     - Index embeddings into ChromaDB
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run for all customers'
    )
    parser.add_argument(
        '--customer',
        type=str,
        help='Run for a specific customer'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Run for a specific file'
    )
    parser.add_argument(
        '--customer-id',
        type=str,
        help='Customer ID (for use with --file)'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        choices=STAGES,
        help='Stages to skip'
    )
    parser.add_argument(
        '--only',
        nargs='+',
        choices=STAGES,
        help='Only run these stages'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='ChromaDB path (passed to indexing stage)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.customer and not args.file:
        parser.print_help()
        return 1

    if args.skip and args.only:
        logger.error("Cannot use --skip and --only together")
        return 1

    # Initialize pipeline
    pipeline = RAGPipeline(skip_stages=args.skip, only_stages=args.only)

    # Add db-path to environment if provided
    if args.db_path:
        os.environ['CHROMA_DB_PATH'] = args.db_path

    try:
        if args.all:
            success = pipeline.run_all_customers()
        elif args.customer:
            success = pipeline.run_single_customer(args.customer)
        elif args.file:
            success = pipeline.run_single_file(args.file, customer_id=args.customer_id)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
