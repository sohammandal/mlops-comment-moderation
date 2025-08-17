#!/usr/bin/env python3
"""
High-Performance Text Augmentation Pipeline using TextAttack

This module provides an optimized text augmentation system using TextAttack's
WordSwapWordNet transformation. Features include:
- Multiprocessing with intelligent core reservation
- Caching system for avoiding duplicate augmentations
- Checkpoint-based fault tolerance for resumable processing
- Platform-specific optimizations (Windows/Linux)
- Resource management and progress tracking

Optimized for high-vCPU environments like Vertex AI Workbench.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import platform
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import pandas as pd
import psutil
from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps import WordSwapWordNet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('textattack_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Platform-specific optimizations
if platform.system() == 'Linux':
    mp.set_start_method('fork', force=True)
elif platform.system() == 'Windows':
    mp.set_start_method('spawn', force=True)

# Pre-download NLTK data once before starting worker processes
# This is a key optimization to prevent redundant downloads.
logger.info("Downloading required NLTK packages...")
for resource in ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords', 'universal_tagset', 'punkt']:
    nltk.download(resource, quiet=True)
logger.info("NLTK packages are ready.")


class CheckpointManager:
    """Thread-safe checkpoint manager for resumable processing."""

    def __init__(self, checkpoint_file: str = 'textattack_checkpoint.json'):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data: Dict = {}
        self.lock = threading.Lock()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load existing checkpoint from disk."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_data = json.load(f)
                completed_count = len(self.checkpoint_data.get('completed_batches', []))
                logger.info(f"Loaded checkpoint: {completed_count} completed batches")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self.checkpoint_data = {}
        else:
            self.checkpoint_data = {}

    def save_checkpoint(self, total_batches: int, completed_batches: List[int],
                       results: Dict[str, str]) -> None:
        """Save current progress to disk."""
        with self.lock:
            self.checkpoint_data = {
                'total_batches': total_batches,
                'completed_batches': completed_batches,
                'results': results,
                'timestamp': time.time()
            }
            try:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.checkpoint_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    def get_completed_batches(self) -> set:
        """Get set of completed batch indices."""
        return set(self.checkpoint_data.get('completed_batches', []))

    def get_partial_results(self) -> Dict[str, str]:
        """Get previously computed results."""
        return self.checkpoint_data.get('results', {})

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info("Checkpoint cleared after successful completion")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")


class CacheManager:
    """Thread-safe cache manager for text augmentation results."""

    def __init__(self, cache_file: str = 'textattack_cache.pkl'):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, str] = {}
        self.lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def get(self, key: str) -> Optional[str]:
        """Get cached result."""
        with self.lock:
            return self.cache.get(key)

    def set(self, key: str, value: str) -> None:
        """Cache a result."""
        with self.lock:
            self.cache[key] = value

    def save_cache(self) -> None:
        """Save cache to disk safely."""
        with self.lock:
            try:
                temp_file = self.cache_file.with_suffix('.tmp')
                with open(temp_file, 'wb') as f:
                    pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)

                backup_file = self.cache_file.with_suffix('.bak')

                if self.cache_file.exists():
                    if backup_file.exists():
                        backup_file.unlink()
                    self.cache_file.rename(backup_file)

                temp_file.rename(self.cache_file)

                if backup_file.exists():
                    backup_file.unlink()

            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
                # Fallback to a simpler, less safe save
                try:
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e2:
                    logger.error(f"Cache save completely failed: {e2}")

    def __len__(self) -> int:
        return len(self.cache)

# Global augmenter and cache manager for worker processes
augmenter: Optional[Augmenter] = None
cache_manager: Optional[CacheManager] = None


def init_worker() -> None:
    """Initialize worker process with augmenter."""
    global augmenter
    # The NLTK data is already downloaded in the main process.
    # Initialize augmenter with optimized settings.
    augmenter = Augmenter(
        transformation=WordSwapWordNet(),
        pct_words_to_swap=0.05,
        transformations_per_example=1
    )


def augment_text_batch(texts_with_indices: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Augment a batch of texts with caching.

    Args:
        texts_with_indices: List of (index, text) tuples

    Returns:
        List of (index, augmented_text) tuples
    """
    results = []

    for idx, text in texts_with_indices:
        if not isinstance(text, str) or len(text.strip()) == 0:
            results.append((idx, text))
            continue

        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()

        if cache_manager:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                results.append((idx, cached_result))
                continue

        try:
            augmented = augmenter.augment(text)[0]
            if cache_manager:
                cache_manager.set(cache_key, augmented)
            results.append((idx, augmented))
        except Exception as e:
            logger.warning(f"Augmentation failed for text index {idx}: {e}")
            results.append((idx, text))

    return results


def calculate_optimal_processes() -> int:
    """Calculate optimal number of processes based on system resources."""
    cpu_count = mp.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    if cpu_count >= 100:
        reserved_cores = min(6, max(4, cpu_count // 25))
        max_processes = min(cpu_count - reserved_cores, 128)
    elif cpu_count >= 50:
        reserved_cores = min(4, max(3, cpu_count // 20))
        max_processes = min(cpu_count - reserved_cores, 64)
    elif cpu_count >= 20:
        reserved_cores = min(3, max(2, cpu_count // 15))
        max_processes = min(cpu_count - reserved_cores, 32)
    else:
        reserved_cores = min(2, max(1, cpu_count // 10))
        max_processes = max(cpu_count - reserved_cores, 1)

    memory_constrained_processes = int(available_memory_gb * 0.8)
    optimal_processes = min(max_processes, memory_constrained_processes)

    logger.info(f"System Resources:")
    logger.info(f"  CPU cores: {cpu_count} (reserving {reserved_cores})")
    logger.info(f"  Available memory: {available_memory_gb:.1f} GB")
    logger.info(f"  Optimal processes: {optimal_processes}")

    return optimal_processes

def process_dataframe(df: pd.DataFrame,
                     num_processes: Optional[int] = None,
                     batch_size: int = 25) -> List[str]:
    """
    Process dataframe with parallel text augmentation.

    Args:
        df: Input dataframe with 'comment_text' column
        num_processes: Number of worker processes (auto-detected if None)
        batch_size: Number of comments per batch

    Returns:
        List of augmented text strings
    """
    global cache_manager
    if num_processes is None:
        num_processes = calculate_optimal_processes()

    cache_manager = CacheManager()
    checkpoint_manager = CheckpointManager()

    comments = df['comment_text'].tolist()
    total_comments = len(comments)

    batches = []
    for i in range(0, total_comments, batch_size):
        batch_texts = [(j, comments[j]) for j in range(i, min(i + batch_size, total_comments))]
        batches.append(batch_texts)

    completed_batch_indices = checkpoint_manager.get_completed_batches()
    partial_results = checkpoint_manager.get_partial_results()

    results = [None] * total_comments

    if partial_results:
        for idx_str, result in partial_results.items():
            results[int(idx_str)] = result
        logger.info(f"Resuming from checkpoint: {len(completed_batch_indices)} batches completed")

    pending_batches = [(i, batch) for i, batch in enumerate(batches)
                      if i not in completed_batch_indices]

    if not pending_batches:
        logger.info("All batches already completed!")
        return results

    logger.info(f"Processing {total_comments:,} comments in {len(pending_batches)} batches "
               f"using {num_processes} processes")

    completed_batches_count = len(completed_batch_indices)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker) as executor:
        future_to_batch_idx = {
            executor.submit(augment_text_batch, batch): batch_idx
            for batch_idx, batch in pending_batches
        }

        for future in as_completed(future_to_batch_idx):
            batch_idx = future_to_batch_idx[future]
            batch_results = future.result()
            completed_batches_count += 1
            completed_batch_indices.add(batch_idx)

            current_results = {}
            for idx, augmented_text in batch_results:
                results[idx] = augmented_text
                current_results[str(idx)] = augmented_text
            
            partial_results.update(current_results)

            processed_comments = min(completed_batches_count * batch_size, total_comments)
            
            # Save checkpoint periodically
            if completed_batches_count % 10 == 0:
                 checkpoint_manager.save_checkpoint(len(batches), list(completed_batch_indices), partial_results)

            if completed_batches_count % max(1, len(pending_batches) // 20) == 0:
                elapsed = time.time() - start_time
                avg_speed = (processed_comments - len(checkpoint_manager.get_completed_batches()) * batch_size) / elapsed if elapsed > 0 else 0
                remaining = total_comments - processed_comments
                eta_minutes = (remaining / avg_speed) / 60 if avg_speed > 0 else 0

                logger.info(f"Progress: {processed_comments:,}/{total_comments:,} comments "
                           f"({avg_speed:.1f}/s, ETA: {eta_minutes:.1f}min, "
                           f"Cache: {len(cache_manager)} entries)")

    cache_manager.save_cache()
    checkpoint_manager.clear_checkpoint()
    return results

def process_file(input_file: str,
                output_file: Optional[str] = None,
                num_processes: Optional[int] = None,
                debug: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Process input file with text augmentation.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (auto-generated if None)
        num_processes: Number of worker processes
        debug: Enable debug mode (process only first 100 rows)

    Returns:
        Tuple of (processed_dataframe, output_file_path)
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    if debug:
        original_len = len(df)
        df = df.head(100)
        logger.info(f"Debug mode: Processing first 100 rows (out of {original_len} total)")

    if output_file is None:
        input_path = Path(input_file)
        suffix = '_debug_augmented' if debug else '_augmented'
        output_file = str(input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}"))

    batch_size = 10 if debug else 25

    augmented_texts = process_dataframe(
        df,
        num_processes=num_processes,
        batch_size=batch_size
    )

    df['comment_text_v2'] = augmented_texts
    df.to_csv(output_file, index=False)

    return df, output_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='High-performance text augmentation using TextAttack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/comments.csv
  %(prog)s --debug --processes 4
  %(prog)s --input large_file.csv --processes 96
        """
    )

    parser.add_argument(
        '--input', '-i',
        default='data/comments_test_01.csv',
        help='Input CSV file path (default: %(default)s)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (auto-generated if not specified)'
    )
    parser.add_argument(
        '--processes', '-p',
        type=int,
        help='Number of worker processes (default: auto-detect optimal)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Debug mode: process only first 100 rows with smaller batches'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear existing checkpoint and start fresh'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.clear_checkpoint:
        checkpoint_file = Path('textattack_checkpoint.json')
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint cleared")

    if args.debug:
        logger.info("Debug mode enabled:")
        logger.info("  - Processing first 100 rows only")
        logger.info("  - Using smaller batch sizes")

    start_time = time.time()

    try:
        df, output_path = process_file(
            args.input,
            args.output,
            args.processes,
            args.debug
        )

        elapsed_total = time.time() - start_time
        total_comments = len(df)

        original_comments = df['comment_text'].tolist()
        augmented_comments = df['comment_text_v2'].tolist()
        
        # Handle cases where augmentation might fail and return None
        new_augmentations = sum(
            1 for orig, aug in zip(original_comments, augmented_comments)
            if aug is not None and orig != aug
        )
        
        # A more accurate cache hit calculation
        cache_hits = sum(1 for aug in augmented_comments if aug is not None) - new_augmentations


        logger.info("Processing completed successfully!")
        logger.info(f"Total time: {elapsed_total/60:.1f} minutes")
        logger.info(f"Comments processed: {total_comments:,}")
        if elapsed_total > 0:
            logger.info(f"Processing rate: {total_comments/elapsed_total:.1f} comments/second")
        logger.info(f"New augmentations: {new_augmentations:,}")
        logger.info(f"Cache hits: {cache_hits:,}")
        if total_comments > 0:
            logger.info(f"Cache efficiency: {cache_hits/total_comments*100:.1f}%")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    if platform.system() == 'Windows':
        mp.freeze_support()
    main()
