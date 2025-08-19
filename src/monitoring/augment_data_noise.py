# src/monitoring/augment_data_noise.py
"""
Lightweight noise-based augmentation:
- Random character typos (insert/delete/replace)
- Random word dropout
Creates a new test dataset: comments_test_v3.csv
"""

import os
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Paths
INPUT_FILE = os.path.join("assets", "comments_test_v2.csv")
OUTPUT_FILE = os.path.join("assets", "comments_test_v3.csv")


def add_typos(text: str, char_prob: float = 0.03) -> str:
    """Introduce random character-level noise into a string."""
    out = []
    for ch in text:
        if random.random() < char_prob:
            choice = random.choice(["delete", "replace", "insert"])
            if choice == "delete":
                continue  # skip char
            elif choice == "replace":
                out.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
            elif choice == "insert":
                out.append(ch)
                out.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out)


def word_dropout(text: str, drop_prob: float = 0.1) -> str:
    """Randomly drop some words from the text."""
    words = word_tokenize(text)
    kept = [w for w in words if random.random() > drop_prob]
    return " ".join(kept) if kept else text


def augment_text(text: str) -> str:
    """Apply both noise augmentations."""
    if not isinstance(text, str) or not text.strip():
        return text
    noisy = add_typos(text)
    noisy = word_dropout(noisy)
    return noisy


def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    if "comment_text" not in df.columns:
        raise ValueError("CSV must contain a 'comment_text' column")

    print(f"Augmenting {len(df)} rows with noise...")
    tqdm.pandas()
    df["comment_text"] = df["comment_text"].progress_apply(augment_text)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved augmented dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
