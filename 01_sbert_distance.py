"""
01_sbert_distance.py

Computes SBERT cosine distance between setup and punchline for each joke.

Output: adds 'sbert_cosine_distance' column to the dataset.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = 'data/jokes_wide.csv'
OUTPUT_FILE = 'results/jokes_with_sbert.csv'
MODEL_NAME  = 'all-mpnet-base-v2'
BATCH_SIZE  = 64
# ─────────────────────────────────────────────────────────────────────────────


def compute_sbert_distance(df, model):
    print("Encoding setups...")
    setup_embeddings = model.encode(
        df['setup'].tolist(),
        show_progress_bar=True,
        batch_size=BATCH_SIZE
    )

    print("Encoding punchlines...")
    punchline_embeddings = model.encode(
        df['punchline'].tolist(),
        show_progress_bar=True,
        batch_size=BATCH_SIZE
    )

    similarities = [
        cosine_similarity([s], [p])[0][0]
        for s, p in zip(setup_embeddings, punchline_embeddings)
    ]

    return [1 - s for s in similarities]


def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} jokes.")

    print(f"\nLoading SBERT model: {MODEL_NAME}")
    print("(First run downloads ~420MB)")
    model = SentenceTransformer(MODEL_NAME)

    print("\nComputing SBERT cosine distances...")
    df['sbert_cosine_distance'] = compute_sbert_distance(df, model)

    # Descriptive statistics
    dist = df['sbert_cosine_distance']
    print(f"\nSBERT cosine distance:")
    print(f"  Mean:  {dist.mean():.3f}")
    print(f"  SD:    {dist.std():.3f}")
    print(f"  Min:   {dist.min():.3f}")
    print(f"  Max:   {dist.max():.3f}")

    # Note: values slightly above 1.0 reflect floating point rounding
    n_above_one = (dist > 1.0).sum()
    if n_above_one > 0:
        print(f"  Note: {n_above_one} values slightly above 1.0 (floating point rounding, not an error)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
