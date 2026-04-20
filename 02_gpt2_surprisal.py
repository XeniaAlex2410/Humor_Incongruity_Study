"""
02_gpt2_surprisal.py

Computes mean per-token surprisal for each punchline using GPT-2 Small.

Model choice: GPT-2 Small 

Output: adds 'gpt2_surprisal' column to the dataset.

import pandas as pd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = 'results/jokes_with_sbert.csv'
OUTPUT_FILE = 'results/jokes_with_measures.csv'
MAX_TOKENS  = 1024   # GPT-2 context window limit
LOG_EVERY   = 500    # Print progress every N jokes
# ─────────────────────────────────────────────────────────────────────────────


def get_surprisal(setup, punchline, model, tokenizer):
    full_text = setup + ' ' + punchline
    full_ids  = tokenizer.encode(full_text, return_tensors='pt')
    setup_ids = tokenizer.encode(setup,     return_tensors='pt')

    punchline_start = setup_ids.shape[1]

    # Truncate to GPT-2 context window if needed
    if full_ids.shape[1] > MAX_TOKENS:
        full_ids = full_ids[:, :MAX_TOKENS]

    # Skip if punchline is entirely outside the context window
    if punchline_start >= full_ids.shape[1]:
        return np.nan

    with torch.no_grad():
        logits = model(full_ids).logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # For each punchline token, get its surprisal given all preceding tokens
    surprisals = [
        -log_probs[0, i - 1, full_ids[0, i].item()].item()
        for i in range(punchline_start, full_ids.shape[1])
    ]

    return np.mean(surprisals) if surprisals else np.nan


def main():
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} jokes.")

    print("\nLoading GPT-2 Small...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model     = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    print("Model loaded.")

    print(f"\nComputing surprisal (~25-30 min on CPU)...")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append(get_surprisal(row['setup'], row['punchline'], model, tokenizer))
        if (i + 1) % LOG_EVERY == 0:
            print(f"  {i + 1}/{len(df)} done...")

    df['gpt2_surprisal'] = results

    # Descriptive statistics
    surp = df['gpt2_surprisal']
    print(f"\nGPT-2 surprisal (mean per token):")
    print(f"  Mean:  {surp.mean():.3f}")
    print(f"  SD:    {surp.std():.3f}")
    print(f"  Min:   {surp.min():.3f}")
    print(f"  Max:   {surp.max():.3f}")
    print(f"  NaN:   {surp.isna().sum()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
