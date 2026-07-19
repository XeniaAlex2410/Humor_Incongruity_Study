"""
Build an extended jokes CSV with additional text features.

Inputs
------
- data/jokes_final_measures.csv        (main jokes file)
- SUBTLEXus74286wordstextversion.txt   (word frequency table, tab-separated)
- SemEval-2021 Task 7 / HAHackathon file (OPTIONAL, for offensiveness)
    -> not present on this machine, so 'offensiveness' is left empty.

Output
------
- data/jokes_with_features.csv         (new file; nothing is overwritten)

Only pandas + the standard library are used.
"""

import os
import re
import math

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_DIR = "/Users/xeniabelobokov/Humor_Incongruity_Study"
JOKES_CSV = os.path.join(PROJECT_DIR, "data", "jokes_final_measures.csv")
SUBTLEX_TXT = os.path.join(PROJECT_DIR, "SUBTLEXus74286wordstextversion.txt")
OUTPUT_CSV = os.path.join(PROJECT_DIR, "data", "jokes_with_features.csv")

# SemEval-2021 Task 7 (HaHackathon) training split -- the only file with labels.
# Validated: all 2499 joke ids are in this file and humor_rating matches exactly.
SEMEVAL_CSV = os.path.join(
    PROJECT_DIR, "data",
    "SemEval2021-HaHackathon-UMUTeam-main", "datasets", "hahackathon_train.csv",
)
SEMEVAL_OFFENSE_COLUMN = "offense_rating"   # column name to merge

# --------------------------------------------------------------------------
# Tokenizer: lowercase, then split on whitespace AND punctuation.
# Keeps every word (including function words like "the", "a", "is").
# A "word" is a maximal run of alphanumeric characters.
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    if not isinstance(text, str):
        return []
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Load SUBTLEX and build a lowercase word -> Zipf lookup.
# SUBTLEX has no Zipf column, so compute it (van Heuven et al., 2014):
#     Zipf = log10(SUBTLWF) + 3        (SUBTLWF = frequency per million words)
# On lowercase-key collisions (e.g. "Apple" vs "apple") keep the higher freq.
# --------------------------------------------------------------------------
def load_subtlex_zipf(path):
    sub = pd.read_csv(path, sep="\t")
    sub["word_lower"] = sub["Word"].astype(str).str.lower()
    sub["zipf"] = sub["SUBTLWF"].apply(
        lambda wf: math.log10(wf) + 3 if wf > 0 else float("nan")
    )
    # keep the row with the highest frequency per lowercased word
    sub = sub.sort_values("SUBTLWF", ascending=False).drop_duplicates("word_lower")
    return dict(zip(sub["word_lower"], sub["zipf"]))


# --------------------------------------------------------------------------
# Per-text feature computation
# --------------------------------------------------------------------------
def text_features(text, zipf_lookup):
    tokens = tokenize(text)
    n_words = len(tokens)
    n_chars = sum(len(t) for t in tokens)

    avg_word_len = (n_chars / n_words) if n_words else float("nan")

    found = [zipf_lookup[t] for t in tokens if t in zipf_lookup]
    avg_freq = (sum(found) / len(found)) if found else float("nan")
    n_unknown = n_words - len(found)

    return n_words, n_chars, avg_word_len, avg_freq, n_unknown


def main():
    print("Loading jokes:", JOKES_CSV)
    df = pd.read_csv(JOKES_CSV)
    print(f"  {len(df)} jokes loaded, columns: {list(df.columns)}")

    print("Loading SUBTLEX:", SUBTLEX_TXT)
    zipf_lookup = load_subtlex_zipf(SUBTLEX_TXT)
    print(f"  {len(zipf_lookup)} unique lowercase words with Zipf values")

    # ---- compute setup and punchline features ----
    setup_feats = df["setup"].apply(lambda t: text_features(t, zipf_lookup))
    punch_feats = df["punchline"].apply(lambda t: text_features(t, zipf_lookup))

    s = pd.DataFrame(
        setup_feats.tolist(),
        columns=["w_s", "c_s", "awl_s", "freq_s", "unk_s"],
        index=df.index,
    )
    p = pd.DataFrame(
        punch_feats.tolist(),
        columns=["w_p", "c_p", "awl_p", "freq_p", "unk_p"],
        index=df.index,
    )

    # ---- word counts ----
    df["number_of_words_setup"] = s["w_s"]
    df["number_of_words_punchline"] = p["w_p"]
    df["number_of_words_total"] = s["w_s"] + p["w_p"]

    # ---- character counts ----
    df["number_of_characters_setup"] = s["c_s"]
    df["number_of_characters_punchline"] = p["c_p"]
    df["number_of_characters_total"] = s["c_s"] + p["c_p"]

    # ---- average word length (chars per word), computed over the whole part ----
    df["average_word_length_setup"] = s["awl_s"]
    df["average_word_length_punchline"] = p["awl_p"]
    total_words = df["number_of_words_total"]
    df["average_word_length_total"] = (
        df["number_of_characters_total"] / total_words
    ).where(total_words > 0)

    # ---- average word frequency (Zipf), averaged over found words ----
    df["average_frequency_setup"] = s["freq_s"]
    df["average_frequency_punchline"] = p["freq_p"]
    # 'total' averages over all found words in setup+punchline combined,
    # reconstructed from the per-part sums so it is a true pooled mean.
    found_sum_s = s["freq_s"] * (s["w_s"] - s["unk_s"])
    found_sum_p = p["freq_p"] * (p["w_p"] - p["unk_p"])
    found_n_s = s["w_s"] - s["unk_s"]
    found_n_p = p["w_p"] - p["unk_p"]
    found_n_total = found_n_s + found_n_p
    df["average_frequency_total"] = (
        (found_sum_s.fillna(0) + found_sum_p.fillna(0)) / found_n_total
    ).where(found_n_total > 0)

    # ---- unknown words ----
    df["number_of_unknown_words_setup"] = s["unk_s"]
    df["number_of_unknown_words_punchline"] = p["unk_p"]
    df["number_of_unknown_words_total"] = s["unk_s"] + p["unk_p"]

    # ---- offensiveness (merge by id, if the SemEval file is available) ----
    matched = 0
    if SEMEVAL_CSV and os.path.exists(SEMEVAL_CSV):
        sem = pd.read_csv(SEMEVAL_CSV)
        off = sem[["id", SEMEVAL_OFFENSE_COLUMN]].rename(
            columns={SEMEVAL_OFFENSE_COLUMN: "offensiveness"}
        )
        before = df["id"].isin(off["id"]).sum()
        df = df.merge(off, on="id", how="left")
        matched = df["offensiveness"].notna().sum()
        print(f"\nOffensiveness merge: {matched} of {len(df)} rows matched "
              f"({before} ids present in SemEval file).")
    else:
        df["offensiveness"] = pd.NA
        print("\nOffensiveness: SemEval file not available -> column left empty "
              "(0 rows matched).")

    # ---- save (derived output; safe to regenerate) ----
    if os.path.exists(OUTPUT_CSV):
        print(f"\nNote: regenerating existing derived file {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    # ---- checks ----
    total_unknown = int(df["number_of_unknown_words_total"].sum())
    unmatched_off = int(df["offensiveness"].isna().sum())

    print("\n================ CHECKS ================")
    print(f"Total rows: {len(df)}  (expected 2499)")
    print(f"\nAll columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print(f"\nOffensiveness could NOT be matched for {unmatched_off} jokes.")
    print(f"Total unknown words (not in SUBTLEX): {total_unknown}")

    new_cols = [
        "id",
        "number_of_words_setup", "number_of_words_punchline", "number_of_words_total",
        "number_of_characters_setup", "number_of_characters_punchline",
        "number_of_characters_total",
        "average_word_length_setup", "average_word_length_punchline",
        "average_word_length_total",
        "average_frequency_setup", "average_frequency_punchline",
        "average_frequency_total",
        "number_of_unknown_words_setup", "number_of_unknown_words_punchline",
        "number_of_unknown_words_total",
        "offensiveness",
    ]
    print("\nFirst 5 rows (new feature columns):")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[new_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
