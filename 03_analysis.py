"""
03_analysis.py

Statistical analysis for humor incongruity study.

Runs four analyses:
  1. Descriptive statistics
  2. Measure independence (SBERT vs GPT-2 correlation)
  3. Main correlations with humor rating + multiple regression
  4. Subgroup analysis by joke type (exploratory, N=15 discourse-centered)


Significance threshold: alpha = 0.05 throughout.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE   = 'results/jokes_with_measures.csv'
OUTPUT_TXT   = 'results/analysis_results.txt'
FIGURE_FILE  = 'results/figure1_scatter.png'
ALPHA        = 0.05
OUTLIER_SD   = 3     # SD threshold for robustness check
# ─────────────────────────────────────────────────────────────────────────────


def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return '(n.s.)'


def print_section(title, output_lines):
    line = f"\n{'='*60}\n{title}\n{'='*60}"
    print(line)
    output_lines.append(line)


def log(text, output_lines):
    print(text)
    output_lines.append(text)


def main():
    output_lines = []

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df_clean = df.dropna(subset=['sbert_cosine_distance', 'gpt2_surprisal']).copy()
    print(f"Full dataset: {len(df_clean)} jokes after dropping NaN values.")

    # Standardise predictors for regression
    df_clean['sbert_z'] = (df_clean['sbert_cosine_distance'] - df_clean['sbert_cosine_distance'].mean()) / df_clean['sbert_cosine_distance'].std()
    df_clean['gpt2_z']  = (df_clean['gpt2_surprisal']        - df_clean['gpt2_surprisal'].mean())        / df_clean['gpt2_surprisal'].std()

    # ── 1. Descriptive statistics ──────────────────────────────────────────────
    print_section("1. DESCRIPTIVE STATISTICS", output_lines)
    for col, label in [
        ('humor_rating',          'Humor rating'),
        ('sbert_cosine_distance', 'SBERT cosine distance'),
        ('gpt2_surprisal',        'GPT-2 surprisal (mean/token)')
    ]:
        s = df_clean[col]
        line = (f"{label:<35} N={len(s)}  Mean={s.mean():.3f}  "
                f"SD={s.std():.3f}  Min={s.min():.3f}  Max={s.max():.3f}")
        log(line, output_lines)

    # ── 2. Measure independence ────────────────────────────────────────────────
    print_section("2. MEASURE INDEPENDENCE (SBERT vs GPT-2)", output_lines)
    r_overlap, p_overlap = stats.pearsonr(
        df_clean['sbert_cosine_distance'],
        df_clean['gpt2_surprisal']
    )
    shared_var = r_overlap ** 2 * 100
    log(f"r({len(df_clean)-2}) = {r_overlap:.3f},  p = {p_overlap:.4f}  {sig_label(p_overlap)}", output_lines)
    log(f"Shared variance: {shared_var:.1f}%", output_lines)

    # ── 3. Main correlations with humor rating ─────────────────────────────────
    print_section("3. CORRELATIONS WITH HUMOR RATING", output_lines)
    r_sbert, p_sbert = stats.pearsonr(df_clean['sbert_cosine_distance'], df_clean['humor_rating'])
    r_gpt2,  p_gpt2  = stats.pearsonr(df_clean['gpt2_surprisal'],        df_clean['humor_rating'])
    n = len(df_clean)

    log(f"SBERT cosine distance:  r({n-2}) = {r_sbert:.3f},  p = {p_sbert:.4f}  {sig_label(p_sbert)}", output_lines)
    log(f"GPT-2 surprisal:        r({n-2}) = {r_gpt2:.3f},  p = {p_gpt2:.4f}  {sig_label(p_gpt2)}", output_lines)

    # Multiple regression
    print_section("3b. MULTIPLE REGRESSION", output_lines)
    model_full = smf.ols('humor_rating ~ sbert_z + gpt2_z', data=df_clean).fit()
    log(f"R² = {model_full.rsquared:.4f},  Adjusted R² = {model_full.rsquared_adj:.4f}", output_lines)
    log(f"F({model_full.df_model:.0f}, {model_full.df_resid:.0f}) = {model_full.fvalue:.3f},  p = {model_full.f_pvalue:.4f}  {sig_label(model_full.f_pvalue)}", output_lines)
    for name, coef, se, p in zip(model_full.params.index, model_full.params, model_full.bse, model_full.pvalues):
        log(f"  {name:<20} β = {coef:.4f}  SE = {se:.4f}  p = {p:.4f}  {sig_label(p)}", output_lines)

    # Robustness check: remove GPT-2 outliers (>3 SD)
    print_section("3c. ROBUSTNESS CHECK (GPT-2 outliers removed)", output_lines)
    threshold = df_clean['gpt2_surprisal'].mean() + OUTLIER_SD * df_clean['gpt2_surprisal'].std()
    df_robust = df_clean[df_clean['gpt2_surprisal'] <= threshold]
    r_robust, p_robust = stats.pearsonr(df_robust['gpt2_surprisal'], df_robust['humor_rating'])
    n_removed = len(df_clean) - len(df_robust)
    log(f"Removed {n_removed} outliers (>{OUTLIER_SD} SD above mean)", output_lines)
    log(f"GPT-2 vs humor (robust): r({len(df_robust)-2}) = {r_robust:.3f},  p = {p_robust:.4f}  {sig_label(p_robust)}", output_lines)

    # ── 4. Subgroup analysis by joke type ──────────────────────────────────────
    print_section("4. SUBGROUP ANALYSIS BY JOKE TYPE (exploratory)", output_lines)
    log("Note: N=15 for discourse-centered. Results are not statistically reliable.", output_lines)
    log("Power analysis: ~80-100 jokes per type needed for medium interaction effect.", output_lines)

    df_typed = df_clean[df_clean['joke_type'].notna() & (df_clean['joke_type'] != '')].copy()

    for jtype in ['frame-based', 'discourse-centered']:
        sub = df_typed[df_typed['joke_type'] == jtype]
        r_s, p_s = stats.pearsonr(sub['sbert_cosine_distance'], sub['humor_rating'])
        r_g, p_g = stats.pearsonr(sub['gpt2_surprisal'],        sub['humor_rating'])

        model_sub = smf.ols('humor_rating ~ sbert_z + gpt2_z', data=sub).fit()

        log(f"\n{jtype.upper()} (N={len(sub)})", output_lines)
        log(f"  SBERT vs humor:  r({len(sub)-2}) = {r_s:.3f},  p = {p_s:.4f}  {sig_label(p_s)}", output_lines)
        log(f"  GPT-2 vs humor:  r({len(sub)-2}) = {r_g:.3f},  p = {p_g:.4f}  {sig_label(p_g)}", output_lines)
        log(f"  Regression R² = {model_sub.rsquared:.3f}", output_lines)

    # Raw measure comparison across groups
    log("\nRaw measure values by joke type:", output_lines)
    for measure in ['sbert_cosine_distance', 'gpt2_surprisal']:
        fb = df_typed[df_typed['joke_type'] == 'frame-based'][measure]
        dc = df_typed[df_typed['joke_type'] == 'discourse-centered'][measure]
        t, p = stats.ttest_ind(fb, dc)
        log(f"  {measure}: frame={fb.mean():.3f}, discourse={dc.mean():.3f},  t={t:.3f},  p={p:.3f}  {sig_label(p)}", output_lines)

    # ── Figure 1 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 1: Scatter plots for main analyses', fontsize=13)

    axes[0].scatter(df_clean['sbert_cosine_distance'], df_clean['humor_rating'],
                    alpha=0.2, s=8, color='steelblue')
    axes[0].set_xlabel('SBERT cosine distance')
    axes[0].set_ylabel('Humor rating')
    axes[0].set_title(f'SBERT vs Humor\nr = {r_sbert:.3f}, p = {p_sbert:.3f}')

    axes[1].scatter(df_clean['gpt2_surprisal'], df_clean['humor_rating'],
                    alpha=0.2, s=8, color='coral')
    axes[1].set_xlabel('GPT-2 surprisal (mean/token)')
    axes[1].set_ylabel('Humor rating')
    axes[1].set_title(f'GPT-2 vs Humor\nr = {r_gpt2:.3f}, p < .001')

    axes[2].scatter(df_clean['sbert_cosine_distance'], df_clean['gpt2_surprisal'],
                    alpha=0.2, s=8, color='mediumseagreen')
    axes[2].set_xlabel('SBERT cosine distance')
    axes[2].set_ylabel('GPT-2 surprisal (mean/token)')
    axes[2].set_title(f'SBERT vs GPT-2\nr = {r_overlap:.3f}, p < .001')

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150)
    print(f"\nFigure saved to {FIGURE_FILE}")

    # ── Save results ───────────────────────────────────────────────────────────
    with open(OUTPUT_TXT, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"Results saved to {OUTPUT_TXT}")


if __name__ == '__main__':
    main()
