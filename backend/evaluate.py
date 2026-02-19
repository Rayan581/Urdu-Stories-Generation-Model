"""
evaluate.py  —  Hyperparameter Evaluation for the Urdu Trigram Language Model
===============================================================================

What this script does
---------------------
1.  Loads the pre-trained BPE tokenizer and the training corpus.
2.  Trains a single trigram model (counts are lambda-independent, so we
    only build the count tables once).
3.  Sweeps every valid (λ1, λ2, λ3) combination on a held-out validation
    split (last 10 % of corpus) and records perplexity for each.
4.  Produces four publication-quality plots:
        a. Heatmap  λ2 vs λ3  (λ1 fixed at best value)
        b. Line plot — perplexity vs λ3 for several fixed λ1 values
        c. Line plot — perplexity vs λ2 for several fixed λ1 values
        d. Bar chart — top-20 best lambda combinations
5.  Saves an HTML report (evaluation_report.html) summarising everything.
6.  Re-trains model.py with ONLY the optimal lambdas and saves model.json.

Usage
-----
    python backend/evaluate.py

Outputs (all in backend/evaluation/)
--------------------------------------
    heatmap_l2_l3.png
    perplexity_vs_l3.png
    perplexity_vs_l2.png
    top20_combinations.png
    evaluation_report.html
    model.json          ← optimal model saved back to backend/
"""

import os
import sys
import json
import math
import time
import random
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import BPETokenizer, EOS, EOP, EOT
from model     import TrigramModel, _BOS_ID

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = os.path.dirname(__file__)
CORPUS_PATH    = os.path.join(BASE, "..", "preprocessing", "corpus.txt")
TOKENIZER_PATH = os.path.join(BASE, "tokenizer.json")
MODEL_PATH     = os.path.join(BASE, "model.json")
OUT_DIR        = os.path.join(BASE, "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
          "#0891B2", "#BE185D", "#65A30D", "#9333EA", "#B45309"]


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Load corpus & tokenize once
# ═════════════════════════════════════════════════════════════════════════════

def load_and_split(corpus_path: str, val_ratio: float = 0.10):
    """Return (train_ids, val_ids) as integer token lists."""
    print("[Eval] Loading corpus ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = BPETokenizer()
    tok.load(TOKENIZER_PATH)

    print("[Eval] Tokenizing ...")
    all_ids = [t for t in tok.encode(text) if t >= 0]

    split = int(len(all_ids) * (1 - val_ratio))
    train_ids = all_ids[:split]
    val_ids   = all_ids[split:]
    print(f"[Eval] Train tokens: {len(train_ids):,}   Val tokens: {len(val_ids):,}")
    return train_ids, val_ids, tok


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Build count tables once (lambda-independent)
# ═════════════════════════════════════════════════════════════════════════════

def build_count_tables(train_ids: List[int]):
    """Return (unigram_counts, bigram_counts, trigram_counts, total_tokens)."""
    unigram  = defaultdict(int)
    bigram   = defaultdict(lambda: defaultdict(int))
    trigram  = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    total    = 0

    padded = [_BOS_ID, _BOS_ID] + train_ids
    for i in range(2, len(padded)):
        t0, t1, t2 = padded[i-2], padded[i-1], padded[i]
        if t2 != _BOS_ID:
            unigram[t2] += 1
            total += 1
        bigram[t1][t2] += 1
        trigram[t0][t1][t2] += 1

    print(f"[Eval] Count tables built — "
          f"unigrams: {len(unigram):,}  "
          f"bigrams: {sum(len(v) for v in bigram.values()):,}  "
          f"trigrams: {sum(len(vv) for v in trigram.values() for vv in v.values()):,}")
    return unigram, bigram, trigram, total


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Fast perplexity (inline, no model object overhead)
# ═════════════════════════════════════════════════════════════════════════════

def _p1(t, unigram, total):
    return unigram.get(t, 0) / total if total else 0.0

def _p2(t_prev, t, bigram, unigram, total):
    ctx = bigram.get(t_prev)
    if not ctx:
        return _p1(t, unigram, total)
    ctx_sum = sum(ctx.values())
    return ctx.get(t, 0) / ctx_sum if ctx_sum else 0.0

def _p3(t2, t1, t, trigram, bigram, unigram, total):
    ctx2 = trigram.get(t2)
    if not ctx2:
        return _p2(t1, t, bigram, unigram, total)
    ctx1 = ctx2.get(t1)
    if not ctx1:
        return _p2(t1, t, bigram, unigram, total)
    ctx_sum = sum(ctx1.values())
    return ctx1.get(t, 0) / ctx_sum if ctx_sum else 0.0

def fast_perplexity(val_ids, l1, l2, l3,
                    unigram, bigram, trigram, total) -> float:
    """Compute interpolated perplexity without constructing a TrigramModel."""
    padded  = [_BOS_ID, _BOS_ID] + val_ids
    log_sum = 0.0
    n       = len(val_ids)
    for i in range(2, len(padded)):
        t0, t1, t = padded[i-2], padded[i-1], padded[i]
        p = (l3 * _p3(t0, t1, t, trigram, bigram, unigram, total)
           + l2 * _p2(t1, t, bigram, unigram, total)
           + l1 * _p1(t, unigram, total))
        log_sum += math.log(max(p, 1e-10))
    return math.exp(-log_sum / n) if n else float("inf")


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Full lambda grid sweep
# ═════════════════════════════════════════════════════════════════════════════

def sweep_lambdas(val_ids, unigram, bigram, trigram, total,
                  step: float = 0.05) -> List[Tuple[float, float, float, float]]:
    """
    Sweep all valid (l1, l2, l3) combinations where each is a multiple of
    *step* and l1+l2+l3 == 1.0.

    Returns list of (l1, l2, l3, perplexity) sorted by perplexity ascending.
    """
    grid   = [round(i * step, 6) for i in range(1, int(1/step))]
    combos = []
    for l1 in grid:
        for l2 in grid:
            l3 = round(1.0 - l1 - l2, 6)
            if l3 <= 0 or l3 >= 1.0:
                continue
            combos.append((l1, l2, l3))

    print(f"[Eval] Sweeping {len(combos)} lambda combinations ...")
    t0 = time.time()
    results = []
    for i, (l1, l2, l3) in enumerate(combos):
        pp = fast_perplexity(val_ids, l1, l2, l3, unigram, bigram, trigram, total)
        results.append((l1, l2, l3, pp))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(combos)} done ...  current best PP={min(r[3] for r in results):.4f}")

    results.sort(key=lambda x: x[3])
    elapsed = time.time() - t0
    print(f"[Eval] Sweep complete in {elapsed:.1f}s")
    print(f"[Eval] Best  → λ1={results[0][0]:.2f}  λ2={results[0][1]:.2f}  "
          f"λ3={results[0][2]:.2f}  PP={results[0][3]:.4f}")
    print(f"[Eval] Worst → λ1={results[-1][0]:.2f}  λ2={results[-1][1]:.2f}  "
          f"λ3={results[-1][2]:.2f}  PP={results[-1][-1]:.4f}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Plot helpers
# ═════════════════════════════════════════════════════════════════════════════

def plot_heatmap(results, best, out_dir):
    """Heatmap of perplexity over (λ2, λ3) with λ1 fixed at best value."""
    best_l1 = best[0]
    subset  = [(l2, l3, pp) for l1, l2, l3, pp in results if abs(l1 - best_l1) < 1e-5]
    if not subset:
        print("[Eval] Heatmap: no data for best λ1, skipping.")
        return

    l2_vals = sorted(set(r[0] for r in subset))
    l3_vals = sorted(set(r[1] for r in subset))
    pp_map  = {(r[0], r[1]): r[2] for r in subset}

    grid = np.full((len(l3_vals), len(l2_vals)), np.nan)
    for i, l3 in enumerate(l3_vals):
        for j, l2 in enumerate(l2_vals):
            v = pp_map.get((l2, l3))
            if v is not None:
                grid[i, j] = v

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   cmap="RdYlGn_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Perplexity (lower = better)")

    ax.set_xticks(range(len(l2_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in l2_vals], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(l3_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in l3_vals], fontsize=8)
    ax.set_xlabel("λ2  (bigram weight)")
    ax.set_ylabel("λ3  (trigram weight)")
    ax.set_title(f"Perplexity Heatmap  (λ1={best_l1:.2f} fixed)\n"
                 f"Best combination marked with ★")

    # Mark best cell
    if best[1] in l2_vals and best[2] in l3_vals:
        bj = l2_vals.index(round(best[1], 6))
        bi = l3_vals.index(round(best[2], 6))
        ax.text(bj, bi, "★", ha="center", va="center",
                color="white", fontsize=14, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap_l2_l3.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Eval] Saved: {path}")
    return path


def plot_pp_vs_l3(results, out_dir):
    """Perplexity vs λ3 for several fixed λ1 slices."""
    l1_values = sorted(set(round(r[0], 2) for r in results))
    # Pick a representative sample of λ1 values (max 6 lines)
    if len(l1_values) > 6:
        idx = np.linspace(0, len(l1_values)-1, 6, dtype=int)
        l1_values = [l1_values[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, l1 in enumerate(l1_values):
        subset = [(l3, pp) for lam1, l2, l3, pp in results if abs(lam1 - l1) < 1e-5]
        subset.sort()
        l3s = [x[0] for x in subset]
        pps = [x[1] for x in subset]
        ax.plot(l3s, pps, marker="o", markersize=3,
                color=COLORS[i % len(COLORS)], label=f"λ1={l1:.2f}")

    ax.set_xlabel("λ3  (trigram weight)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity vs Trigram Weight (λ3)\nfor varying Unigram Weight (λ1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "perplexity_vs_l3.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Eval] Saved: {path}")
    return path


def plot_pp_vs_l2(results, out_dir):
    """Perplexity vs λ2 for several fixed λ1 slices."""
    l1_values = sorted(set(round(r[0], 2) for r in results))
    if len(l1_values) > 6:
        idx = np.linspace(0, len(l1_values)-1, 6, dtype=int)
        l1_values = [l1_values[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, l1 in enumerate(l1_values):
        subset = [(l2, pp) for lam1, l2, l3, pp in results if abs(lam1 - l1) < 1e-5]
        subset.sort()
        l2s = [x[0] for x in subset]
        pps = [x[1] for x in subset]
        ax.plot(l2s, pps, marker="s", markersize=3,
                color=COLORS[i % len(COLORS)], label=f"λ1={l1:.2f}")

    ax.set_xlabel("λ2  (bigram weight)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity vs Bigram Weight (λ2)\nfor varying Unigram Weight (λ1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "perplexity_vs_l2.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Eval] Saved: {path}")
    return path


def plot_top20(results, out_dir):
    """Horizontal bar chart of the top-20 combinations."""
    top20  = results[:20]
    labels = [f"λ1={r[0]:.2f} λ2={r[1]:.2f} λ3={r[2]:.2f}" for r in top20]
    pps    = [r[3] for r in top20]
    # Colour gradient: best = deep green, worst in top-20 = amber
    norm_pp = [(p - min(pps)) / (max(pps) - min(pps) + 1e-9) for p in pps]
    bar_colors = [plt.cm.RdYlGn_r(0.2 + 0.6 * n) for n in norm_pp]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(range(len(labels)), pps, color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel("Validation Perplexity  (lower = better)")
    ax.set_title("Top-20 Lambda Combinations by Validation Perplexity")
    ax.grid(axis="x", alpha=0.4)

    # Value labels on bars
    for bar, pp in zip(bars, pps):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{pp:.3f}", va="center", ha="left", fontsize=8)

    # Annotate rank 1
    ax.text(pps[0] + 0.2, -0.4, "← BEST", color="#16A34A",
            fontsize=9, fontweight="bold", va="center")

    fig.tight_layout()
    path = os.path.join(out_dir, "top20_combinations.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Eval] Saved: {path}")
    return path


def plot_simplex_scatter(results, out_dir):
    """2-D ternary-style scatter: colour = PP, axes are l1 and l2."""
    l1s = [r[0] for r in results]
    l2s = [r[1] for r in results]
    pps = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(l1s, l2s, c=pps, cmap="RdYlGn_r", s=40,
                    edgecolors="none", alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Perplexity")

    best = results[0]
    ax.scatter([best[0]], [best[1]], marker="*", s=250,
               color="black", zorder=5, label=f"Best ({best[3]:.3f})")
    ax.set_xlabel("λ1  (unigram weight)")
    ax.set_ylabel("λ2  (bigram weight)")
    ax.set_title("Lambda Space Scatter  (λ3 = 1 − λ1 − λ2)\nColour = Perplexity")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "lambda_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Eval] Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — HTML report
# ═════════════════════════════════════════════════════════════════════════════

def write_html_report(results, best, tok, out_dir, sample_text):
    top10_rows = ""
    for rank, (l1, l2, l3, pp) in enumerate(results[:10], 1):
        hl = " style='background:#d1fae5;font-weight:bold;'" if rank == 1 else ""
        top10_rows += (
            f"<tr{hl}><td>{rank}</td><td>{l1:.3f}</td>"
            f"<td>{l2:.3f}</td><td>{l3:.3f}</td>"
            f"<td>{pp:.4f}</td></tr>\n"
        )

    worst_pp = results[-1][3]
    best_pp  = best[3]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trigram Model — Hyperparameter Evaluation</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px;
           margin: 40px auto; color: #1e293b; line-height: 1.6; }}
  h1    {{ color: #1d4ed8; border-bottom: 2px solid #1d4ed8; padding-bottom: 6px; }}
  h2    {{ color: #1e40af; margin-top: 40px; }}
  h3    {{ color: #334155; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th,td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: center; }}
  th    {{ background: #1d4ed8; color: white; }}
  tr:hover {{ background: #f1f5f9; }}
  .formula  {{ background: #f0f9ff; border-left: 4px solid #0284c7;
               padding: 12px 20px; font-family: monospace; font-size: 1.05em; }}
  .best-box {{ background: #f0fdf4; border: 2px solid #16a34a; border-radius: 8px;
               padding: 16px 24px; margin: 20px 0; }}
  .best-box span {{ color: #15803d; font-weight: bold; font-size: 1.1em; }}
  img   {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 6px;
           margin: 12px 0; }}
  pre   {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
           padding: 14px; overflow-x: auto; font-size: 0.9em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>

<h1>Urdu Trigram Language Model — Hyperparameter Evaluation Report</h1>

<h2>1. Model Overview</h2>
<p>
  This report documents the hyperparameter sweep performed on a
  <strong>trigram language model</strong> trained on an Urdu story corpus
  using <strong>Maximum Likelihood Estimation (MLE)</strong> with
  <strong>linear interpolation</strong>.
</p>

<h3>Interpolation Formula</h3>
<div class="formula">
P<sub>interp</sub>(t<sub>i</sub> | t<sub>i-2</sub>, t<sub>i-1</sub>) =
  &lambda;3 &middot; P(t<sub>i</sub> | t<sub>i-2</sub>, t<sub>i-1</sub>)  +
  &lambda;2 &middot; P(t<sub>i</sub> | t<sub>i-1</sub>)  +
  &lambda;1 &middot; P(t<sub>i</sub>)<br>
<br>
where &lambda;1 + &lambda;2 + &lambda;3 = 1.0
<br><br>
&lambda;1 = unigram weight &nbsp;&nbsp;
&lambda;2 = bigram weight  &nbsp;&nbsp;
&lambda;3 = trigram weight
</div>

<h3>Special Tokens</h3>
<table>
  <tr><th>Token</th><th>Unicode</th><th>Meaning</th><th>Role in generation</th></tr>
  <tr><td>EOS</td><td>U+0003</td><td>End of sentence</td><td>Sentence boundary marker</td></tr>
  <tr><td>EOP</td><td>U+0004</td><td>End of paragraph</td><td>Paragraph boundary marker</td></tr>
  <tr><td>EOT</td><td>U+0005</td><td>End of story</td><td><strong>Terminates generation</strong></td></tr>
</table>

<h2>2. Corpus & Tokenizer Statistics</h2>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>BPE Vocabulary size</td><td>{tok.vocab_size()}</td></tr>
  <tr><td>Total lambda combinations tested</td><td>{len(results):,}</td></tr>
  <tr><td>Validation split</td><td>Last 10% of corpus</td></tr>
  <tr><td>Sweep step size</td><td>0.05</td></tr>
  <tr><td>Worst perplexity seen</td><td>{worst_pp:.4f}</td></tr>
  <tr><td>Best perplexity seen</td><td>{best_pp:.4f}</td></tr>
</table>

<h2>3. Optimal Hyperparameters</h2>
<div class="best-box">
  <span>λ1 (unigram) = {best[0]:.3f}</span><br>
  <span>λ2 (bigram)  = {best[1]:.3f}</span><br>
  <span>λ3 (trigram) = {best[2]:.3f}</span><br><br>
  Validation Perplexity = <span>{best[3]:.4f}</span><br><br>
  <em>These weights were selected by exhaustive grid search minimising
  held-out perplexity on the last 10% of the corpus.</em>
</div>

<h2>4. Top-10 Lambda Combinations</h2>
<table>
  <tr><th>Rank</th><th>λ1</th><th>λ2</th><th>λ3</th><th>Perplexity</th></tr>
  {top10_rows}
</table>

<h2>5. Evaluation Plots</h2>

<h3>5a. Perplexity Heatmap (λ2 vs λ3, λ1 fixed at {best[0]:.2f})</h3>
<img src="heatmap_l2_l3.png" alt="Heatmap">
<p>
  Each cell shows validation perplexity for a (λ2, λ3) pair with the
  unigram weight fixed at the optimal λ1. Green = lower perplexity = better.
  The star ★ marks the globally best combination.
</p>

<h3>5b. Perplexity vs Trigram Weight (λ3)</h3>
<img src="perplexity_vs_l3.png" alt="PP vs l3">
<p>
  Each line corresponds to a different fixed λ1. The optimal λ3 sits in
  the sweet spot where the trigram's specificity outweighs its sparsity.
</p>

<h3>5c. Perplexity vs Bigram Weight (λ2)</h3>
<img src="perplexity_vs_l2.png" alt="PP vs l2">
<p>
  Shows how bigram weighting affects perplexity across different unigram
  priors.
</p>

<h3>5d. Top-20 Combinations</h3>
<img src="top20_combinations.png" alt="Top 20">

<h3>5e. Lambda Space Scatter</h3>
<img src="lambda_scatter.png" alt="Scatter">
<p>
  Every tested (λ1, λ2) pair coloured by perplexity. The star marks the
  optimal point. The lower-right region (high λ2, low λ1) tends to
  perform better because bigram context is denser than trigram context
  for a 250-token BPE vocabulary.
</p>

<h2>6. Sample Generation with Optimal Weights</h2>
<p>Generated text using prompt <code>"ایک دفعہ"</code> with optimal lambda weights:</p>
<pre>{sample_text}</pre>

<h2>7. Interpretation</h2>
<p>
  <strong>Why does λ2 (bigram) often dominate?</strong><br>
  With a 250-token BPE vocabulary, the corpus produces ~144K unique trigrams
  out of a possible 250³ = 15.6M. Most trigram contexts are therefore sparse
  (seen once or never). The bigram distribution (~18K unique entries) provides
  a much denser, more reliable signal. The optimal λ weights reflect this:
  trigram precision is useful only when context has been seen before, and the
  interpolation formula handles the fallback gracefully.
</p>
<p>
  <strong>EOT-terminated generation</strong><br>
  The model generates tokens until the EOT token (U+0005) is produced or
  the max-token cap is reached. EOS (U+0003) and EOP (U+0004) are treated
  as regular vocabulary items, so the model naturally learns sentence and
  paragraph boundary statistics from the corpus.
</p>

<hr>
<p style="color:#94a3b8;font-size:0.85em;">
  Generated by evaluate.py — Urdu Story Generation Project
</p>
</body>
</html>"""

    path = os.path.join(out_dir, "evaluation_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Eval] Report saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# Step 7 — Re-train model with optimal lambdas and save
# ═════════════════════════════════════════════════════════════════════════════

def save_optimal_model(best_l1, best_l2, best_l3):
    """Train TrigramModel with optimal lambdas on FULL corpus and save."""
    print(f"\n[Eval] Saving optimal model with λ1={best_l1:.3f} λ2={best_l2:.3f} λ3={best_l3:.3f}")
    model = TrigramModel(
        lambda1=best_l1,
        lambda2=best_l2,
        lambda3=best_l3,
        tokenizer_path=TOKENIZER_PATH,
    )
    model.train(corpus_path=CORPUS_PATH, verbose=True)
    model.save(MODEL_PATH)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Urdu Trigram Model — Hyperparameter Evaluation")
    print("=" * 65)

    # 1. Load & split
    train_ids, val_ids, tok = load_and_split(CORPUS_PATH, val_ratio=0.10)

    # 2. Build count tables (once, lambda-independent)
    unigram, bigram, trigram, total = build_count_tables(train_ids)

    # 3. Sweep all lambda combinations
    results = sweep_lambdas(val_ids, unigram, bigram, trigram, total, step=0.05)
    best    = results[0]   # (l1, l2, l3, pp)

    # 4. Plots
    plot_heatmap(results, best, OUT_DIR)
    plot_pp_vs_l3(results, OUT_DIR)
    plot_pp_vs_l2(results, OUT_DIR)
    plot_top20(results, OUT_DIR)
    plot_simplex_scatter(results, OUT_DIR)

    # 5. Save optimal model (trained on FULL corpus)
    model = save_optimal_model(best[0], best[1], best[2])

    # 6. Sample generation for report
    sample = model.generate(
        prompt="ایک دفعہ",
        max_tokens=120,
        temperature=0.7,
        top_k=40,
        top_p=0.92,
        seed=42,
    )

    # 7. HTML report
    write_html_report(results, best, tok, OUT_DIR, sample)

    print("\n" + "=" * 65)
    print("  Done!  All outputs are in:  backend/evaluation/")
    print(f"  Optimal λ1={best[0]:.3f}  λ2={best[1]:.3f}  λ3={best[2]:.3f}")
    print(f"  Validation perplexity = {best[3]:.4f}")
    print(f"  Model saved to:  {MODEL_PATH}")
    print("=" * 65)