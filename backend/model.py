"""
Trigram Probabilistic Language Model with MLE + Linear Interpolation.

Formula
-------
P_interp(t_i | t_{i-2}, t_{i-1}) =
    λ3 · P(t_i | t_{i-2}, t_{i-1})   ← trigram
  + λ2 · P(t_i | t_{i-1})             ← bigram
  + λ1 · P(t_i)                        ← unigram
  where λ1 + λ2 + λ3 = 1.0

Key design decisions
--------------------
- Candidate-restricted sampling: at generation time only tokens that were
  actually SEEN in the corpus are eligible. This eliminates "floor noise"
  from interpolation giving tiny but non-zero mass to unseen tokens.
- Context-aware candidate set: prefer trigram successors → bigram → unigram.
- Top-k / nucleus (top-p) filtering for coherent generation.
- Lambda auto-tuning via held-out perplexity grid search.

Special tokens (from tokenizer):
    EOS  \\u0003  End of sentence
    EOP  \\u0004  End of paragraph
    EOT  \\u0005  End of story  ← generation stops here
"""

import os
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tokenizer import BPETokenizer, EOS, EOP, EOT

_BOS_ID = -1   # Virtual beginning-of-sequence sentinel


# ─────────────────────────────────────────────────────────────────────────────
# defaultdict factories (module-level so they survive JSON round-trips)
# ─────────────────────────────────────────────────────────────────────────────

def _int_dd():
    return defaultdict(int)

def _dd_int_dd():
    return defaultdict(_int_dd)


# ─────────────────────────────────────────────────────────────────────────────
class TrigramModel:
    """
    From-scratch trigram language model with linear interpolation.

    Parameters
    ----------
    lambda1 : float   Unigram weight  (default 0.1)
    lambda2 : float   Bigram weight   (default 0.4)
    lambda3 : float   Trigram weight  (default 0.5)
    tokenizer_path : str
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.4,
        lambda3: float = 0.5,
        tokenizer_path: str = "backend/tokenizer.json",
    ):
        if abs(lambda1 + lambda2 + lambda3 - 1.0) > 1e-6:
            raise ValueError(
                f"Interpolation weights must sum to 1.0, got {lambda1+lambda2+lambda3:.4f}"
            )
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Count tables
        self.unigram_counts: Dict[int, int] = defaultdict(int)
        self.bigram_counts:  Dict[int, Dict[int, int]] = defaultdict(_int_dd)
        self.trigram_counts: Dict[int, Dict[int, Dict[int, int]]] = defaultdict(_dd_int_dd)

        self.total_tokens: int = 0
        self.vocab_size:   int = 0
        self.trained:      bool = False

        # Tokens actually seen during training (used to restrict generation)
        self._seen_tokens: List[int] = []

        # Load tokenizer
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size()
        self._eot_id = self.tokenizer.vocab[EOT]
        self._eos_id = self.tokenizer.vocab[EOS]
        self._eop_id = self.tokenizer.vocab[EOP]

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, corpus_path: str, verbose: bool = True):
        """
        Build count tables from *corpus_path*.

        Two _BOS_ID sentinels are prepended so every token has a full
        trigram context from position 0.
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        if verbose:
            print("[Trigram] Encoding corpus ...")

        token_ids = self.tokenizer.encode(text)
        # Drop any -1 unknowns
        token_ids = [t for t in token_ids if t >= 0]

        if verbose:
            print(f"[Trigram] Corpus length: {len(token_ids):,} tokens")

        padded = [_BOS_ID, _BOS_ID] + token_ids

        for i in range(2, len(padded)):
            t0, t1, t2 = padded[i-2], padded[i-1], padded[i]

            if t2 != _BOS_ID:
                self.unigram_counts[t2] += 1
                self.total_tokens += 1

            self.bigram_counts[t1][t2] += 1
            self.trigram_counts[t0][t1][t2] += 1

        self._seen_tokens = sorted(self.unigram_counts.keys())
        self.trained = True

        if verbose:
            print(
                f"[Trigram] Training complete.\n"
                f"          Unique unigrams : {len(self.unigram_counts):,}\n"
                f"          Unique bigrams  : {sum(len(v) for v in self.bigram_counts.values()):,}\n"
                f"          Unique trigrams : "
                f"{sum(len(vv) for v in self.trigram_counts.values() for vv in v.values()):,}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Probability
    # ─────────────────────────────────────────────────────────────────────────

    def _p_unigram(self, t: int) -> float:
        if self.total_tokens == 0:
            return 1.0 / max(self.vocab_size, 1)
        return self.unigram_counts.get(t, 0) / self.total_tokens

    def _p_bigram(self, t_prev: int, t: int) -> float:
        ctx = self.bigram_counts.get(t_prev)
        if not ctx:
            return self._p_unigram(t)
        return ctx.get(t, 0) / sum(ctx.values())

    def _p_trigram(self, t_2: int, t_1: int, t: int) -> float:
        ctx2 = self.trigram_counts.get(t_2)
        if not ctx2:
            return self._p_bigram(t_1, t)
        ctx1 = ctx2.get(t_1)
        if not ctx1:
            return self._p_bigram(t_1, t)
        return ctx1.get(t, 0) / sum(ctx1.values())

    def probability(self, t: int, t_1: int, t_2: int) -> float:
        """
        Interpolated probability:
            P = lambda3*P3(t|t-2,t-1) + lambda2*P2(t|t-1) + lambda1*P1(t)
        """
        return (
            self.lambda3 * self._p_trigram(t_2, t_1, t)
            + self.lambda2 * self._p_bigram(t_1, t)
            + self.lambda1 * self._p_unigram(t)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Perplexity
    # ─────────────────────────────────────────────────────────────────────────

    def perplexity(self, token_ids: List[int]) -> float:
        """Compute perplexity of *token_ids* under the interpolated model."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        token_ids = [t for t in token_ids if t >= 0]
        if not token_ids:
            return float("inf")

        padded = [_BOS_ID, _BOS_ID] + token_ids
        log_sum = 0.0
        for i in range(2, len(padded)):
            p = self.probability(padded[i], padded[i-1], padded[i-2])
            log_sum += math.log(max(p, 1e-10))

        return math.exp(-log_sum / len(token_ids))

    # ─────────────────────────────────────────────────────────────────────────
    # Context-aware candidate set
    # ─────────────────────────────────────────────────────────────────────────

    def _candidate_tokens(self, t_2: int, t_1: int) -> List[int]:
        """
        Return candidate next tokens for context (t_2, t_1).

        Priority order:
          1. Tokens seen in trigram context (t_2, t_1) — most specific
          2. Tokens seen in bigram context (t_1) — fallback
          3. All tokens seen in training corpus — last resort

        This keeps generation within distributions the model actually learned,
        eliminating the interpolation floor noise on zero-count tokens.
        """
        tri_ctx = self.trigram_counts.get(t_2, {}).get(t_1)
        if tri_ctx:
            return list(tri_ctx.keys())

        bi_ctx = self.bigram_counts.get(t_1)
        if bi_ctx:
            return list(bi_ctx.keys())

        return self._seen_tokens

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_sample(tokens: List[int], probs: List[float]) -> int:
        """Pure-Python weighted random draw."""
        r = random.random()
        cumulative = 0.0
        for t, p in zip(tokens, probs):
            cumulative += p
            if r <= cumulative:
                return t
        return tokens[-1]

    def _build_distribution(
        self,
        candidates: List[int],
        t_2: int,
        t_1: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[List[int], List[float]]:
        """
        Score candidates with interpolated probabilities, then apply
        temperature, top-k, and nucleus (top-p) filtering.
        """
        scored = [(t, self.probability(t, t_1, t_2)) for t in candidates]

        # Temperature scaling — lower T sharpens the distribution
        if temperature != 1.0:
            scored = [(t, p ** (1.0 / temperature)) for t, p in scored]

        # Sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Top-k filter
        if top_k > 0:
            scored = scored[:top_k]

        # Nucleus (top-p) filter
        if 0.0 < top_p < 1.0:
            total = sum(p for _, p in scored)
            if total > 0:
                normed = [(t, p / total) for t, p in scored]
                cumulative, kept = 0.0, []
                for t, p in normed:
                    kept.append((t, p))
                    cumulative += p
                    if cumulative >= top_p:
                        break
                scored = kept

        # Final normalisation
        total = sum(p for _, p in scored)
        if total == 0 or not scored:
            n = max(len(scored), 1)
            tokens = [t for t, _ in scored] if scored else self._seen_tokens[:1]
            probs  = [1.0 / len(tokens)] * len(tokens)
        else:
            tokens = [t for t, _ in scored]
            probs  = [p / total for _, p in scored]

        return tokens, probs

    # ─────────────────────────────────────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str   = "",
        max_tokens:  int   = 200,
        temperature: float = 0.7,
        top_k:       int   = 40,
        top_p:       float = 0.92,
        seed:        Optional[int] = None,
    ) -> str:
        """
        Generate Urdu text from *prompt* until EOT or *max_tokens*.

        Parameters
        ----------
        prompt      : str    Urdu seed text.  May be empty.
        max_tokens  : int    Hard token cap.
        temperature : float  Sampling temperature (0.5–0.8 recommended).
        top_k       : int    Keep only top-k candidates per step (0=disabled).
        top_p       : float  Nucleus threshold (0–1, 0=disabled).
        seed        : int    RNG seed for reproducibility.

        Returns
        -------
        str  Decoded text (prompt + generated continuation).
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        if seed is not None:
            random.seed(seed)

        prompt_ids = [t for t in self.tokenizer.encode(prompt) if t >= 0] if prompt else []

        if len(prompt_ids) >= 2:
            t_2, t_1 = prompt_ids[-2], prompt_ids[-1]
        elif len(prompt_ids) == 1:
            t_2, t_1 = _BOS_ID, prompt_ids[-1]
        else:
            t_2, t_1 = _BOS_ID, _BOS_ID

        generated = list(prompt_ids)

        for _ in range(max_tokens):
            candidates = self._candidate_tokens(t_2, t_1)
            tokens, probs = self._build_distribution(
                candidates, t_2, t_1, temperature, top_k, top_p
            )
            next_id = self._weighted_sample(tokens, probs)
            generated.append(next_id)

            if next_id == self._eot_id:
                break

            t_2, t_1 = t_1, next_id

        return self.tokenizer.decode(generated)

    # ─────────────────────────────────────────────────────────────────────────
    # Lambda auto-tuning
    # ─────────────────────────────────────────────────────────────────────────

    def tune_lambdas(
        self,
        validation_ids: List[int],
        step: float = 0.1,
        verbose: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Grid-search (lambda1, lambda2, lambda3) combinations to minimise
        validation perplexity.  Updates self.lambda* in place.

        Parameters
        ----------
        validation_ids : list[int]   Pre-encoded held-out token sequence.
        step : float                 Grid resolution (0.1 = 10-step grid).
        verbose : bool

        Returns
        -------
        (best_lambda1, best_lambda2, best_lambda3)
        """
        best_pp  = float("inf")
        best_lam = (self.lambda1, self.lambda2, self.lambda3)
        steps    = [round(i * step, 10) for i in range(1, int(1.0 / step))]
        combos   = 0

        for l1 in steps:
            for l2 in steps:
                l3 = round(1.0 - l1 - l2, 10)
                if l3 <= 0 or l3 > 1.0:
                    continue
                combos += 1
                self.lambda1, self.lambda2, self.lambda3 = l1, l2, l3
                pp = self.perplexity(validation_ids)
                if pp < best_pp:
                    best_pp  = pp
                    best_lam = (l1, l2, l3)

        self.lambda1, self.lambda2, self.lambda3 = best_lam
        if verbose:
            print(
                f"[Trigram] Lambda tuning done ({combos} combos).\n"
                f"          Best lambda1={best_lam[0]:.2f}  "
                f"lambda2={best_lam[1]:.2f}  "
                f"lambda3={best_lam[2]:.2f}  "
                f"Perplexity={best_pp:.4f}"
            )
        return best_lam

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save count tables and hyperparameters to JSON."""
        if not self.trained:
            raise RuntimeError("Cannot save: model not trained.")

        data = {
            "lambda1":        self.lambda1,
            "lambda2":        self.lambda2,
            "lambda3":        self.lambda3,
            "total_tokens":   self.total_tokens,
            "vocab_size":     self.vocab_size,
            "seen_tokens":    self._seen_tokens,
            "unigram_counts": {str(k): v for k, v in self.unigram_counts.items()},
            "bigram_counts":  {
                str(k): {str(k2): v2 for k2, v2 in inner.items()}
                for k, inner in self.bigram_counts.items()
            },
            "trigram_counts": {
                str(k): {
                    str(k2): {str(k3): v3 for k3, v3 in inner2.items()}
                    for k2, inner2 in inner.items()
                }
                for k, inner in self.trigram_counts.items()
            },
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[Trigram] Model saved to: {path}")

    def load(self, path: str):
        """Restore model from JSON file saved by save()."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lambda1      = data["lambda1"]
        self.lambda2      = data["lambda2"]
        self.lambda3      = data["lambda3"]
        self.total_tokens = data["total_tokens"]
        self.vocab_size   = data["vocab_size"]
        self._seen_tokens = [int(t) for t in data.get("seen_tokens", [])]

        self.unigram_counts = defaultdict(int,
            {int(k): v for k, v in data["unigram_counts"].items()})

        self.bigram_counts = defaultdict(_int_dd)
        for k, inner in data["bigram_counts"].items():
            self.bigram_counts[int(k)] = defaultdict(int,
                {int(k2): v2 for k2, v2 in inner.items()})

        self.trigram_counts = defaultdict(_dd_int_dd)
        for k, inner in data["trigram_counts"].items():
            self.trigram_counts[int(k)] = defaultdict(_int_dd)
            for k2, inner2 in inner.items():
                self.trigram_counts[int(k)][int(k2)] = defaultdict(int,
                    {int(k3): v3 for k3, v3 in inner2.items()})

        self.trained = True
        print(f"[Trigram] Model loaded from: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────────────

    def top_k_next(self, prompt: str, k: int = 10) -> List[Tuple[str, float]]:
        """Top-k most probable next tokens after *prompt* (for inspection)."""
        if not self.trained:
            raise RuntimeError("Model not trained.")

        ids = [t for t in self.tokenizer.encode(prompt) if t >= 0] if prompt else []
        t_2 = ids[-2] if len(ids) >= 2 else _BOS_ID
        t_1 = ids[-1] if len(ids) >= 1 else _BOS_ID

        scored = [(t, self.probability(t, t_1, t_2)) for t in self._seen_tokens]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            (self.tokenizer.vocab_inv.get(t, f"<{t}>"), p)
            for t, p in scored[:k]
        ]

    def __repr__(self) -> str:
        status = "trained" if self.trained else "untrained"
        return (
            f"TrigramModel({status}, "
            f"lambda=({self.lambda1},{self.lambda2},{self.lambda3}), "
            f"vocab={self.vocab_size})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick smoke-test only.
    For full hyperparameter tuning and optimal model selection run:
        python backend/evaluate.py

    evaluate.py will:
      - sweep all (λ1, λ2, λ3) combinations on a validation split
      - produce plots and an HTML report in backend/evaluation/
      - retrain with optimal lambdas and save backend/model.json
    """
    BASE           = os.path.dirname(__file__)
    CORPUS_PATH    = os.path.join(BASE, "..", "preprocessing", "corpus.txt")
    TOKENIZER_PATH = os.path.join(BASE, "tokenizer.json")
    MODEL_PATH     = os.path.join(BASE, "model.json")

    # ── Check if a saved model already exists ─────────────────────────────
    if os.path.exists(MODEL_PATH):
        print(f"[Trigram] Loading existing model from {MODEL_PATH}")
        model = TrigramModel(tokenizer_path=TOKENIZER_PATH)
        model.load(MODEL_PATH)
    else:
        print("[Trigram] No saved model found. Training with default lambdas.")
        print("[Trigram] Run evaluate.py for optimal lambda selection.\n")
        model = TrigramModel(
            lambda1=0.1,
            lambda2=0.4,
            lambda3=0.5,
            tokenizer_path=TOKENIZER_PATH,
        )
        model.train(corpus_path=CORPUS_PATH, verbose=True)
        model.save(MODEL_PATH)

    # ── Generation smoke-test ─────────────────────────────────────────────
    print(f"\n[Trigram] Model: {model}")
    print("\n── Generation Test ──────────────────────────────────────")
    out = model.generate(
        prompt="ایک دفعہ",
        max_tokens=150,
        temperature=0.7,
        top_k=40,
        top_p=0.92,
        seed=42,
    )
    print(out)

    # ── Perplexity ────────────────────────────────────────────────────────
    sample_ids = [t for t in model.tokenizer.encode("یہ ایک آزمائشی جملہ ہے") if t >= 0]
    print(f"\nSample perplexity: {model.perplexity(sample_ids):.4f}")

    # ── Top-5 next tokens ─────────────────────────────────────────────────
    print("\nTop-5 next tokens after 'ایک دفعہ':")
    for tok, prob in model.top_k_next("ایک دفعہ", k=5):
        print(f"  {tok!r:25s}  {prob:.6f}")