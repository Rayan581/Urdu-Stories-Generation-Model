"""
BPE (Byte Pair Encoding) Tokenizer for Urdu Language
Designed for use with a trigram language model with interpolation.

Special tokens:
    EOS = "\u0003"  # End of sentence
    EOP = "\u0004"  # End of paragraph
    EOT = "\u0005"  # End of story
"""

import os
import json
import pickle
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

# ─── Special Unicode Tokens ────────────────────────────────────────────────────
EOS = "\u0003"   # End of sentence
EOP = "\u0004"   # End of paragraph
EOT = "\u0005"   # End of story

SPECIAL_TOKENS = {EOS, EOP, EOT}

# Target vocabulary size
TARGET_VOCAB_SIZE = 250


class BPETokenizer:
    """
    Binary Pair Encoding Tokenizer tailored for Urdu script.

    Attributes
    ----------
    vocab : dict[str, int]
        Maps string token (char or merged pair) → integer ID.
    vocab_inv : dict[int, str]
        Inverse mapping: integer ID → string token.
    merges : list[tuple[str, str]]
        Ordered list of (left, right) merge rules learned during training.
    trained : bool
        Whether the tokenizer has been trained.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.vocab_inv: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.trained: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_special(ch: str) -> bool:
        return ch in SPECIAL_TOKENS

    def _text_to_symbols(self, text: str) -> List[str]:
        """
        Convert raw text into a list of atomic symbols.
        Special tokens (EOS, EOP, EOT) are kept as single indivisible items;
        every other character becomes its own symbol.
        """
        symbols: List[str] = []
        for ch in text:
            symbols.append(ch)  # each character is already a single str
        return symbols

    @staticmethod
    def _get_pair_counts(symbols: List[str]) -> Counter:
        """Count all adjacent pairs, skipping boundaries that involve specials."""
        counts: Counter = Counter()
        for i in range(len(symbols) - 1):
            left, right = symbols[i], symbols[i + 1]
            # Never merge across a special-token boundary
            if left in SPECIAL_TOKENS or right in SPECIAL_TOKENS:
                continue
            counts[(left, right)] += 1
        return counts

    @staticmethod
    def _merge_pair(symbols: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
        """Replace every non-overlapping occurrence of *pair* with *new_token*."""
        left, right = pair
        merged: List[str] = []
        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == left
                and symbols[i + 1] == right
                # Don't merge across special boundaries (already excluded by pair
                # counting, but double-checked here for safety)
                and symbols[i] not in SPECIAL_TOKENS
                and symbols[i + 1] not in SPECIAL_TOKENS
            ):
                merged.append(new_token)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    # ──────────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, corpus_path: str, target_vocab_size: int = TARGET_VOCAB_SIZE, verbose: bool = True):
        """
        Train the BPE tokenizer on *corpus_path*.

        Steps
        -----
        1. Read corpus and split into atomic symbols (chars + special tokens).
        2. Build initial vocabulary from all unique characters found.
        3. Iteratively find the most frequent adjacent pair, assign it a new
           token ID, record the merge rule, and replace all occurrences.
        4. Stop when vocabulary reaches *target_vocab_size*.

        Parameters
        ----------
        corpus_path : str
            Path to the plain-text Urdu corpus (preprocessing/corpus.txt).
        target_vocab_size : int
            Desired final vocabulary size (default 250).
        verbose : bool
            Print progress during training.
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        # ── 1. Load corpus ───────────────────────────────────────────────────
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        if verbose:
            print(f"[BPE] Corpus loaded: {len(text):,} characters")

        # ── 2. Build initial character-level symbols ─────────────────────────
        symbols: List[str] = self._text_to_symbols(text)

        # Discover all unique base characters
        unique_chars = set(symbols)

        # Assign IDs: specials first (lowest IDs), then sorted Urdu chars
        self.vocab = {}
        self.vocab_inv = {}

        # Reserve IDs 0-2 for special tokens so they're always present
        ordered_specials = [EOS, EOP, EOT]
        for idx, sp in enumerate(ordered_specials):
            self.vocab[sp] = idx
            self.vocab_inv[idx] = sp

        # Add remaining base characters in a stable order
        next_id = len(ordered_specials)
        for ch in sorted(unique_chars - set(ordered_specials)):
            self.vocab[ch] = next_id
            self.vocab_inv[next_id] = ch
            next_id += 1

        if verbose:
            print(f"[BPE] Initial vocab size (base chars): {len(self.vocab)}")

        if len(self.vocab) > target_vocab_size:
            raise ValueError(
                f"Initial character vocabulary ({len(self.vocab)}) already exceeds "
                f"target size ({target_vocab_size}). Use a larger target or a smaller corpus."
            )

        # ── 3. BPE merge loop ────────────────────────────────────────────────
        self.merges = []
        iteration = 0

        while len(self.vocab) < target_vocab_size:
            pair_counts = self._get_pair_counts(symbols)

            if not pair_counts:
                if verbose:
                    print("[BPE] No more pairs to merge. Stopping early.")
                break

            # Pick the most frequent pair (ties broken by lexicographic order
            # for determinism)
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            best_count = pair_counts[best_pair]

            if best_count < 2:
                if verbose:
                    print("[BPE] All remaining pairs occur only once. Stopping early.")
                break

            # Create a new merged token string
            new_token = best_pair[0] + best_pair[1]

            # Assign next integer ID
            new_id = next_id
            next_id += 1

            self.vocab[new_token] = new_id
            self.vocab_inv[new_id] = new_token
            self.merges.append(best_pair)

            # Replace all occurrences in the working symbol list
            symbols = self._merge_pair(symbols, best_pair, new_token)

            iteration += 1
            if verbose and iteration % 10 == 0:
                print(
                    f"[BPE] Iteration {iteration:3d} | vocab={len(self.vocab):3d} | "
                    f"merged {best_pair!r} (×{best_count}) → {new_token!r}"
                )

        self.trained = True
        if verbose:
            print(f"[BPE] Training complete. Final vocab size: {len(self.vocab)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Encode / Decode
    # ──────────────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        Encode a raw Urdu string into a list of integer token IDs.

        The merge rules are applied in the same order they were learned during
        training, which guarantees consistency with the trained vocabulary.

        Parameters
        ----------
        text : str
            Input text (may contain EOS / EOP / EOT special tokens).

        Returns
        -------
        list[int]
            Sequence of token IDs.

        Raises
        ------
        RuntimeError
            If called before training.
        """
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained yet. Call train() first.")

        symbols = self._text_to_symbols(text)

        # Apply each merge rule in learned order
        for left, right in self.merges:
            symbols = self._merge_pair(symbols, (left, right), left + right)

        # Convert string tokens → IDs (unknown chars get a fallback)
        ids: List[int] = []
        for sym in symbols:
            if sym in self.vocab:
                ids.append(self.vocab[sym])
            else:
                # Fallback: encode character by character (handles unseen chars)
                for ch in sym:
                    ids.append(self.vocab.get(ch, -1))  # -1 = unknown
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a UTF-8 string.

        Parameters
        ----------
        ids : list[int]
            Sequence of token IDs produced by encode().

        Returns
        -------
        str
            Reconstructed text.

        Raises
        ------
        RuntimeError
            If called before training.
        """
        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained yet. Call train() first.")

        parts = []
        for token_id in ids:
            token_str = self.vocab_inv.get(token_id, "")
            parts.append(token_str)
        return "".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save the trained tokenizer to *path* (JSON format for readability).

        Parameters
        ----------
        path : str
            Destination file path (e.g. "backend/tokenizer.json").
        """
        if not self.trained:
            raise RuntimeError("Cannot save: tokenizer has not been trained.")

        data = {
            "vocab": self.vocab,
            "merges": [[left, right] for left, right in self.merges],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Tokenizer saved to: {path}")

    def load(self, path: str):
        """
        Load a previously saved tokenizer from *path*.

        Parameters
        ----------
        path : str
            Path to the JSON file created by save().
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab_inv = {int(v): k for k, v in self.vocab.items()}
        self.merges = [(left, right) for left, right in data["merges"]]
        self.trained = True
        print(f"[BPE] Tokenizer loaded from: {path}  (vocab size: {len(self.vocab)})")

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience / introspection
    # ──────────────────────────────────────────────────────────────────────────

    def vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return a copy of the vocabulary mapping."""
        return dict(self.vocab)

    def __repr__(self) -> str:
        status = f"trained, vocab_size={len(self.vocab)}" if self.trained else "untrained"
        return f"BPETokenizer({status})"


# ──────────────────────────────────────────────────────────────────────────────
# Entry point: train and quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    CORPUS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "preprocessing", "corpus.txt"
    )
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.json")

    tokenizer = BPETokenizer()

    # ── Train ──────────────────────────────────────────────────────────────
    tokenizer.train(corpus_path=CORPUS_PATH, target_vocab_size=TARGET_VOCAB_SIZE, verbose=True)

    # ── Save ──────────────────────────────────────────────────────────────
    tokenizer.save(SAVE_PATH)

    # ── Smoke test ────────────────────────────────────────────────────────
    sample = f"یہ ایک آزمائشی جملہ ہے{EOS}"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("\n── Smoke Test ──────────────────────────────────────")
    print(f"Original : {sample!r}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded!r}")
    print(f"Match    : {sample == decoded}")
    print(f"Vocab    : {tokenizer.vocab_size()} tokens")