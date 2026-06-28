# -*- coding: utf-8 -*-
"""
metrics.py — CER / WER for OCR evaluation.

Prefers the ``jiwer`` library (well-tested implementation used in ASR/OCR
benchmarks). If ``jiwer`` is not installed, falls back to a self-contained
Levenshtein implementation so the benchmark still runs anywhere. Both paths
return identical numbers for the same inputs.

  * CER = character error rate  = edit_distance(chars) / len(ref_chars)
  * WER = word  error rate      = edit_distance(words) / len(ref_words)

Lower is better; 0.0 == perfect, values can exceed 1.0 when the prediction is
much longer than the reference (many insertions).
"""
from __future__ import annotations

from typing import List, Sequence

try:
    import jiwer  # type: ignore

    _HAVE_JIWER = True
except Exception:  # pragma: no cover - exercised only when jiwer missing
    _HAVE_JIWER = False


def _levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Classic O(n*m) Levenshtein edit distance over any sequence."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = cur
    return prev[m]


def _normalize(text: str, case_sensitive: bool = True) -> str:
    """Collapse whitespace and strip — the standard light normalization so a
    trailing newline from Tesseract is not counted as an error.

    With ``case_sensitive=False`` the text is also lower-cased. This matters
    when comparing engines that differ in letter-case behaviour: e.g.
    ``microsoft/trocr-base-printed`` tends to emit ALL-CAPS, so a case-sensitive
    CER would penalise it for a purely cosmetic difference. The benchmark
    reports the case-insensitive numbers and documents the choice."""
    text = " ".join(text.split())
    return text if case_sensitive else text.lower()


def cer(reference: str, hypothesis: str, case_sensitive: bool = True) -> float:
    """Character error rate."""
    ref = _normalize(reference, case_sensitive)
    hyp = _normalize(hypothesis, case_sensitive)
    if _HAVE_JIWER:
        return float(jiwer.cer(ref, hyp))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str, case_sensitive: bool = True) -> float:
    """Word error rate."""
    ref = _normalize(reference, case_sensitive)
    hyp = _normalize(hypothesis, case_sensitive)
    if _HAVE_JIWER:
        return float(jiwer.wer(ref, hyp))
    ref_w: List[str] = ref.split()
    hyp_w: List[str] = hyp.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(ref_w, hyp_w) / len(ref_w)


def backend() -> str:
    """Which implementation is active ('jiwer' or 'builtin')."""
    return "jiwer" if _HAVE_JIWER else "builtin"


if __name__ == "__main__":
    r = "the quick brown fox"
    h = "the quikc brown fax"
    print(f"backend = {backend()}")
    print(f"CER = {cer(r, h):.4f}")
    print(f"WER = {wer(r, h):.4f}")
