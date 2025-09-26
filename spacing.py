import re
import pandas as pd
import math
from typing import Callable, List, Tuple


# Load word costs
def load_word_costs(csv_file: str = "unigram_freq.csv") -> Tuple[Callable[[str], float], int]:
    """
    Load unigram counts from CSV and return:
      - word_cost(word) -> negative-log-probability (cost)
      - maxword: maximum word length in the vocabulary
    """

    # Unigram_freq.csv only has top 30000 most frequent words to balance speed and accuracy
    df = pd.read_csv(csv_file)

    # Ensure strings and ints
    words = df['word'].astype(str).tolist()
    counts = df['count'].astype(float).tolist()

    # total occurrences (denominator)
    N = float(sum(counts))
    if N <= 0:
        raise ValueError("Total counts (N) must be > 0")

    logN = math.log(N)
    LOG10 = math.log(10.0)

    # Precompute negative-log-prob (cost) for every known word:
    # cost(word) = -log(count/N) = logN - log(count)
    cost_map = {}
    maxword = 0
    for w, c in zip(words, counts):
        if c <= 0:
            continue
        cost_map[w] = logN - math.log(c)
        if len(w) > maxword:
            maxword = len(w)

    # Missing-word cost formula derived from avoid_long_words:
    # avoid_long_words(key,N) = 10 / (N * 10^len(key))
    # -log(p) = log(N) + (len(key)-1) * log(10)
    def missing_cost(key: str) -> float:
        return logN + (len(key) - 1) * LOG10

    def word_cost(key: str) -> float:
        # key assumed already lowercase (or same case as keys in CSV)
        return cost_map.get(key, missing_cost(key))

    # If vocabulary empty, set a reasonable maxword
    if maxword == 0:
        maxword = 20

    return word_cost, maxword


# Dynamic programming segmenter
def infer_spaces_dp(s: str, word_cost: Callable[[str], float], maxword: int) -> List[str]:
    """
    Segment string s (no punctuation) into words that minimize total cost.
    Uses dynamic programming: O(len(s) * maxword).
    Returns list of words (lowercase, since we look up lowercase).
    """

    n = len(s)
    # cost[i] = best (minimum) cost for s[:i]
    cost = [0.0] + [float('inf')] * n
    # backpointer: length of last word for best segmentation ending at i
    back = [0] * (n + 1)

    # localize for speed
    wc = word_cost
    M = min(maxword, n)

    for i in range(1, n + 1):
        best_cost = float('inf')
        best_k = 1
        start = max(0, i - M)
        # Try words s[j:i] where j from start..i-1
        # iterate j forward is fine; either forward or backward works

        for j in range(start, i):
            w = s[j:i]
            c = cost[j] + wc(w)
            if c < best_cost:
                best_cost = c
                best_k = i - j
        cost[i] = best_cost
        back[i] = best_k

    # Reconstruct segmentation
    out = []
    i = n
    while i > 0:
        k = back[i]
        out.append(s[i - k:i])
        i -= k
    out.reverse()
    return out


# Wrapper that preserves punctuation and spacing.
def smart_segment(text: str, word_cost: Callable[[str], float], maxword: int) -> str:
    """
    Split text on non-alphabetic runs, segment alphabetic runs with DP,
    and keep punctuation/whitespace untouched. Adds a space after
    ending punctuation if missing.
    """

    # Keep delimiters (punctuation/whitespace/numbers)
    tokens = re.split(r'([^A-Za-z]+)', text)
    out_parts = []

    for tok in tokens:
        if tok.isalpha():
            # Segment alphabetic token in lowercase
            seg = infer_spaces_dp(tok.lower(), word_cost, maxword)
            out_parts.append(" ".join(seg))
        else:
            # Ensure a space follows ending punctuation
            if tok and tok[-1] in ".!?,":
                if not tok.endswith(" "):
                    tok += " "
            out_parts.append(tok)

    return "".join(out_parts)


