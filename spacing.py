import re
import pandas as pd
import math
from typing import Dict, Union, cast
from functools import lru_cache

TrieNode = Dict[str, Union["TrieNode", float]]
END_MARK = "_end_"
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def build_trie(word_cost_map: Dict[str, float]) -> TrieNode:
    """
    Construct a trie (prefix tree) from a mapping of words to cost values.

    Each word is inserted one character at a time into a nested dictionary
    structure. The final character node of each word stores its associated
    cost under the `END_MARK` key.

    :param word_cost_map: A mapping where each key is a word and each value
                          is the float cost assigned to that word.
    :type word_cost_map: Dict[str, float]

    :return: A nested dictionary representing the trie. Each character maps
             to another node, and terminal nodes contain an `END_MARK` key
             storing the word’s cost.
    :rtype: TrieNode
    """
    trie: TrieNode = {}
    for word, cost in word_cost_map.items():
        node = trie
        for ch in word:
            nxt = node.get(ch)
            if not isinstance(nxt, dict):
                nxt = {}
                node[ch] = nxt
            node = nxt
        node[END_MARK] = float(cost)
    return trie

@lru_cache(maxsize=1)
def cached_trie(csv_file: str = "unigram_freq.csv") -> TrieNode:
    """
Load unigram frequency data from a CSV file, convert word frequencies
    into negative log-probability costs, build a trie from those costs, and
    cache the result for reuse.

    The CSV file must contain at least the following columns:
      - "word":  the word string
      - "count": the frequency or occurrence count

    After building the trie, the result is stored in an LRU cache with
    a maximum size of 1, preventing repeated recomputation.

    :param csv_file: Path to the CSV file containing "word" and "count"
                     columns. Defaults to "unigram_freq.csv".
    :type csv_file: str, optional

    :return: A trie where each path corresponds to a word and each terminal
             node stores the associated cost under the `END_MARK` key.
    :rtype: TrieNode
    """
    df = pd.read_csv(csv_file)
    words = df["word"].astype(str).str.lower().tolist()
    counts = df["count"].astype(float).tolist()
    total = float(sum(counts)) or 1.0
    costs = {w: -math.log(max(c / total, 1e-12)) for w, c in zip(words, counts)}
    return build_trie(costs)

def infer_spaces_trie(
    s: str,
    trie_root: TrieNode,
    unknown_char_cost: float = 12.0,
):
    """
Segment a continuous string by inferring word boundaries using a trie-based
    dynamic programming approach.

    Behavior:
        • Converts the input string to lowercase and processes it character by character.
        • Walks the trie from each index to find valid words and retrieves their costs
          using the `END_MARK` key.
        • Uses a DP table to accumulate the minimum segmentation cost at each position.
        • Applies an `unknown_char_cost` penalty for characters not matched in the trie,
          ensuring progression even for unrecognized substrings.
        • Reconstructs the lowest-cost segmentation by backtracking from the end of the string.

    :param s: The alphabetic string to segment.
    :type s: str

    :param trie_root: The root of the trie containing words and associated costs.
    :type trie_root: TrieNode

    :param unknown_char_cost: The penalty for characters or sequences not found in the trie.
                              Defaults to 12.0.
    :type unknown_char_cost: float, optional

    :return: A tuple containing:
             • The segmented version of the string (with spaces),
             • The total cost of that segmentation.
    :rtype: tuple[str, float]
    """
    if not s:
        return s, 0.0

    lower = s.lower()
    n = len(lower)

    dp_cost = [float("inf")] * (n + 1)
    back = [-1] * (n + 1)
    dp_cost[0] = 0.0

    for i in range(n):
        base = dp_cost[i]
        if base == float("inf"):
            continue

        # Walk the trie forward from i
        node: TrieNode = trie_root
        j = i
        while j < n:
            nxt = node.get(lower[j])
            if not isinstance(nxt, dict):
                break
            node = nxt
            j += 1

            # If this node ends a word, consider cutting here
            if END_MARK in node:
                word_cost = cast(float, node[END_MARK])
                new_cost = base + word_cost
                if new_cost < dp_cost[j]:
                    dp_cost[j] = new_cost
                    back[j] = i

        # Unknown fall-through (advance by one char with penalty)
        unk_cost = base + unknown_char_cost
        if unk_cost < dp_cost[i + 1]:
            dp_cost[i + 1] = unk_cost
            back[i + 1] = i

    # Reconstruct
    out = []
    j = n
    while j > 0:
        i = back[j]
        if i < 0:
            return lower, dp_cost[n]  # safety
        out.append(lower[i:j])
        j = i

    out.reverse()
    return " ".join(out), dp_cost[n]

def smart_segment(text: str) -> str:
    """
Segment alphabetic portions of a string using a trie-based word-break model
    while preserving all non-alphabetic content.

    Behavior:
        • Splits the text into alphabetic and non-alphabetic tokens.
        • Sends alphabetic tokens to ``infer_spaces_trie`` for segmentation.
        • Normalizes curly apostrophes (’ → ').
        • Fixes contraction splits such as:
            - "is n't"  → "isn't"
            - "I 'm"    → "I'm"
            - "haven'tread" → "haven't read"
        • Removes extra spaces around apostrophes.
        • Adds a trailing space to punctuation marks (.,!?,) when missing.
        • Preserves HTML, numbers, symbols, and spacing exactly (except for
          punctuation spacing adjustments).

    :param text: The full input string to segment.
    :type text: str

    :return: The segmented string with corrected contractions and preserved
             formatting.
    :rtype: str
    """

    # Keep delimiters (punctuation/whitespace/numbers)
    tokens = re.split(r"([^A-Za-z'’]+)", text)
    out_parts = []

    for tok in tokens:
        if WORD_RE.fullmatch(tok):

            seg, _ = infer_spaces_trie(tok, cached_trie())

            # Normalize curly apostrophes first
            seg = seg.replace("’", "'")

            #Merge split "n ' t" into "n't", then merges base + space + n't (Specifically handles isn't)
            seg = re.sub(r"(?i)\bn\s*'\s*t\b", "n't", seg)
            seg = re.sub(r"(?i)\b([A-Za-z]+)\s+n'\s*t\b", r"\1n't", seg)

            # Maintains no spaces around apostrophes
            seg = re.sub(r"\s*(['’])\s*", r"\1", seg)

            # Merges contractions split by the model
            seg = re.sub(r"(?i)\b([A-Za-z]+)\s*(['’](?:t|s|re|ve|ll|d|m))\b", r"\1\2", seg)

            # 3) Inserts a space if a contraction is immediately followed by another letter
            seg = re.sub(r"(?i)(['’](?:t|s|re|ve|ll|d|m))(?=[A-Za-z])", r"\1 ", seg)

            out_parts.append(seg)
        else:
            # Ensure a space follows ending punctuation
            if tok and tok[-1] in ".!?,":
                if not tok.endswith(" "):
                    tok += " "
            out_parts.append(tok)

    return "".join(out_parts)

sampletext = ("quitegood,don'texpectanythinghighculture.......theactingisbad,thestorylinefails,butitisstillafairlynicemovietowatch.why?becauseit'sdark,alittlebitstupid,likeunpredictableandjustentertainingandfuntowatch.donotexpectanything,likeisaid,justseeitforyourselfandyouknowwhatimean.<br/><br/>itisamovie,withoutaplotormemorableacting,butthereareenoughscenesthatwillmakeyoulaugh,cryoratleastmakeyoufeelcompelledtowatchittotheend...<br/><br/>thisisalliwantedtosay....<br/><br/>7/10")
print (smart_segment(sampletext))



