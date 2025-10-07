import re
import math
from typing import Dict, Union, cast
from functools import lru_cache
from unigram_freq import UNIGRAM_FREQ_MAP

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
                          is the word frequency.
    :type word_cost_map: Dict[str, float]

    :return: A nested dictionary representing the trie. Each character maps
             to another node, and terminal nodes contain an `END_MARK` key
             storing the word’s cost.
    :rtype: TrieNode
    """
    trie_root: TrieNode = {}
    for word, cost in word_cost_map.items():
        current_node = trie_root
        for char in word:
            current_node = current_node.setdefault(char, {})
        current_node[END_MARK] = float(cost)
    return trie_root


@lru_cache(maxsize=1)
def _cached_trie() -> TrieNode:
    """
    Load unigram frequency data from the Python dictionary UNIGRAM_FREQ_MAP,
    convert word frequencies into negative log-probability costs, build a
    trie from those costs, and cache the result for reuse.

    :return: A trie where each path corresponds to a word and each terminal
             node stores the associated cost under the `END_MARK` key.
    :rtype: TrieNode
    """
    total = float(sum(UNIGRAM_FREQ_MAP.values())) or 1.0
    inv_total = 1.0 / total  # precompute for efficiency

    # Compute costs directly from dictionary items
    word_costs = {w: -math.log(max(c * inv_total, 1e-12)) for w, c in UNIGRAM_FREQ_MAP.items()}

    return build_trie(word_costs)


import re
import math
from typing import Dict, Union, cast
from functools import lru_cache
from unigram_freq import UNIGRAM_FREQ_MAP

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
    trie_root: TrieNode = {}
    for word, cost in word_cost_map.items():
        current_node = trie_root
        for char in word:
            current_node = current_node.setdefault(char, {})
        current_node[END_MARK] = float(cost)
    return trie_root


@lru_cache(maxsize=1)
def _cached_trie() -> TrieNode:
    """
    Load unigram frequency data from the Python dictionary UNIGRAM_FREQ,
    convert word frequencies into negative log-probability costs, build a
    trie from those costs, and cache the result for reuse.

    :return: A trie where each path corresponds to a word and each terminal
             node stores the associated cost under the `END_MARK` key.
    :rtype: TrieNode
    """
    total_count = float(sum(UNIGRAM_FREQ_MAP.values())) or 1.0
    inv_total = 1.0 / total_count  # precompute for efficiency

    # Compute costs directly from dictionary items
    word_costs = {word: -math.log(max(count * inv_total, 1e-12))
                  for word, count in UNIGRAM_FREQ_MAP.items()}

    return build_trie(word_costs)


def infer_spaces_trie(
    text: str,
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

    :param text: The alphabetic string to segment.
    :type text: str

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
    if not text:
        return text, 0.0

    lowercase_text = text.lower()
    text_length = len(lowercase_text)

    min_cost = [float("inf")] * (text_length + 1)
    backtrack_index = [-1] * (text_length + 1)
    min_cost[0] = 0.0

    for start_index in range(text_length):
        current_cost = min_cost[start_index]
        if current_cost == float("inf"):
            continue

        # Walk the trie forward from start_index
        current_node: TrieNode = trie_root
        end_index = start_index
        while end_index < text_length:
            next_node = current_node.get(lowercase_text[end_index])
            if not isinstance(next_node, dict):
                break
            current_node = next_node
            end_index += 1

            # If this node ends a word, consider cutting here
            if END_MARK in current_node:
                word_cost = cast(float, current_node[END_MARK])
                new_cost = current_cost + word_cost
                if new_cost < min_cost[end_index]:
                    min_cost[end_index] = new_cost
                    backtrack_index[end_index] = start_index

        # Unknown fall-through (advance by one char with penalty)
        unknown_cost_total = current_cost + unknown_char_cost
        if unknown_cost_total < min_cost[start_index + 1]:
            min_cost[start_index + 1] = unknown_cost_total
            backtrack_index[start_index + 1] = start_index

    # Reconstruct the segmented string
    segmented_words = []
    current_index = text_length
    while current_index > 0:
        prev_index = backtrack_index[current_index]
        if prev_index < 0:
            return lowercase_text, min_cost[text_length]  # safety
        segmented_words.append(lowercase_text[prev_index:current_index])
        current_index = prev_index

    segmented_words.reverse()
    return " ".join(segmented_words), min_cost[text_length]


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
    segmented_parts = []

    for token in tokens:
        if WORD_RE.fullmatch(token):

            segmented_text, _ = infer_spaces_trie(token, _cached_trie())

            # Normalize curly apostrophes
            segmented_text = segmented_text.replace("’", "'")

            # Merge split "n ' t" into "n't", then merges base + space + n't
            segmented_text = re.sub(r"(?i)\bn\s*'\s*t\b", "n't", segmented_text)
            segmented_text = re.sub(r"(?i)\b([A-Za-z]+)\s+n'\s*t\b", r"\1n't", segmented_text)

            # Maintains no spaces around apostrophes
            segmented_text = re.sub(r"\s*(['’])\s*", r"\1", segmented_text)

            # Merges contractions split by the model
            segmented_text = re.sub(
                r"(?i)\b([A-Za-z]+)\s*(['’](?:t|s|re|ve|ll|d|m))\b", r"\1\2", segmented_text
            )

            # Inserts a space if a contraction is immediately followed by another letter
            segmented_text = re.sub(r"(?i)(['’](?:t|s|re|ve|ll|d|m))(?=[A-Za-z])", r"\1 ", segmented_text)

            segmented_parts.append(segmented_text)
        else:
            # Ensure a space follows ending punctuation
            if token and token[-1] in ".!?,":
                if not token.endswith(" "):
                    token += " "
            segmented_parts.append(token)

    return "".join(segmented_parts)


sampletext = "okay...firsttoAnnericeBOOKfans....<br/><br/>surelestat'seyesarenotblue...sureheisn'tblondinthismovie...buteventhoughMariusisnotlestat'smaker...eventhoughtheyCOMPLETELYalteredthestory.....<br/><br/>howcanusayitsnotagoodmovie..<br/><br/>thismovie...istheBESTvampiremovieieversaw...andlestatispicturedperfectlyinit....maybenothisfeatures...butidon'tthinkonecanfindabetterlestat....thewayhespeaks...andthewayhelooksatmeremortals...hisarrogance..andsheerloveforfameispicturedflawlessly.<br/><br/>ifuforonce...consideritjustamovie..andnottryandrelateeveryscenetothebook...uwilllovethemovieasmuchasido.<br/><br/>now...tothenonreaders..<br/><br/>bepreparedtofallabsolutelyinlovewiththismovie....ithaseverything....andthegothmusic...islikeanaddedtreat...thedialogues...arebeautiful...andcatching...andeventhoughitsavampiremovie..uwillfindyourselfsmiling...atthewitofthecharacters...anduwillfindyourselfsympathizingwiththevampires..<br/><br/>overall...oneofmyfavmovies...!!10/10"

print (smart_segment(sampletext))


