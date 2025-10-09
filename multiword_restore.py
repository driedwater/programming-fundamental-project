from typing import Optional
from afinn_loader import get_afinn

afinn = get_afinn()

def build_multiword_info_tuples() -> tuple[dict[tuple[str, ...], int], int]:
    """
    Build a tuple-keyed multiword dictionary from the global AFINN.

    Returns:
        (multiword_dict, max_length)
          - multiword_dict: maps token tuples (e.g., ("not","good")) -> score
          - max_length: length of the longest multiword phrase in tokens
    """
    multiword_dict: dict[tuple[str, ...], int] = {}
    max_length = 1

    # Only keep entries whose keys contain spaces
    for phrase, score in afinn.items():
        if " " in phrase:
            token_tuple = tuple(phrase.split())
            multiword_dict[token_tuple] = score
            if len(token_tuple) > max_length:
                max_length = len(token_tuple)

    return multiword_dict, max_length


def convert_aliases_to_tuples(alias_map: dict[str, str]) -> dict[tuple[str, ...], tuple[str, ...]]:
    """
    Convert a string->string alias map into tuple->tuple keys/values.

    Example:
        {"not great": "not good"} -> {("not","great"): ("not","good")}
    """
    return {tuple(src.split()): tuple(dst.split()) for src, dst in alias_map.items()}


def fold_multiword_phrases(
    tokens: list[str],
    multiword_dict: dict[tuple[str, ...], int],
    max_length: int,
    alias_map: Optional[dict[tuple[str, ...], tuple[str, ...]]] = None
) -> tuple[list[str], list[dict]]:
    """
    Fold multiword phrases using tuple keys in ~O(n) time (for small constant max_length).

    Algorithm:
      • Walk left→right through tokens.
      • At each position, try the longest window first (up to max_length).
      • Look up the tuple window in the dict; if not found, optionally try alias→canonical.
      • On match: append the folded phrase once and jump ahead by its length.
      • Else: append the single token and advance by 1.

    Args:
        tokens: Tokenized sentence/segment.
        multiword_dict: Tuple-keyed multiword sentiment dictionary.
        max_length: Maximum n-gram size to consider (in tokens).
        alias_map: Optional tuple-keyed alias map.

    Returns:
        folded_tokens, matches
          - folded_tokens: final tokens with phrases collapsed as "w1 w2 ..."
          - matches: list of {"term","score","start","length"} for debugging/analysis
    """
    position = 0
    total_tokens = len(tokens)
    folded_tokens: list[str] = []
    matches: list[dict] = []

    while position < total_tokens:
        matched_length = 0
        matched_phrase: Optional[tuple[str, ...]] = None
        matched_score: Optional[int] = None

        # Ensures n-gram will never be built longer than what's available
        max_window = min(max_length, total_tokens - position)

        # Greedy longest-first match
        for size in range(max_window, 1, -1):
            window = tuple(tokens[position : position + size])

            # Direct lookup
            score = multiword_dict.get(window)
            candidate = window

            # If not in dict, try alias map (map to canonical, then lookup)
            if score is None and alias_map:
                alias_candidate = alias_map.get(window)
                if alias_candidate is not None:
                    score = multiword_dict.get(alias_candidate)
                    if score is not None:
                        candidate = alias_candidate

            if score is not None:
                matched_length = size
                matched_phrase = candidate
                matched_score = score
                break

        if matched_length:
            # Fold the phrase once
            term_str = " ".join(matched_phrase)
            folded_tokens.append(term_str)
            matches.append({
                "term": term_str,
                "score": matched_score,
                "start": position,
                "length": matched_length,
            })
            position += matched_length
        else:
            # Keep the single token
            folded_tokens.append(tokens[position])
            position += 1

    return folded_tokens, matches


# Build the tuple-keyed dictionary + longest phrase length ONCE using Afinn
multiword_tuples, max_phrase_length = build_multiword_info_tuples()


def fold_multiword_phrases_using_globals(
    tokens: list[str],
    alias_map_strings: Optional[dict[str, str]] = None
) -> tuple[list[str], list[dict]]:
    """
    Convenience wrapper that uses module-level caches:
      - multiword_tuples (tuple-keyed phrases)
      - max_phrase_length (longest phrase length)
      - optional alias map converted to tuple form

    Args:
        tokens: Tokenized sentence/segment.
        alias_map_strings: Optional string-keyed alias map.

    Returns:
        folded_tokens, matches
    """
    alias_map_tuples = convert_aliases_to_tuples(alias_map_strings or {})
    return fold_multiword_phrases(tokens, multiword_tuples, max_phrase_length, alias_map_tuples)
