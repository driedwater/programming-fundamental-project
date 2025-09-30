from typing import Dict, List, Tuple, Optional
from afinn_loader import get_afinn

afinn = get_afinn()

def build_multiword_info(afinn: dict[str, int]) -> tuple[dict[str, int], int]:
    """
        From the AFINN dict, extract only multi-word entries (keys with spaces)
        and compute the maximum phrase length in tokens.

        Returns:
            multiword_dict: { "not good": -3, "does not work": -2, ... }
            max_n: int, length of the longest multi-word phrase
    """
    multiword_dict: Dict[str, int] = {}
    max_n = 1
    for k, v in afinn.items():
        if " " in k:
            multiword_dict[k] = v
            n = len(k.split())
            if n > max_n:
                max_n = n
    return multiword_dict, max_n


def fold_multiword_phrases(tokens: list[str],
                           multiword_dict: dict[str, int],
                           max_n: int,
                           alias_map: Optional[dict[str, str]] = None
                           ) -> tuple[list[str], list[dict]]:
    """
    Try longest n-grams first. For each candidate phrase, check:
      - candidate in multiword_dict
      - alias_map[candidate] in multiword_dict (if alias_map provided)
    On match, collapse the span into a single phrase token (the canonical form
    if the alias matched).
    """
    i = 0
    out: List[str] = []
    matches: List[dict] = []

    while i < len(tokens):
        matched = False
        for n in range(min(max_n, len(tokens) - i), 1, -1):
            candidate = " ".join(tokens[i:i+n])
            canonical = alias_map.get(candidate, candidate) if alias_map else candidate

            # Try raw candidate, then alias→canonical
            if candidate in multiword_dict or canonical in multiword_dict:
                chosen = candidate if candidate in multiword_dict else canonical
                score = multiword_dict.get(chosen, multiword_dict.get(candidate))
                out.append(chosen)
                matches.append({
                    "term": chosen,
                    "score": score,
                    "start": i,
                    "length": n
                })
                i += n
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1

    return out, matches

