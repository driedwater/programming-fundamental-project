from typing import Dict, List, Tuple, Optional
from afinn_loader import get_afinn

afinn = get_afinn()

def build_multiword_info(afinn: dict[str, int]) -> tuple[dict[str, int], int]:
    """
    Extract multi-word phrases from an AFINN dictionary and compute
    the maximum phrase length.

    The function filters entries whose keys contain spaces and collects them
    into a new dictionary. It also tracks the maximum number of words found
    in any phrase.

    :param afinn: A dictionary mapping words or phrases to sentiment scores.
    :type afinn: dict[str, int]

    :return: A tuple containing:
             - A dictionary of multi-word phrases and their associated scores.
             - The length of the longest multi-word phrase (in tokens).
    :rtype: tuple[dict[str, int], int]
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
    Fold multi-word phrases into single tokens based on a sentiment dictionary.

    The function attempts to match the longest possible n-gram at each token
    position (up to ``max_n``). If a candidate phrase is found in the
    ``multiword_dict`` or matches via an alias in ``alias_map``, it is
    collapsed into a single token. Unmatched tokens remain unchanged.

    :param tokens: A list of individual word tokens to scan for multi-word matches.
    :type tokens: list[str]

    :param multiword_dict: A mapping of known multi-word phrases to sentiment scores.
    :type multiword_dict: dict[str, int]

    :param max_n: The maximum phrase length (in tokens) to consider while matching.
    :type max_n: int

    :param alias_map: Optional mapping of alternate phrase forms to their canonical
                      equivalents. Defaults to ``None``.
    :type alias_map: dict[str, str], optional

    :return: A tuple containing:
             - A new list of tokens where matched multi-word phrases are collapsed.
             - A list of dictionaries describing each match with keys:
               ``term``, ``score``, ``start``, and ``length``.
    :rtype: tuple[list[str], list[dict]]
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

