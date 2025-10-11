from functools import lru_cache
from typing import Dict

@lru_cache(maxsize=1)
def load_alias_map(path: str) -> dict[str, str]:
    """
    Load and cache alias-to-canonical phrase mappings from a TSV file.

    Each non-empty, non-comment line in the file should contain an alias
    and its corresponding canonical phrase, separated by a tab character.
    Lines beginning with "#" or consisting only of whitespace are ignored.
    The result is lowercased and stored in a dictionary for fast lookup.

    :param path: The file path to the TSV file containing alias mappings.
    :type path: str

    :return: A dictionary mapping each alias (lowercased) to its canonical
             phrase (also lowercased).
    :rtype: dict[str, str]

    :raises ValueError: If a non-comment line does not contain a tab
                        separating alias and canonical forms.
    """
    alias_map: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                alias, canonical = s.split("\t", 1)
            except ValueError:
                raise ValueError(f"[aliases] Bad line {ln}: {s!r} (expected 'alias<TAB>canonical')")
            alias_map[alias.strip().lower()] = canonical.strip().lower()
    return alias_map