from functools import lru_cache
from typing import Dict

@lru_cache(maxsize=1)
def load_alias_map(path: str) -> dict[str, str]:
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