from pathlib import Path
from functools import lru_cache
import threading
import requests

LOCAL_AFINN_PATH = Path(__file__).resolve().parent / "afinn" / "AFINN-en-165.txt"
REMOTE_AFINN_URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-en-165.txt"

# Used in get_afinn()
_lock = threading.Lock()

#Returns the afinn txt file in dictionary format
def _parse_afinn(text) -> dict:
    afinn_dict = {}
    for line in text.splitlines():
        # each line uses \t to separate the word and word score
        # e.g. abandon -2
        splitted_line = line.strip().split('\t')
        word = splitted_line[0]
        score = splitted_line[1]

        afinn_dict[word] = int(score)
    return afinn_dict


# Caches the afinn lexicon so it can be reused
@lru_cache(maxsize=1)


def get_afinn() -> dict:
    with _lock:    #Prevents multiple threads to call this simultaneously

        #Uses the locally stored afinn file first
        if LOCAL_AFINN_PATH.exists():
            return _parse_afinn(LOCAL_AFINN_PATH.read_text(encoding="utf-8"))

        # Local file missing → fetch and persist for next runs
        resp = requests.get(REMOTE_AFINN_URL, timeout=10)
        resp.raise_for_status()
        LOCAL_AFINN_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_AFINN_PATH.write_text(resp.text, encoding="utf-8")
        return _parse_afinn(resp.text)




