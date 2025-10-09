from bs4 import BeautifulSoup
import unicodedata2
import re
import nltk
import os
from typing import List, Dict, Optional, TypedDict
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
from afinn_loader import get_afinn
from multiword_restore import fold_multiword_phrases_using_globals
from aliases import load_alias_map
from collections import Counter, deque




# Download nltk packages required from the preprocesses
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download('stopwords', quiet=True)

"""
Build a regex pattern to match contractions (using the contraction mapping in contractions.py):
1. Escape special characters in each contraction (like ' in y'all).
2. Sort by longest to shortest so "you'd've" matches before "you'd" (Prevents false positive).
3. Join with "|" so regex can match any one of them.
"""
CONTRACTION_PATTERN = re.compile(r'(' + '|'.join(sorted(map(re.escape, CONTRACTION_MAP), key=len, reverse=True)) + r')')

# Initialize tokenizer, lemmatizer, and afinn globally
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
afinn = get_afinn()
alias_map = None

# Cache stopwords to avoid reloading every call
_STOPWORD_CACHE = None

# Matches whitespace that follows a sentence-ending punctuation mark (., !, or ?)
# Used to split text into sentences without losing the punctuation.
sentence_splitting = re.compile(r'(?<=[.!?])\s+')

# Matches any character that is NOT:
#   - a letter (A–Z, a–z)
#   - whitespace
#   - a hyphen (-)
# Used to remove unwanted symbols while preserving valid hyphenated words.
remove_special_char = re.compile(r'[^a-zA-Z\s-]')

# Matches hyphens that appear at invalid word boundaries, specifically:
#   - a hyphen not preceded by a letter
#   - OR a hyphen not followed by a letter
# Used to strip hyphens that don't connect two alphabetic characters.
hyphen_removal = re.compile(r'(?<![a-zA-Z])-|-(?![a-zA-Z])')

def save_stopwords(filepath: str = "stopwords.txt") -> None:
    """
    Create a filtered stopword list and write it to a file.

    This function loads the default NLTK English stopwords, compares them
    against all words found in the AFINN lexicon (including words extracted
    from multi-word phrases), removes any overlapping terms to retain
    sentiment-bearing words, and writes the filtered list to disk.

    :param filepath: The filename or path where the filtered stopword list
                     should be saved. Defaults to ``"stopwords.txt"``.
    :type filepath: str, optional

    :return: None
    :rtype: None
    """

    # load nltk english stopwords
    stopwords_set = set(stopwords.words("english"))

    # Collect all Afinn words (both single and multi-word phrases)
    # Split multi-word phrases into individual words
    afinn_words = set()
    for term in afinn.keys():
        afinn_words.update(term.split())

    # Keep only nltk stopwords not found in Afinn
    filtered = stopwords_set - afinn_words

    # Write stopwords to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(filtered)))


def load_stopwords(filepath: str = "stopwords.txt") -> set[str]:
    """
    Load and cache stopwords from a file.

    Checks a global cache to avoid reloading on repeated calls. If the file
    does not exist, it is created using ``save_stopwords()``. The stopwords
    are read into a set for fast membership checking and stored in memory
    for reuse.

    :param filepath: The path to the stopword file. Defaults to
                     ``"stopwords.txt"``.
    :type filepath: str, optional

    :return: A set of stopwords loaded from the file.
    :rtype: set[str]
    """

    # Load stopwords once, cached in memory
    global _STOPWORD_CACHE
    if _STOPWORD_CACHE is not None:
        return _STOPWORD_CACHE

    # Create stopwords.txt if filepath does not exist
    if not os.path.exists(filepath):
        save_stopwords(filepath)

    # Open the file and return contents as set
    with open(filepath, "r", encoding="utf-8") as f:
        _STOPWORD_CACHE = set(f.read().splitlines())
    return _STOPWORD_CACHE



def remove_stopwords(tokens: list[str],
                     filepath: str = "stopwords.txt") -> list[str]:
    """
    Remove stopwords from a list of tokens.

    The function loads a stopword set from the specified file (or from a
    cached version if it has been previously loaded) and filters out any
    tokens that appear in that set.

    :param tokens: A list of token strings to be filtered.
    :type tokens: list[str]

    :param filepath: Path to the stopword file used for filtering.
                     Defaults to ``"stopwords.txt"``.
    :type filepath: str, optional

    :return: A list of tokens with stopwords removed.
    :rtype: list[str]
    """
    stopword_set = load_stopwords(filepath)
    # Remove tokens that are stopwords
    filtered_tokens = [token for token in tokens if token not in stopword_set]
    return filtered_tokens


#Tagging to discover if the word is an adjective, verb, noun or adverb
def get_wordnet_position(tag: str) -> str:
    """
    Map a Penn Treebank-style POS tag to its WordNet equivalent.

    The function checks the first character of the provided part-of-speech
    tag and returns the corresponding WordNet constant (e.g., ADJ, VERB,
    NOUN, or ADV). If the tag does not match any recognized pattern, it
    defaults to ``wordnet.NOUN``.

    :param tag: A part-of-speech tag (e.g., "NN", "VBZ", "JJ", "RB").
    :type tag: str

    :return: The corresponding WordNet POS tag constant.
    :rtype: str
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Lemmatization (reduces word to its dictionary form, e.g., "running" -> "run")
def lemmatize_text(text: list[str]) -> list[str]:
    """
    Lemmatize a list of tokens using POS-aware WordNet mappings.

    The function first tags each token with its part-of-speech using NLTK,
    maps the tag to the corresponding WordNet POS format, and then lemmatizes
    each word. If the original token exists in the AFINN lexicon, it is kept
    unchanged to preserve sentiment-awareness.

    :param text: A list of token strings to lemmatize.
    :type text: list[str]

    :return: A list of lemmatized tokens, with original forms retained when
             present in the AFINN dictionary.
    :rtype: list[str]
    """

    #Tagging the words
    pos_tags=nltk.pos_tag(text)

    lemmas = []
    for word, tag in pos_tags:
        # Map POS tag to WordNet's expected tag set
        wn_pos = get_wordnet_position(tag)

        #Lemmatizing the word
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        #Prefer form that exists in the AFINN Dictionary
        lemmas.append(word if word in afinn else lemma)

    return lemmas


# Removes html tags like line break <br />
def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from the given text.

    Parses the text as HTML and extracts only the visible content,
    stripping out any markup such as <div>, <p>, <span>, etc.

    :param text: The string containing HTML content.
    :type text: str

    :return: The text with all HTML tags removed.
    :rtype: str
    """

    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# Changes accented characters to normal, like éxpéctéd to expected
def convert_accented_characters(text: str) -> str:
    """
    Convert accented Unicode characters to their closest ASCII equivalents.

    This uses Unicode normalization to strip diacritical marks, so strings like
    "résumé" or "café" become "resume" and "cafe".

    :param text: The input string possibly containing accented characters.
    :type text: str

    :return: A string with accented characters replaced by ASCII counterparts.
    :rtype: str
    """

    text = unicodedata2.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters_and_numbers(text: str) -> str:
    """
    Remove unwanted symbols and non-alphabetic characters.

    Special characters, digits, and emoticons are stripped out, with limited
    exceptions for whitespace and valid hyphens between letters.

    :param text: The string to remove special characters from.
    :type text: str

    :return: The cleaned string with disallowed characters removed.
    :rtype: str
    """

    # Step 1: Remove all characters except letters, digits, whitespace, and hyphens
    text = remove_special_char.sub('', text)

    # Step 2: Remove hyphens not between letters/digits
    text = hyphen_removal.sub('', text)
    return text


def convert_contractions(text: str) -> str:
    """
    Expand contractions in the text using a predefined contraction map.

    The function searches for patterns defined in ``CONTRACTION_PATTERN`` and
    replaces each match with its expanded form according to ``CONTRACTION_MAP``.
    Matching is case-sensitive, so only exact forms are replaced.

    :param text: The string containing possible contractions.
    :type text: str

    :return: The text with contractions expanded where applicable.
    :rtype: str
    """

    # Inner function is called for each match found by regex
    def replace_contractions(match: re.Match[str]) -> str:
        # Get the matched contraction from the text
        contraction = match.group(0)
        # Lookup contraction in contraction dictionary
        # return expanded form, otherwise return original form
        expansion = CONTRACTION_MAP.get(contraction)
        return expansion

    # With complied regex CONTRACTION_PATTERN
    # Go through text and find a match, then call replace_contractions function
    return CONTRACTION_PATTERN.sub(replace_contractions, text)


def complete_tokenization(
    text: str,
    alias_tsv_path: Optional[str] = "afinn_aliases.tsv"
) -> list[dict[str, int | str | list[str]]]:

    """
    Fully tokenize text into paragraphs, sentences, and cleaned token lists.

    This function performs multi-stage preprocessing and tokenization,
    generating a structured list of sentence-level results with paragraph and
    sentence indexing.

    Behavior:
        • Splits the text into paragraphs using ``<br /><br />`` as a delimiter.
        • Ignores empty paragraphs (after stripping whitespace).
        • Further splits each paragraph into sentences using
          ``sentence_splitting`` (punctuation-based).
        • Converts text to lowercase and removes accented characters.
        • Strips HTML tags before contraction handling.
        • Expands contractions prior to removing special characters to
          preserve meaning.
        • Removes unwanted symbols, digits, and invalid hyphen usage.
        • Tokenizes the cleaned text into word-like units.
        • Filters out empty tokens and trims whitespace.
        • Identifies and folds multi-word phrases using AFINN aliases and
          the cached multiword dictionary.
        • Removes stopwords and lemmatizes only single-word tokens.
        • Preserves the original order and inserts lemmatized tokens where
          appropriate.
        • Returns a hierarchical structure containing paragraph index,
          sentence index, original sentence text, and the final tokens.

    :param text: The full input string to tokenize, possibly containing
                 HTML and multi-paragraph content.
    :type text: str

    :param alias_tsv_path: Optional path to a TSV file containing alias
                           mappings for multi-word phrases. Defaults to
                           ``"afinn_aliases.tsv"``.
    :type alias_tsv_path: str, optional

    :return: A list of dictionaries, each representing a sentence with:
             - ``"para"`` (int): Paragraph index (1-based),
             - ``"sentence"`` (int): Sentence index within the paragraph (1-based),
             - ``"original"`` (str): The original sentence text,
             - ``"tokens"`` (list[str]): The final processed tokens.
    :rtype: list[SentenceTokens]
    """

    # Splits paragraphs on <br /><br />
    # Removes any leading or trailing whitespaces
    # if p.strip() ensures empty paragraphs are ignored
    paragraphs = [p.strip() for p in text.split("<br /><br />") if p.strip()]

    hierarchical_tokens = []

    global alias_map
    if alias_map is None and alias_tsv_path:
        try:
            alias_map = load_alias_map(alias_tsv_path)
        except Exception:
            alias_map = {}


    # Loop through paragraphs
    for p, para in enumerate(paragraphs, 1):
        # Loop through sentences
        # Split paragraph into sentences based on punctuation (. ! ?),
        # keeping the punctuation attached to the sentence.
        for s, sentence in enumerate(sentence_splitting.split(para), 1):
            clean_lower = sentence.lower()

            clean_accented = convert_accented_characters(clean_lower)

            clean_html = remove_html_tags(clean_accented)

            # Convert contractions before removing special characters to keep functionality
            clean_contractions = convert_contractions(clean_html)

            clean_special = remove_special_characters_and_numbers(clean_contractions)

            # Tokenize text
            clean_tokens = tokenizer.tokenize(clean_special)

            # clean whitespace
            whitespace_tokens = [token.strip() for token in clean_tokens if token.strip()]

            # Fold multi-words with alias matching
            folded_tokens, matches = fold_multiword_phrases_using_globals(
                whitespace_tokens, alias_map
            )

            # Applies stopword removal and lemmatization only for single-word tokens
            singles = [t for t in folded_tokens if " " not in t]

            # While keeping the order, compute lemmas for singles
            singles_wo_stop = remove_stopwords(singles)

            # Precompute lemmas once; use a deque for O(1) pops from the left
            singles_lemmas = deque(lemmatize_text(singles_wo_stop))

            # Counts how many times each kept single should appear (preserve multiplicity)
            keep_counts = Counter(singles_wo_stop)

            #Rebuild the sequence of words in order keeping multi-word tokens as-is
            clean_completed: List[str] = []
            idx_single = 0
            for tok in folded_tokens:
                if " " in tok:
                    clean_completed.append(tok)
                else:
                    # If the current single-word token is not a stopword and unused,
                    # replace it with the next available lemma in the same positional order.
                    # Checks multiplicity
                    if idx_single < len(singles) and keep_counts[singles[idx_single]] > 0:
                        clean_completed.append(singles_lemmas.popleft())  # O(1)
                        keep_counts[singles[idx_single]] -= 1
                    # advance original singles cursor
                    idx_single += 1


            hierarchical_tokens.append({
                "para": p,
                "sentence": s,
                "original": sentence,
                "tokens": clean_completed
            })

    return  hierarchical_tokens

sampletext = ("This place is not good at all. He had some really bad luck yesterday. The food was fucking amazing and the staff were damn good.")
print (complete_tokenization(sampletext))



