from bs4 import BeautifulSoup
import unicodedata2
import re
import nltk
import os
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
from afinn_loader import get_afinn

# Download nltk packages required from the preprocesses
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download('stopwords', quiet=True)

# Build a regex pattern to match contractions (using the contraction mapping in contractions.py):
# 1. Escape special characters in each contraction (like ' in y'all).
# 2. Sort by longest to shortest so "you'd've" matches before "you'd" (Prevents false positive).
# 3. Join with "|" so regex can match any one of them.
CONTRACTION_PATTERN = re.compile(r'(' + '|'.join(sorted(map(re.escape, CONTRACTION_MAP), key=len, reverse=True)) + r')')

# Initialize tokenizer, lemmatizer, and afinn globally
tokenizer = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
afinn = get_afinn()

# Cache stopwords to avoid reloading every call
_STOPWORD_CACHE = None

def save_stopwords(filepath="stopwords.txt"):
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


def load_stopwords(filepath="stopwords.txt"):
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


# Remove stopwords from a list of tokens.
def remove_stopwords(tokens, filepath="stopwords.txt"):
    stopword_set = load_stopwords(filepath)
    # Remove tokens that are stopwords
    filtered_tokens = [token for token in tokens if token not in stopword_set]
    return filtered_tokens


#Tagging to discover if the word is an adjective, verb, noun or adverb
def get_wordnet_position(tag):
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
def lemmatize_text(text):
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
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# Changes accented characters to normal, like éxpéctéd to expected
def convert_accented_characters(text):
    text = unicodedata2.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text):
    """
    Substitute special characters with ''. regex to identify anything
    that's NOT a letter, digit, or whitespace. removes emoticons like :D
    """

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def convert_contractions(text):
    """
    Expands contractions using CONTRACTION_MAP from contractions.py.
    Case-sensitive, only exact matches are replaced.
    """

    # Inner function is called for each match found by regex
    def replace_contractions(match: re.Match):
        # Get the matched contraction from the text
        contraction = match.group(0)
        # Lookup contraction in contraction dictionary
        # return expanded form, otherwise return original form
        expansion = CONTRACTION_MAP.get(contraction)
        return expansion

    # With complied regex CONTRACTION_PATTERN
    # Go through text and find a match, then call replace_contractions function
    return CONTRACTION_PATTERN.sub(replace_contractions, text)


def complete_tokenization(text):
    # Splits paragraphs on <br /><br />
    # Removes any leading or trailing whitespaces
    # if p.strip() ensures empty paragraphs are ignored
    paragraphs = [p.strip() for p in text.split("<br /><br />") if p.strip()]

    # Regex to identify where the sentence ends
    sentence_endings = re.compile(r'(?<=[.!?])\s+')

    hierarchical_tokens = []

    # Loop through paragraphs
    for p, para in enumerate(paragraphs, 1):
        # Loop through sentences
        for s, sentence in enumerate(sentence_endings.split(para), 1):
            clean_lower = sentence.lower()
            clean_accented = convert_accented_characters(clean_lower)
            clean_html = remove_html_tags(clean_accented)
            # Convert contractions before removing special characters to keep functionality
            clean_contractions = convert_contractions(clean_html)
            clean_special = remove_special_characters(clean_contractions)

            # Tokenize text
            clean_tokens = tokenizer.tokenize(clean_special)
            # clean whitespace
            whitespace_tokens = [token.strip() for token in clean_tokens if token.strip()]
            clean_stopwords = remove_stopwords(whitespace_tokens)
            clean_completed = lemmatize_text(clean_stopwords)

            hierarchical_tokens.append({
                "para": p,
                "sentence": s,
                "original": sentence,
                "tokens": clean_completed
            })

    return  hierarchical_tokens


if __name__ == "__main__":
    sample_text = "***who now??? :D**** boring 10/10 dogs studies cats whom Our <p>Y'all</p> NOT GOOD éXpéctéd<br /><br /><br /><br /> Must've needn't. needn wouldn't <br /><br /> ourselves you ya'll. meow y'all're <br /><br /> y'all've wrapped"
    # sample_text = """
    # ***May Contain Spoilers*** OK, it wasn't exactly as good as expected in fact it was a lot different than I had thought it would be but it still turned out to be a pretty good movie.<br /><br />I usually don't care too much for that type of music but in this movie it worked perfectly (I mean duh he's a rock star) but anyway I loved Stuart Townsend in this, and Aaliyah, although she had a small part in the movie was amazing.<br /><br />And even though Tom Cruise played Lestat in the Interview with a Vampire, I have to admit that I am glad he turned down the role even though I normally hate when they use different people to play the same characters in like sequels and stuff.<br /><br />Overall, the movie was great and I enjoyed watching it, even if there were parts that could have been better. Great vampire movie.
    # """
    import time
    result = complete_tokenization(sample_text)
    print(time.time())
    print(result)



