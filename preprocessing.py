from bs4 import BeautifulSoup
import unicodedata2
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP

# Build a regex pattern to match contractions (using the contraction mapping in contractions.py):
# 1. Escape special characters in each contraction (like ' in y'all).
# 2. Sort by longest to shortest so "you'd've" matches before "you'd" (Prevents false positive).
# 3. Join with "|" so regex can match any one of them.
# 4. (?<!\S) and (?!\S) ensure we only match whole words (not inside other words like y'all in fy'all ).
CONTRACTION_PATTERN = re.compile(r'(' + '|'.join(sorted(map(re.escape, CONTRACTION_MAP), key=len, reverse=True)) + r')')

#ToktokTokenizer from nltk
tokenizer = ToktokTokenizer()

def save_stopwords(filepath="stopwords.txt", afinn_path="AFINN-en-165.txt"):
    """
    Create a local file stopwords.txt, that saves NLTK stopwords to a file,
    excluding any words that:
    1) appears as a single word in the Afinn lexicon
    2) appears as a multi-word phrase in the Afinn lexicon
    (Basically excludes all words in the Afinn lexicon)
    """

    # Download and load nltk english stopwords
    nltk.download('stopwords')
    stopwords_set = set(stopwords.words("english"))

    # Collect all Afinn words (both single and multi-word parts)
    with open(afinn_path, "r", encoding="utf-8") as f:
        afinn_words = {
            word
            for line in f
            # Take the part before the tab
            # Split multi-word phrases into single words
            for word in line.split("\t")[0].split()
        }

    # Keep only nltk stopwords not found in Afinn
    filtered = stopwords_set - afinn_words

    # Write stopwords to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(filtered)))

def load_stopwords(filepath="stopwords.txt"):
    # Create stopwords.txt if filepath does not exist
    if not os.path.exists(filepath):
        save_stopwords(filepath)

    # Open the file and return contents as set
    with open(filepath, "r", encoding="utf-8") as f:
        loaded = set(f.read().splitlines())
    return loaded

def remove_stopwords(text, filepath="stopwords.txt"):
    # Call function to load stopwords.txt as a set
    stopword_set = load_stopwords(filepath)
    # Split text into tokens
    # Must do contraction handling before toktoktokenizer to prevent ' symbols
    tokens = tokenizer.tokenize(text)

    # Remove trailing and leading whitespace in text
    tokens = [token.strip().lower() for token in tokens]
    # Filters tokens, only keeping tokens not in the stopword set
    processed_tokens = [token for token in tokens if token not in stopword_set]

    processed_text = ' '.join(processed_tokens)
    return processed_text

# Removes html tags like line break <br />
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# Changes accented characters to normal, like éxpéctéd to expected
def convert_accented_characters(text):
    text = unicodedata2.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Substitute special characters with ''. regex to identify anything that's NOT a letter, digit, or whitespace. removes emoticons like :D
def remove_special_characters(text):
    text = re.sub(r'[^a-zA-z0-9\s]', '', text)
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
    # if p.strip() ensure empty paragraphs are ignored
    paragraphs = [p.strip() for p in text.split("<br /><br />") if p.strip()]

    # Regex to identify where the sentence ends
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    # Apply the split to each paragraph
    paragraphs_sentences = [sentence_endings.split(p) for p in paragraphs]

    hierarchical_tokens = []

    # Loop through paragraphs
    for p, para in enumerate(paragraphs_sentences, 1):
        # Loop through sentences
        for s, sentence in enumerate(para, 1):
            # Convert accented text first so text like neédn't are likely to be accepted further down
            text = convert_accented_characters(sentence.lower())
            # Convert contractions before removing stopwords as it is affected by toktoktokenizer()
            # Convert contractions before removeing special characters to keep functionality
            text = convert_contractions(text)
            # Remove html first before removing other special characters
            text = remove_html_tags(text)
            # Remove all special characters
            text = remove_special_characters(text)
            # Remove stopwords
            text = remove_stopwords(text)

            hierarchical_tokens.append({
                "para": p,
                "sentence": s,
                "original": sentence,
                "tokens": text
            })

    return  hierarchical_tokens

if __name__ == "__main__":
    sample_text = "***who now??? :D**** whom Our <p></p> NOT GOOD éXpéctéd<br /><br /> Must've needn't. needn wouldn't <br /><br /> ourselves you ya'll. meow y'all're <br /><br /> y'all've wrapped"
    # sample_text.islower()
    # sample_text = """
    # ***May Contain Spoilers*** OK, it wasn't exactly as good as expected in fact it was a lot different than I had thought it would be but it still turned out to be a pretty good movie.<br /><br />I usually don't care too much for that type of music but in this movie it worked perfectly (I mean duh he's a rock star) but anyway I loved Stuart Townsend in this, and Aaliyah, although she had a small part in the movie was amazing.<br /><br />And even though Tom Cruise played Lestat in the Interview with a Vampire, I have to admit that I am glad he turned down the role even though I normally hate when they use different people to play the same characters in like sequels and stuff.<br /><br />Overall, the movie was great and I enjoyed watching it, even if there were parts that could have been better. Great vampire movie.
    # """

    result = complete_tokenization(sample_text)
    print(result)
