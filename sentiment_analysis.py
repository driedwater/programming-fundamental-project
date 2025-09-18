from afinn_loader import get_afinn

# def load_afinn_to_dictionary(path: str) -> dict:
#     """
#     load the afinn txt file to python dictionary
#
#     :param path: path to the afinn txt file
#     :returns: afinn txt file in dictionary format
#     """
#     afinn_dict = {}
#
#     # open the afinn txt file
#     with open(path, "r") as file:
#         for line in file:
#             # each line uses \t to separate the word and word score
#             # e.g. abandon -2
#             splitted_line = line.strip().split('\t')
#             word = splitted_line[0]
#             score = splitted_line[1]
#
#             afinn_dict[word] = int(score)
#     return afinn_dict


def get_sentence_sentiment_score(afinn: dict, tokenized_sentence: dict) -> int:
    """
    This function calculates the sentiment score of the sentence inputted by splitting
    each word, check if word exist in afinn_dict then adding up the score and dividing by
    total word count.
    Afterwards it will rescale the score so that it ranges from -1 to 1 instead of -5 to 5.

    :param afinn_dict: the dataset to use to find the score of each word
    :param sentence: the sentence we want to find the sentiment score of
    :returns: sentiment score of the sentence
    """
    
    score = 0
    sentence_word_count = len(tokenized_sentence)
    afinn_words_list = list(afinn.keys())

    for word in tokenized_sentence:
        found = binary_search(afinn_words_list, word)

        if found:
            score += afinn[word]
    
    if sentence_word_count == 0:
        score = 0
    else:
        score = round(score / sentence_word_count, 5)

    # rescale score so that the max range is -1 to 1 instead of -5 to 5
    rescaled_score = score / 5

    return rescaled_score


def binary_search(input_list: list, word: str) -> bool:
    """
    this function will use binary search algorithm to search for an item in a list

    :params input_list: the list of items to search in
    :params word: the word to search in the list

    :returns: true if found, false if not found
    """

    left_index = 0
    right_index = len(input_list) - 1

    while left_index <= right_index:

        mid_index = (left_index + right_index) // 2

        if input_list[mid_index] == word:
            return True

        elif input_list[mid_index] < word:
            left_index = mid_index + 1

        else:
            right_index = mid_index - 1

    return False

def add_score_to_dict(sentences_list: list[dict], score_list: list) -> dict:
    """
    add sentiment score to a dictionary

    :params sentences_list: the list containing dictionaries where the function will add scores to
    :params score_list: the list of scores to add to each dictionary
    """

    for index, sentence_dict in enumerate(sentences_list):
        current_score = score_list[index]
        sentence_dict["score"] = current_score

    return sentences_list


def compute_all_sentences(sentences_list: list[dict]) -> list[dict]:
    """
    computes all the sentiment score of the sentence dictionary in the list
    and output the score.

    :params sentences_list: the list of dictionaries containing the sentences used to compute the sentiment score 
    :returns: the same list[dict] but with scoring in each dictionary
    """
    afinn = get_afinn()
    score_list = []

    for sentence_dict in sentences_list:
        tokenized_sentence = sentence_dict['tokens']
        print(tokenized_sentence)

        score = get_sentence_sentiment_score(afinn, tokenized_sentence)
        score_list.append(score)

    modified_dict = add_score_to_dict(sentences_list, score_list)

    return modified_dict


if __name__ == "__main__":
    sentences = [{'para': 1, 'sentence': 1, 'original': "***May Contain Spoilers*** OK, it wasn't exactly as good as expected in fact it was a lot different than I had thought it would be but it still turned out to be a pretty good movie.", 'tokens': 'may contain spoilers ok not exactly good expected in fact lot different thought would still turned pretty good movie'}, {'para': 2, 'sentence': 1, 'original': "I usually don't care too much for that type of music but in this movie it worked perfectly (I mean duh he's a rock star) but anyway I loved Stuart Townsend in this, and Aaliyah, although she had a small part in the movie was amazing.", 'tokens': ['usually', 'not', 'care', 'much', 'type', 'of', 'music', 'in', 'movie', 'worked', 'perfectly', 'mean', 'duh', 'rock', 'star', 'anyway', 'loved', 'stuart', 'townsend', 'in', 'aaliyah', 'although', 'small', 'part', 'in', 'movie', 'amazing']}, 
    {'para': 2, 'sentence': 1, 'original': ",,,", 'tokens': []}]
    print(compute_all_sentences(sentences))