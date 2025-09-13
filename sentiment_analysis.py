def load_afinn_to_dictionary(path: str) -> dict:
    """
    load the afinn txt file to python dictionary

    :param path: path to the afinn txt file
    :returns: afinn txt file in dictionary format
    """
    afinn_dict = {}

    # open the afinn txt file
    with open(path, "r") as file:
        for line in file:
            # each line uses \t to separate the word and word score 
            # e.g. abandon -2
            splitted_line = line.strip().split('\t')
            word = splitted_line[0]
            score = splitted_line[1]

            afinn_dict[word] = int(score)
    return afinn_dict


def get_sentence_sentiment_score(afinn_dict: dict, sentence: str) -> int:
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
    sentence_word_list = sentence.split(" ")
    sentence_word_count = len(sentence_word_list)
    afinn_words_list = list(afinn_dict.keys())

    for word in sentence_word_list:
        found = binary_search(afinn_words_list, word)

        if found:
            score += afinn_dict[word]

    final_score = round(score / sentence_word_count, 5)

    # rescale score so that the max range is -1 to 1 instead of -5 to 5
    rescaled_score = final_score / 5

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
    afinn = load_afinn_to_dictionary("AFINN-en-165.txt")
    score_list = []

    for sentence_dict in sentences_list:
        tokenized_sentence = sentence_dict['tokens']

        score = get_sentence_sentiment_score(afinn, tokenized_sentence)
        score_list.append(score)

    modified_dict = add_score_to_dict(sentences_list, score_list)

    return modified_dict