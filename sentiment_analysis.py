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


def get_sentence_sentiment_score(afinn_dict: list[dict], sentence: str) -> int:
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
    word_list = sentence.split(" ")
    word_count = len(word_list)
    
    for word in word_list:
        # make it more efficient
        # e.g. binary search
        
        if word in afinn_dict:
            score += afinn_dict[word]
    
    final_score = round(score / word_count, 5)

    # rescale score so that the max range is -1 to 1 instead of -5 to 5
    rescaled_score = final_score / 5

    return rescaled_score


def add_score_to_dict(sentences_list: list, score_list: list) -> dict:
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