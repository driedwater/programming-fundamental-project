from afinn_loader import get_afinn

def get_sentence_score(afinn: dict, tokenized_sentence: dict) -> float:
    """
    This function calculates the sentiment score of the tokenised sentence,
    check if word exist in afinn then adding up the score and dividing by
    total tokenised word count.
    Afterwards it will rescale the score so that it ranges from -1 to 1 instead of -5 to 5.

    :param afinn: the dataset to use to find the score of each word
    :type afinn: dict

    :param tokenized_sentence: the sentence we want to find the sentiment score of
    :type tokenized_sentence: dict

    :returns: sentiment score of the sentence
    :rtype: float
    """
    
    score = 0
    tokens_word_count = len(tokenized_sentence)

    for word in tokenized_sentence:
        if word in afinn:
            score += afinn[word]
    
    if tokens_word_count == 0:
        score = 0
    else:
        score = round(score / tokens_word_count, 5)

    # rescale score so that the max range is -1 to 1 instead of -5 to 5
    rescaled_score = score / 5

    return rescaled_score


def add_score_to_dict(sentences_list: list[dict], score_list: list) -> list[dict]:
    """
    add sentiment score to each dictionary in the list of dictionaries

    :params sentences_list: the list containing dictionaries where the function will add scores to
    :type sentences_list: list[dict]

    :params score_list: the list of scores to add to each dictionary
    :type score_list: list

    :returns: updated list of dictionaries containing scores for each sentence
    :rtype: list[dict]
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
    :type sentences_list: list[dict]

    :returns: the same sentences_list but with scoring in each dictionary
    :rtype: list[dict]
    """
    afinn = get_afinn()
    score_list = []

    for sentence_dict in sentences_list:
        tokenized_sentence = sentence_dict['tokens']

        score = get_sentence_score(afinn, tokenized_sentence)
        score_list.append(score)

    modified_dict = add_score_to_dict(sentences_list, score_list)

    return modified_dict


if __name__ == "__main__":
    sentences = [{'para': 1, 'sentence': 1, 'original': "***May Contain Spoilers*** OK, it wasn't exactly as good as expected in fact it was a lot different than I had thought it would be but it still turned out to be a pretty good movie.", 'tokens': 'may contain spoilers ok not exactly good expected in fact lot different thought would still turned pretty good movie'}, {'para': 2, 'sentence': 1, 'original': "I usually don't care too much for that type of music but in this movie it worked perfectly (I mean duh he's a rock star) but anyway I loved Stuart Townsend in this, and Aaliyah, although she had a small part in the movie was amazing.", 'tokens': ['usually', 'not', 'care', 'much', 'type', 'of', 'music', 'in', 'movie', 'worked', 'perfectly', 'mean', 'duh', 'rock', 'star', 'anyway', 'loved', 'stuart', 'townsend', 'in', 'aaliyah', 'although', 'small', 'part', 'in', 'movie', 'amazing']}, 
    {'para': 2, 'sentence': 1, 'original': ",,,", 'tokens': []}]
    print(compute_all_sentences(sentences))