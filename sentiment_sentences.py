def most_positive_sentence(scored_text: list[dict]) -> tuple[float, str] | str:
    """
    This function identifies the sentence with the highest sentiment score in the entire text

    :param scored_text: output of the text after sentiment analysis
    :type scored_text: list[dict]

    :returns: tuple of the score and sentence of the most positive sentence
              or an error message if an exception occurred
    :rtype: tuple[float, str] | str
    """

    try:
        # Retrieve all the score and skip the sentence if it does not have any tokenized word to be scored
        score_list = [line['score'] for line in scored_text if line['tokens'] != []]
        max_score = max(score_list)
        # Retrieve all sentences that have this max score
        max_sentences = [line['original'] for line in scored_text if line['score'] == max_score]
        # Combine the max sentences with \n
        combined_sentence = '\n'.join(max_sentences)

        return (max_score, combined_sentence)

    except:
        return ("Insufficient sentences available")


def most_negative_sentence(scored_text: list[dict]) -> tuple[float, str] | str:
    """
    This function identifies the sentence with the lowest sentiment score in the entire text

    :param scored_text: output of the text after sentiment analysis
    :type scored_text: list[dict]

    :returns: tuple of the score and sentence of the most negative sentence or string if an error occurred
    :rtype: tuple[float, str] | str
    """

    try:
        # Retrieve all the score and skip the sentence if it does not have any tokenized word to be scored
        score_list = [line['score'] for line in scored_text if line['tokens'] != []]
        min_score = min(score_list)
        # Retrieve all sentences that have this min score
        min_sentences = [line['original'] for line in scored_text if line['score'] == min_score]
        # Combine the min sentences with \n
        combined_min_sentence = '\n'.join(min_sentences)

        return (min_score, combined_min_sentence)

    except:
        return ("Insufficient sentences available")