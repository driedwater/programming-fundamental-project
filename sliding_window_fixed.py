def total_sentences(text: list[dict]) -> int:
    """
    Determine the number of sentences in the text

    :param text: a list of dictionaries containing the sentences
    :type text: list[dict]

    :returns: total number of sentences
    :rtype: int
    """

    num_of_sentences = len(text)

    return num_of_sentences


def sliding(scored_text: list[dict]) -> list[tuple[str, float]] | None:
    """
    Performs the sliding of the window with 3 sentences in each window. 
    Where all 3 sentences have to be of the same paragraph.

    :param scored_text: output of the text after sentiment analysis
    :type scored_text: list[dict]

    :returns: a list containing the pairs of sentences in window and their combined score for each window
              or None if an exception occurred
    :rtype: list[tuple[str, float]] | None
    """

    # window_list will store the sentences and sum of the score for the 3 sentences in the window
    window_list = []
    sentence_para = 0
    try:
        # Check that there is a 3rd sentence within the text
        while sentence_para + 2 < total_sentences(scored_text):
            # first, second and third refers to the 
            # first, second and third element in the sliding window
            first = scored_text[sentence_para]
            second = scored_text[sentence_para + 1]
            third = scored_text[sentence_para + 2]
            # Identify the paragraph num of the first sentence
            para_number = first["para"]

            # Move window by 1 to the next sentence if any of the 3 sentences in the current window is blank
            if (first["tokens"] == [] or
                    second["tokens"] == [] or
                    third["tokens"] == []):
                sentence_para += 1

            # Ensures that all 3 sentences are from the same paragraph
            # Compare second and third sentences' para number to first sentence para number
            elif (second["para"] == para_number and
                  third["para"] == para_number):

                # Combine the 3 sentences in this window into one
                sentence_segment = " ".join([first["original"], second["original"], third["original"]])
                # Add the scores of the 3 sentences tgt
                temp_score = first["score"] + second["score"] + third["score"]
                # Add the sentences and total score in the current segment to a temporary list
                window_list.append([sentence_segment, temp_score])
                sentence_para += 1

            # Move to the next sentence if it does not fulfil the above conditions
            else:
                sentence_para += 1

        # Return the temporary list if it is not empty, else return None
        if window_list != []:
            return window_list

        else:
            return None

    except:
        return None


def positive_segment(window_score: list[list]) -> list[tuple[list[str], float]]:
    """
    Identify the segment with the highest sentiment score and the sentences in that segment

    :param window_score: results of the sliding window which contains the sentences and score of each window
    :type window_score: list[list]

    :returns: a list containing the sentences and total score of the most positive segment
    :rtype: list[tuple[list[str], float]
    """

    # Retrieve the highest sentiment score
    most_positive_score = max(segment[1] for segment in window_score)
    # Retrieve the segments that has the highest score
    most_positive_sentences = [segment[0] for segment in window_score
                               if segment[1] == most_positive_score]

    return [most_positive_sentences, most_positive_score]


def negative_segment(window_score: list[list]) -> list[tuple[list[str], float]]:
    """
    Identify the segment with the lowest sentiment score and the sentences in that segment

    :param window_score: results of the sliding window which contains the sentences and score of each window
    :type window_score: list[list]

    :returns: a list containing the sentences and total score of the most negative segment
    :rtype: list[tuple[list[str], float]
    """

    # Retrieve the lowest sentiment score
    most_negative_score = min(segment[1] for segment in window_score)
    # Retrieve the segments that has the lowest score
    most_negative_sentences = [segment[0] for segment in window_score
                               if segment[1] == most_negative_score]

    return [most_negative_sentences, most_negative_score]


def sliding_window(scored_text: list[dict]) -> list[tuple[list[str], float]] | str:
    """
    Combine the sliding window function with the most positive and negative segment function

    :param scored_text: output of the text after sentiment analysis
    :type scored_text: list[dict]

    :returns: a list with 2 tuples containing the sentences and total score of the most positive segment
              followed by the sentences and total score of the most negative segment
              or an error message if an exception occurred during the process
    :rtype: list[tuple[list[str], float]] or str
    """

    result = sliding(scored_text)
    if result is None:
        # No segment calculated due to insuffient sentences in the window
        return "Unable to calculate sliding window"

    else:
        most_positive_segment = positive_segment(result)
        most_negative_segment = negative_segment(result)

        return [most_positive_segment, most_negative_segment]