from sliding_window_fixed import total_sentences


def update_segment(scored_text: list[dict], start_pos: int, end_pos: int, temp_score: float) -> dict[str, str | float]:
    """
    This function combines the sentences in the current continuous segment and
    the total score into a dictionary based on the range of the segment provided

    :param scored_text: output of the text after sentiment analysis
    :param start_pos: The starting sentence position in the current segment
    :param end_pos: The ending sentence position in the current segment
    :param temp_score: Total score of the segment

    :type scored_text: list[dict]
    :type start_pos: int
    :type end_pos: int
    :type temp_score: float

    :returns: The combined sentences and score of the current segment
    :rtype: dict[str, str | float] 
    """

    updated_segment = {'sentence': " ".join(scored_text[line]['original'] for line in range(start_pos, end_pos + 1)),
                       'score': temp_score}

    return updated_segment


def sliding_window_2(scored_text: list[dict]) -> list[list[dict[str, float]]] | str:
    """
    Finds the most positive and most negative sentence segments using a sliding window approach.
    A fixed window is not set for this function.

    :param scored_text: output of the text after sentiment analysis
    :type scored_text: list[dict]

    :returns: the most positive segments and the most negative segments
              Or a string error message if processing fails.
    :rtype: list[list[dict[str, float]]] | str 
    """

    try:
        para_pos = 0
        max_segments = []
        min_segments = []
        length = total_sentences(scored_text)

        # -inf is the maximum negative value
        max_score = float('-inf')
        # inf is the maximum positive value
        min_score = float('inf')

        while para_pos < length:
            para_number = scored_text[para_pos]['para']
            line_pos = para_pos
            max_temp_score = float('-inf')
            min_temp_score = float('inf')

            # Position of the first line of the current para
            max_start = para_pos
            min_start = para_pos

            # If current line is from the same paragraph as the previous line, otherwise exit this loop and continue in the outside loop
            while line_pos < length and scored_text[line_pos]['para'] == para_number:
                current_score = scored_text[line_pos]["score"]

                # If token is blank then skip to the next line and start from the current while loop again
                if not scored_text[line_pos]["tokens"]:
                    line_pos += 1
                    continue

                # Max scoring logic:
                # If existing scoring is negative then there is no need to add the current score because it will make it more negative
                # negative + negative = negative, negative + positive = negative
                # To checks if maxtempscore > maxtempscore + current e.g. 3 > 3+(-2) = 3 >-2 true, else 3 > 3+2 = 3 > 5 false
                # If maxtempscore > maxtempscore + current then we do not want to the current negative score

                if max_temp_score < 0:
                    max_temp_score = current_score
                    max_start = line_pos

                # If the existing scoring is not negative then add the current score to the existing scoring
                else:
                    max_temp_score += current_score

                # Check maxtempscore in the current segment is more the most max score found
                # Identify the current segment by getting the start position of the segment and end position which is the current position
                if max_temp_score > max_score:
                    max_score = max_temp_score
                    max_segments = [update_segment(scored_text, max_start, line_pos, max_temp_score)]

                elif max_temp_score == max_score:
                    max_segments.append(update_segment(scored_text, max_start, line_pos, max_temp_score))

                # Min scoring logic:
                # If positive min temp score is added to the current score it will cause it to be more positive
                # If mintempscore < mintempscore + current then we do not want to add the current positive score
                if min_temp_score > 0:
                    min_temp_score = current_score
                    min_start = line_pos

                else:
                    min_temp_score += current_score

                # Compare mintempscore which is the score for the current segment against the most min score found
                if min_temp_score < min_score:
                    min_score = min_temp_score
                    min_segments = [update_segment(scored_text, min_start, line_pos, min_temp_score)]

                elif min_temp_score == min_score:
                    min_segments.append(update_segment(scored_text, min_start, line_pos, min_temp_score))

                # If scoring is not more than the max score and not less than the min score then move on the next line
                line_pos += 1

            # If line is in the next para then update the para pos to the current line pos
            para_pos = line_pos
        if max_segments == [] and min_segments == []:
            return "Unable to calculate sliding window"
        else:
            return [max_segments, min_segments]

    except:
        return "Unable to calculate sliding window"