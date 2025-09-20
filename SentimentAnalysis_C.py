# Identify the sentence with the highest sentiment score in the entire text
def most_positive_sentence(Dictionary):
    max_score = None
    for line in Dictionary:
        original_sentence = line["original"]
        sentence = line["tokens"]
        score = line["score"]

        # Skip if tokenized sentence is blank, score will remain as none
        if len(sentence) == 0:
            continue
        # Check if previous score was blank or current score is higher than the previous score
        # If true then update the max value and line
        elif max_score is None or score > max_score:
            max_score = score
            max_line = original_sentence

        # If there is more than 1 sentence with the highest score then add it to the existing max line
        elif score == max_score:
            max_line += "; " + original_sentence

    # Return insufficient sentence to determine score if tokenized sentence is blank/None
    if max_score is None:
        return "Insufficient sentences available"
    return [max_score, max_line]


# Identify the sentence with the lowest sentiment score in the entire text
def most_negative_sentence(Dictionary):
    min_score = None
    for item in Dictionary:
        original_sentence = item["original"]
        sentence = item["tokens"]
        score = item["score"]
        # Skip if tokenized sentence is blank, score will remain as none
        if len(sentence) == 0:
            continue
        # Check if previous score was blank or current score is lower than the previous score
        elif min_score is None or score < min_score:
            min_score = score
            min_line = original_sentence

        # If there is more than 1 sentence with the lowest score then add it to the existing min line
        elif score == min_score:
            min_line += "; " + original_sentence


    # Return insufficient sentence to determine score if tokenized sentence is blank/None
    if min_score is None:
        return "Insufficient sentences available"
    return [min_score, min_line]
