# Identify the sentence with the highest sentiment score in the entire text
def most_positive_sentence(Dictionary: list[dict]) -> tuple[float, str]:
    max_score = None
    for line in Dictionary:
        original_sentence = line["original"]
        token_sentence = line["tokens"]
        current_score = line["score"]

        # Skip if tokenized sentence is blank, score will remain as none
        if len(token_sentence) == 0:
            continue

        # Check if previous score was blank or current score is higher than the previous score
        # If true then update the max value and line
        elif max_score is None or current_score > max_score:
            max_score = current_score
            max_line = original_sentence

        # If there is more than 1 sentence with the highest score then add it to the existing max line
        elif current_score == max_score:
            max_line += "\n" + original_sentence

    # Return insufficient sentence to determine score if tokenized sentence is blank/None
    if max_score is None:
        return "Insufficient sentences available"
    
    return [max_score, max_line]


# Identify the sentence with the lowest sentiment score in the entire text
def most_negative_sentence(Dictionary: list[dict]) -> tuple[float, str]:
    min_score = None
    for line in Dictionary:
        original_sentence = line["original"]
        token_sentence = line["tokens"]
        current_score = line["score"]
        
        # Skip if tokenized sentence is blank, score will remain as none
        if len(token_sentence) == 0:
            continue

        # Check if previous score was blank or current score is lower than the previous score
        elif min_score is None or current_score < min_score:
            min_score = current_score
            min_line = original_sentence

        # If there is more than 1 sentence with the lowest score then add it to the existing min line
        elif current_score == min_score:
            min_line += "\n" + original_sentence

    # Return insufficient sentence to determine score if tokenized sentence is blank/None
    if min_score is None:
        return "Insufficient sentences available"
    
    return [min_score, min_line]
