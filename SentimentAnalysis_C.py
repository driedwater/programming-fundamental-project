# Identify the sentence with the highest sentiment score in the entire text
def most_positive_sentence(Dictionary: list[dict]) -> tuple[float, str] | str:
    max_score = float('-inf')
    for line in Dictionary:
        original_sentence = line["original"]
        token_sentence = line["tokens"]
        current_score = line["score"]

        # Skip if tokenized sentence is blank, max score will not change
        if len(token_sentence) == 0:
            continue

        # Check if current score is higher than the previous max score
        elif current_score > max_score:
            max_score = current_score
            max_line = original_sentence

        # If there is more than 1 sentence with the highest score then add it to the existing max line
        elif current_score == max_score:
            max_line += "\n" + original_sentence

    # Return insufficient sentence to determine score if tokenized sentence is blank
    if max_score == float('-inf'):
        return "Insufficient sentences available"
    
    return [max_score, max_line]


# Identify the sentence with the lowest sentiment score in the entire text
def most_negative_sentence(Dictionary: list[dict]) -> tuple[float, str] | str:
    min_score = float('inf')
    for line in Dictionary:
        original_sentence = line["original"]
        token_sentence = line["tokens"]
        current_score = line["score"]
        
        # Skip if tokenized sentence is blank, min score will not change
        if len(token_sentence) == 0:
            continue

        # Check current score is lower than the previous min score
        elif current_score < min_score:
            min_score = current_score
            min_line = original_sentence

        # If there is more than 1 sentence with the lowest score then add it to the existing min line
        elif current_score == min_score:
            min_line += "\n" + original_sentence

    # Return insufficient sentence to determine score if tokenized sentence is blank
    if min_score == float('inf'):
        return "Insufficient sentences available"
    
    return [min_score, min_line]