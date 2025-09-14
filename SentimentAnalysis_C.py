#Identify the sentence with the highest sentiment score in the entire text
def most_positive_sentence(Dictionary):
    max_value = None
    for item in Dictionary:
        original_sentence = item["original"]
        sentence = item["tokens"]
        value = item["score"]
        # Skip if tokens is blank, score will remain as none
        if sentence == "":
            continue
        # Check if previous score was blank or current score is higher than the previous score and update the max value and line
        elif max_value is None or value > max_value:
            max_value = value
            max_line = original_sentence

        #If there is more than 1 sentence with the highest score then add it to the existing max line
        elif value == max_value:
            max_line += "; " + original_sentence

    #Return None if tokenized sentence is blank
    if max_value is None:
        return None, None
    return f'The highest sentiment score is {max_value} \nThe most postive sentence(s) is "{max_line}"'


#Identify the sentence with the lowest sentiment score in the entire text
def most_negative_sentence(Dictionary):
    min_value = None
    for item in Dictionary:
        original_sentence = item["original"]
        sentence = item["tokens"]
        value = item["score"]
        #Skip if tokens is blank, score will remain as none
        if sentence == "":
            continue
        #Check if previous score was blank or current score is lower than the previous score
        elif min_value is None or value < min_value:
            min_value = value
            min_line = original_sentence

        #If there is more than 1 sentence with the lowest score then add it to the existing min line
        elif value == min_value:
            min_line += "; " + original_sentence

    #Return None if tokenized sentence is blank
    if min_value is None:
        return None, None

    return f'The lowest sentiment score is {min_value} \nThe most negative sentence(s) is "{min_line}"'


