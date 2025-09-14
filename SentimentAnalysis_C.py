# Inputs = [{'para': 1, 'sentence': 1, 'original': '***who now???', 'tokens': '', 'score': 0},
#           {'para': 1, 'sentence': 2, 'original': ':D**** whom Our <p></p> NOT GOOD éXpéctéd', 'tokens': 'not good expected', 'score': 0.3},
#           {'para': 2, 'sentence': 1, 'original': "Must've needn't.", 'tokens': 'must need not', 'score': -0.852},
#           {'para': 2, 'sentence': 2, 'original': "needn wouldn't", 'tokens': 'would not', 'score': 0.91},
#           {'para': 3, 'sentence': 1, 'original': "ourselves you ya'll.", 'tokens': 'yall', 'score': 0.99},
#           {'para': 3, 'sentence': 2, 'original': "meow y'all're", 'tokens': 'meow', 'score': 0.99},
#           {'para': 4, 'sentence': 1, 'original': "y'all've wrapped", 'tokens': 'wrapped', 'score': -0.2579}]

#SENTIMENT ANALYSIS
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


