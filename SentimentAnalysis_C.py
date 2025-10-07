# Identify the sentence with the highest sentiment score in the entire text
def most_positive_sentence(Dictionary: list[dict]) -> tuple[float, str] | str:
    try:
        # Retrieve all the score and skip the sentence if it does not have any tokenized word to be scored
        score_list=[d['score'] for d in Dictionary if d['tokens'] != ""]
        max_score = max(score_list)
        # Retrieve all sentences that have this max score
        max_sentences=[line['original'] for line in Dictionary if line['score'] == max_score]
        # Combine the max sentences with \n
        combined_sentence = '\n'.join(max_sentences)

        return (max_score, combined_sentence)

    except:
        return ("Insufficient sentences available")


# Identify the sentence with the lowest sentiment score in the entire text
def most_negative_sentence(Dictionary: list[dict]) -> tuple[float, str] | str:
    try:
        # Retrieve all the score and skip the sentence if it does not have any tokenized word to be scored
        score_list = [d['score'] for d in Dictionary if d['tokens'] != ""]
        min_score = min(score_list)
        # Retrieve all sentences that have this min score
        min_sentences = [line['original'] for line in Dictionary if line['score'] == min_score]
        # Combine the min sentences with \n
        combined_min_sentence = '\n'.join(min_sentences)
        
        return (min_score, combined_min_sentence)

    except:
        return ("Insufficient sentences available")