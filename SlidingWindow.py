# Determine the number of sentences in the text
def max_sentences(Dict: list[dict]) -> int:
    num_of_sentences = len(Dict)

    return num_of_sentences


# Perform sliding window with 3 sentences of the same paragraph
def sliding(Dictionary: list[dict]) -> list[tuple[str, float]] | None:
    #window_list will store the sentences and sum of the score for the 3 sentences in the window
    window_list = []
    sentence_para = 0
    try:
        # Check that there is a 3rd sentence within the text
        while sentence_para+2 < max_sentences(Dictionary):
            # first, second and third refers to the 
            # first, second and third element in the sliding window
            first = Dictionary[sentence_para]
            second = Dictionary[sentence_para + 1]
            third = Dictionary[sentence_para + 2]
            #Identify the paragraph num of the first sentence
            para_number = first["para"]
            
            # Move window by 1 to the next sentence if any of the 3 sentences in the current window is blank
            if (first["tokens"] == [] or 
                second["tokens"] == [] or 
                third["tokens"] ==[]):
                sentence_para+=1

            # Ensures that all 3 sentences are from the same paragraph
            # Compare second and third sentences' para number to first sentence para number
            elif (second["para"] == para_number and 
                  third["para"] == para_number):

                # Combine the 3 sentences in this window into one
                sentence_segment = first["original"] + " " + second["original"] + " " + third["original"]
                # Add the scores of the 3 sentences tgt
                temp_score = first["score"] + second["score"] + third["score"]
                # Add the sentences and total score in the current segment to a temporary list
                window_list.append([sentence_segment, temp_score])
                # Move to the next sentence
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


# Identify the segment with the highest sentiment score and the sentences in the segment
def positive_paragraph_segment(window_score: list[list]) -> list[tuple[list[str], float]]:
    # Retrieve the highest sentiment score from the list of segments and their scores
    most_positive_score = max(segment[1] for segment in window_score)
    # Retrieve the sentence in the segment that has the highest score
    most_positive_segment = [segment[0] for segment in window_score 
                             if segment[1] == most_positive_score]
    
    return [most_positive_segment, most_positive_score]


# Identify the segment with the lowest sentiment score and the sentences in the segment
def negative_paragraph_segment(window_score: list[list]) -> list[tuple[list[str], float]]:
    # Retrieve the lowest sentiment score from the list of segments and their scores
    most_negative_score = min(segment[1] for segment in window_score)
    # Retrieve the sentences in the segment that has the lowest score
    most_negative_segment = [segment[0] for segment in window_score 
                             if segment[1] == most_negative_score]
    
    return [most_negative_segment, most_negative_score]


def sliding_window(Dictionary: list[dict]) -> list[tuple[list[str], float]] | str:
    result = sliding(Dictionary)
    if result is None:
        # No segment calculated due to insuffient sentences in the window
        return "Insufficient sentences for sliding window."
    
    else:
        # Print most positive and negative segment and score
        positive_seg_score = positive_paragraph_segment(result)
        negative_seg_score = negative_paragraph_segment(result)

        #return f'Most positive segment: {"; ".join(positive_seg)}\nMost positive segment score: {positive_score}\nMost negative segment: {"; ".join(negative_seg)}\nMost negative segment score: {negative_score}'
        return [positive_seg_score, negative_seg_score]