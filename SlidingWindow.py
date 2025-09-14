#Determine the number of sentences in the paragraph
def maxSentences(Dict):
    numOfSentences = len(Dict)
    return numOfSentences

#Perform sliding window with 3 sentences for the paragraph number indicated
def sliding(Dictionary, para_number):
    slidingWindowScore = []
    sentence_para = 0
    try:
        #Check that there is a 3rd sentence in the text
        while sentence_para+2 < maxSentences(Dictionary):
            #Move window by 1 to the next sentence if any of the 3 sentences in the current window is blank
            if Dictionary[sentence_para]["tokens"] == "" or Dictionary[sentence_para + 1]["tokens"] == "" or \
            Dictionary[sentence_para + 2]["tokens"] == "":
                sentence_para+=1

            #Checks if all 3 sentences are in the same paragraph
            elif Dictionary[sentence_para]["para"] == para_number and Dictionary[sentence_para+1]["para"] == para_number and Dictionary[sentence_para+2]["para"] == para_number:
                tempScore = Dictionary[sentence_para]["score"] + Dictionary[sentence_para + 1]["score"] + \
                        Dictionary[sentence_para + 2]["score"]
                sentenceSegment = Dictionary[sentence_para]["original"] + " " + Dictionary[sentence_para + 1]["original"] + " " + \
                              Dictionary[sentence_para + 2]["original"]
                # sentenceSegment = Dictionary[sentence_para]["tokens"] + " " + Dictionary[sentence_para + 1]["tokens"] + " " + \
                #               Dictionary[sentence_para + 2]["tokens"]
                #Add the sentences and total score in the current segment to a temporary list
                slidingWindowScore.append([sentenceSegment, tempScore])
                #Move to the next sentence
                sentence_para += 1

            #Move to the next sentence if it does not fulfil the above conditions
            #E.g. Sentence paragraph has not reached the indicated paragraph no.
            else:
                sentence_para += 1

        #Return the temporary list if it is not empty, else return None
        if slidingWindowScore != []:
            return slidingWindowScore
        else:
            return None
    except:
        return None


# Identify the segment with the highest sentiment score and the sentences in the segment
def positive_paragraph_segment(window_score):
    #Retrieve the highest sentiment score from the list of segments and their scores
    most_positive_score = max(item[1] for item in window_score)
    #Retrieve the sentence in the segment that has the highest score
    most_positive_segment = [sublist[0] for sublist in window_score if
                             len(sublist) > 1 and sublist[1] == most_positive_score]
    return most_positive_segment, most_positive_score

#Identify the segment with the lowest sentiment score and the sentences in the segment
def negative_paragraph_segment(window_score):
    #Retrieve the lowest sentiment score from the list of segments and their scores
    most_negative_score = min(item[1] for item in window_score)
    #Retrieve the sentences in the segment that has the lowest score
    most_negative_segment = [sublist[0] for sublist in window_score if len(sublist) > 1 and sublist[1] == most_negative_score]
    return most_negative_segment, most_negative_score


def sliding_window(Dictionary, para):
    result = sliding(Dictionary, para)
    if result is None:
        #No segment calculated due to insuffient sentences in the window
        return "Insufficient sentences for sliding window."
    else:
        #Print most positive and negative segment and score
        positive_seg, positive_score = positive_paragraph_segment(result)
        negative_seg, negative_score = negative_paragraph_segment(result)
        return f'Most positive segment: {"; ".join(positive_seg)}\nMost positive segment score: {positive_score}\nMost negative segment: {"; ".join(negative_seg)}\nMost negative segment score: {negative_score}'

