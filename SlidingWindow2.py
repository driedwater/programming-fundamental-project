def max_sentences(Dict: list[dict]) -> int:
    num_of_sentences = len(Dict)
    return num_of_sentences


def sliding_window_2(Dictionary: list[dict] )-> list[list[dict[str, float | str]]]:
    para_pos = 0
    max_segments = []
    min_segments = []
    length = max_sentences(Dictionary)

    max_score = float('-inf')
    min_score = float('inf')

    try:
        while para_pos < length:
            para_number = Dictionary[para_pos]['para']
            line_pos = para_pos

            # -inf is the maximum negative value
            max_temp_score = float('-inf')
            #inf is the maximum positive value
            min_temp_score = float('inf')

            #position of the first line of the current para
            max_start = para_pos
            min_start = para_pos

            #If current line is from the same paragraph as the previous line, otherwise exit this loop and continue in the outside loop
            while line_pos < length and Dictionary[line_pos]['para'] == para_number:
                current_score =  Dictionary[line_pos]["score"]

                # If token is blank then skip to the next line and start from the current while loop again
                if not Dictionary[line_pos]["tokens"]:
                    line_pos += 1
                    continue

                #Max scoring logic:
                #If existing scoring is negative then there is no need to add the current score because it will make it more negative
                # negative + negative = negative, negative+positive = negative
                #To ensure maxtempscore > maxtempscore+current e.g. 3 > 3+(-4) = 3 >-1 true, else -2 < -2+1 = -2 > 1 false
                #If maxtempscore+current < maxtempscore then we do not want to add the negative score
                if max_temp_score < 0:
                    max_temp_score = current_score
                    max_start = line_pos
                    
                #If the existing scoring is not negative then add the current score to the existing scoring
                else:
                    max_temp_score+=current_score

                # Check maxtempscore in the current segment is more the most max score found
                #identify the current segment by getting the start position of the segment and end position which is the current position
                if max_temp_score > max_score:
                    max_score = max_temp_score
                    max_segments = [{
                        'sentence': " ".join(Dictionary[i]['original'] for i in range(max_start, line_pos + 1)),
                        'score': max_temp_score
                    }]
                    
                elif max_temp_score == max_score:
                    max_segments.append({
                        'sentence': " ".join(Dictionary[i]['original'] for i in range(max_start, line_pos + 1)),
                        'score': max_temp_score
                    })

                #Min scoring logic:
                #If positive min temp score is added to the current score it will cause it to be more positive
                # If mintempscore+current > mintempscore then we do not want to add the positive score
                if min_temp_score > 0 :
                    min_temp_score = current_score
                    min_start = line_pos
                    
                else:
                    min_temp_score += current_score

                #Compare mintempscore which is the score for the current segment against the most min score found
                if min_temp_score < min_score:
                    min_score = min_temp_score
                    min_segments = [{
                        'sentence': " ".join(Dictionary[i]['original'] for i in range(min_start, line_pos + 1)),
                        'score': min_temp_score
                    }]
                    
                elif min_temp_score == min_score:
                    min_segments.append({
                        'sentence': " ".join(Dictionary[i]['original'] for i in range(min_start, line_pos + 1)),
                        'score': min_temp_score
                    })

                #If scoring is not more than the max score and not less than the min score then move on the next line
                line_pos+=1

            #If line is in the next para then update the para pos to the current line pos
            para_pos=line_pos

        return [max_segments,min_segments]

    except:
        return "Unable to calculate sliding window"