import json

from flask import Flask, render_template, request, redirect, url_for
from preprocessing import complete_tokenization
from sentiment_analysis import compute_all_sentences
from SentimentAnalysis_C import most_positive_sentence, most_negative_sentence
from chart import sentiment_gauge
from SlidingWindow import sliding_window
from SlidingWindow2 import sliding_window_2
import urllib.parse
from spacing import load_word_costs, smart_segment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    message, content = None, None

    if request.method == "POST":
        file = request.files.get("file")

        # Check if there is a file and filename
        if not file or file.filename == "":
            message = "Choose a file to upload first!"
        # Accept only .txt files
        elif not file.filename.endswith(".txt"):
            message = "Please upload a .txt file!"
        else:
            # Reading file content directly without saving locally
            try:
                content = file.read().decode("utf-8")
                print("origin:", content)
                # Part of project requirements, checks strictly for strings with no spaces
                if " " not in content.strip():
                    word_cost, maxword = load_word_costs("unigram_freq.csv")
                    content = smart_segment(content, word_cost, maxword)
                    print("spacy:", content)

                tokens = complete_tokenization(content)
                sentences_dict = compute_all_sentences(tokens)
                json_data = json.dumps(sentences_dict)
                # Encode content for URL
                encoded_content = urllib.parse.quote(content)
                return redirect(url_for("results", json_data=json_data, file_content=encoded_content))
            # Added try except to handle decoding errors (tested with corrupt .txt file)
            except Exception:
                message = "Error reading file. Make sure it's a valid text file."

    return render_template("index.html", message=message, content=content)

@app.route('/results')
def results():
    json_data = request.args.get("json_data")
    file_content = request.args.get("file_content", "")
    # Decode content from URL
    file_content = urllib.parse.unquote(file_content)
    sentences_dict = json.loads(json_data)
    most_positive = most_positive_sentence(sentences_dict)
    most_negative = most_negative_sentence(sentences_dict)
    sw_result = sliding_window(sentences_dict)
    sw2_result = sliding_window_2(sentences_dict)

    if not sw2_result or not sw_result or not most_positive or not most_negative:
        # Error case: sw_result contains empty dictionary or empty list
        # Display error message for pos_extract and neg_extract instead of the sentence
        most_positive, most_negative = "Insufficient sentences available"
        pos_extract, neg_extract = "Unable to calculate sliding window"
    else:
           
        #Sliding window 1 (Fixed window size of 3)
        positive_para, negative_para = sw_result
        pos_extract = " ".join(positive_para[0])
        neg_extract = " ".join(negative_para[0])
        pos_extract_fig = sentiment_gauge(positive_para[1])
        neg_extract_fig = sentiment_gauge(negative_para[1])

        #Sliding window 2 (No fixed window size)
        max_segments, min_segments = sw2_result
        # Get the longest sentence from paragraph extract if there's more than 1 sentence with the same sentiment score
        most_positive_dict = max(max_segments, key=lambda d: len(d["sentence"])) if max_segments else {"sentence": "", "score": 0}
        most_negative_dict = max(min_segments, key=lambda d: len(d["sentence"])) if min_segments else {"sentence": "", "score": 0}
        pos_extract2 = most_positive_dict["sentence"]
        neg_extract2 = most_negative_dict["sentence"]
        # only render chart when a score is given
        pos_extract_fig2 = sentiment_gauge(most_positive_dict["score"])
        neg_extract_fig2 = sentiment_gauge(most_negative_dict["score"])

    return render_template(
        "results.html",
        entire_text=file_content,
        pos_sentence=most_positive[1],
        neg_sentence=most_negative[1],
        pos_fig=sentiment_gauge(most_positive[0]),
        neg_fig=sentiment_gauge(most_negative[0]),
        pos_extract=pos_extract,
        pos_extract_fig=pos_extract_fig,
        neg_extract=neg_extract,
        neg_extract_fig = neg_extract_fig,
        pos_extract_fig2=pos_extract_fig2,
        pos_extract2=pos_extract2,
        neg_extract_fig2=neg_extract_fig2,
        neg_extract2=neg_extract2
    )

if __name__ == "__main__":
    app.run(debug=True)




