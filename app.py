import json

from flask import Flask, render_template, request, redirect, url_for
from preprocessing import complete_tokenization
from sentiment_analysis import compute_all_sentences
from sentiment_sentences import most_positive_sentence, most_negative_sentence
from chart import sentiment_gauge
from sliding_window_fixed import sliding_window
from sliding_window_unfixed import sliding_window_2
import urllib.parse
from spacing import smart_segment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for file upload and preprocessing.

    **GET method**
        Displays the home page and upload form.

    **POST method**
        Handles uploaded `.txt` files, performs segmentation if needed,
        tokenizes and computes sentiment for each sentence, and redirects
        to the results page.

    :return: Rendered `index.html` page or redirect to `/results`
    :rtype: flask.Response
    """
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
                # Part of project requirements, checks strictly for strings with no spaces
                if " " not in content.strip():
                    content = smart_segment(content)

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
    """
    Route for displaying sentiment analysis results.

    This route processes precomputed sentiment scores, extracts:
    - The most positive and negative sentences
    - Fixed-size and dynamic sliding window summaries

    It also handles potential errors from any imported analysis functions.

    :query json_data: JSON string containing tokenized sentences and scores
    :query file_content: Encoded original text
    :return: Rendered `results.html` template with sentiment data or error message
    :rtype: flask.Response
    """
    json_data = request.args.get("json_data")
    file_content = request.args.get("file_content", "")
    # Decode content from URL
    file_content = urllib.parse.unquote(file_content)
    sentences_dict = json.loads(json_data)

    # Default values
    message_sentences_positive = ""
    message_sentences_negative = ""
    message_sliding_1 = ""
    message_sliding_2 = ""
    most_positive = ""
    most_negative = ""
    pos_sentence = neg_sentence = ""
    pos_fig = neg_fig = ""
    pos_extract = neg_extract = ""
    pos_extract_fig = neg_extract_fig = ""
    pos_extract2 = neg_extract2 = ""
    pos_extract_fig2 = neg_extract_fig2 = ""
    try:
        most_positive = most_positive_sentence(sentences_dict)
        pos_sentence = most_positive[1]
        pos_fig = sentiment_gauge(most_positive[0])
    except Exception:
        pos_sentence = most_positive[1]

    try:
        most_negative = most_negative_sentence(sentences_dict)
        neg_sentence = most_negative[1]
        neg_fig = sentiment_gauge(most_negative[0])
    except Exception:
        neg_sentence = most_negative[1]

    try:
        # Sliding window 1 (Fixed window size of 3)
        sw_result = sliding_window(sentences_dict)
        positive_para, negative_para = sw_result
        pos_extract = " ".join(positive_para[0])
        neg_extract = " ".join(negative_para[0])
        pos_extract_fig = sentiment_gauge(positive_para[1])
        neg_extract_fig = sentiment_gauge(negative_para[1])
    except Exception:
        pos_extract = sw_result
        neg_extract = sw_result

    try:
        # Sliding window 2 (No fixed window)
        sw2_result = sliding_window_2(sentences_dict)
        max_segments, min_segments = sw2_result
        most_positive_dict = max(max_segments, key=lambda d: len(d["sentence"])) if max_segments else {"sentence": "", "score": 0}
        most_negative_dict = max(min_segments, key=lambda d: len(d["sentence"])) if min_segments else {"sentence": "", "score": 0}
        pos_extract2 = most_positive_dict["sentence"].strip()
        neg_extract2 = most_negative_dict["sentence"].strip()
        pos_extract_fig2 = sentiment_gauge(most_positive_dict["score"])
        neg_extract_fig2 = sentiment_gauge(most_negative_dict["score"])
    except Exception:
        pos_extract2 = sw2_result
        neg_extract2 = sw2_result

    return render_template(
        "results.html",
        message_sentences_positive=message_sentences_positive,
        message_sentences_negative=message_sentences_negative,
        message_sliding_1=message_sliding_1,
        message_sliding_2=message_sliding_2,
        entire_text=file_content,
        pos_sentence=pos_sentence,
        neg_sentence=neg_sentence,
        pos_fig=pos_fig,
        neg_fig=neg_fig,
        pos_extract=pos_extract,
        pos_extract_fig=pos_extract_fig,
        neg_extract=neg_extract,
        neg_extract_fig=neg_extract_fig,
        pos_extract2=pos_extract2,
        pos_extract_fig2=pos_extract_fig2,
        neg_extract2=neg_extract2,
        neg_extract_fig2=neg_extract_fig2
    )
    
if __name__ == "__main__":
    app.run(debug=True)