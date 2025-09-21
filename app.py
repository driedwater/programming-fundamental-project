import json

from flask import Flask, render_template, request, redirect, url_for
from preprocessing import complete_tokenization
from sentiment_analysis import compute_all_sentences
from SentimentAnalysis_C import most_positive_sentence, most_negative_sentence
from SlidingWindow import sliding_window
from chart import pos_figure,neg_figure,pos_extract_figure,neg_extract_figure

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
                tokens = complete_tokenization(content)
                sentences_dict = compute_all_sentences(tokens)
                json_data = json.dumps(sentences_dict)

                return redirect(url_for("results", json_data=json_data))
            # Added try except to handle decoding errors (tested with corrupt .txt file)
            except Exception:
                message = "Error reading file. Make sure it's a valid text file."

    return render_template("index.html", message=message, content=content)

@app.route('/display')
def results():
    json_data = request.args.get("json_data")
    sentences_dict = json.loads(json_data)

    most_positive = most_positive_sentence(sentences_dict)
    most_negative = most_negative_sentence(sentences_dict)
    positive_para, negative_para = sliding_window(sentences_dict)

    return render_template(
        "display.html",
        pos_sentence=most_positive[1],
        neg_sentence=most_negative[1],
        pos_fig=pos_figure(most_positive[0]),
        neg_fig=neg_figure(most_negative[0]),
        pos_extract_fig=pos_extract_figure(positive_para[1]),
        pos_extract=positive_para[0],
        neg_extract_fig=neg_extract_figure(negative_para[1]),
        neg_extract=negative_para[0]
    )

if __name__ == "__main__":
    app.run(debug=True)




