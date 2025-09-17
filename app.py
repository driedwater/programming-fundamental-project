from flask import Flask, render_template, request
from preprocessing import complete_tokenization
from sentiment_analysis import compute_all_sentences
from SentimentAnalysis_C import most_positive_sentence, most_negative_sentence
from SlidingWindow import sliding_window

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
                print(compute_all_sentences(tokens))
                print(most_positive_sentence(compute_all_sentences(tokens)))
                print(most_negative_sentence(compute_all_sentences(tokens)))
                print(sliding_window(compute_all_sentences(tokens),1))
            # Added try except to handle decoding errors (tested with corrupt .txt file)
            except Exception:
                message = "Error reading file. Make sure it's a valid text file."

    return render_template("index.html", message=message, content=content)

if __name__ == "__main__":
    app.run(debug=True)




