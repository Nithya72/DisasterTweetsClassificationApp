from flask import Flask, render_template, request
from classify_tweets import disaster_classifier

app = Flask(__name__)


@app.route("/tweets_classify", methods=["POST"])
def tweets_classify():
    tweet = request.form['tweet']
    tweet, classification = disaster_classifier.classify_disaster_tweets(tweet)
    return render_template("index.html", title="Tweets Classification", tweet=tweet, classification=classification)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == '__main__':
    app.run()
