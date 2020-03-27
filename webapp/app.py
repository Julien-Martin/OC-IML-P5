from flask import Flask, render_template, request, jsonify
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import re
from sklearn.feature_extraction.text import CountVectorizer

# App definition
app = Flask(__name__, template_folder="templates")
app.config['TESTING'] = True

# Importing model
with open('model/model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Importing vectorizer
with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Importing MultiLabelBinarizer
with open('model/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def main():
    return "Stack Overflow Tag prediction by Julien Martin"


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "GET":
        return "Prediction page"
    if request.method == "POST":
        json_ = request.json
        data = json_normalize(json_)
        # concat title and body
        data["document"] = data["title"] + " " + data["body"]
        data.drop(columns=["title", "body"], inplace=True)
        X_tfidf = vectorizer.transform(data)
        prediction = classifier.predict(X_tfidf)
        return jsonify({
            "prediction": str(prediction)
        })


if __name__ == "__main__":
    app.run()
