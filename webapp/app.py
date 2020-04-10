from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np

# App definition
app = Flask(__name__, template_folder="templates")
app.config['TESTING'] = True

# Importing model supervised
with open('model/model_super.pkl', 'rb') as file:
    classifier_super = pickle.load(file)

# Importing model unsupervised
with open('model/model_nosup.pkl', 'rb') as file:
    classifier_nosuper = pickle.load(file)

# Importing vectorizer super
with open('model/vectorizer_super.pkl', 'rb') as file:
    vectorizer_super = pickle.load(file)

# Importing vectorizer no_super
with open('model/vectorizer_nosuper.pkl', 'rb') as file:
    vectorizer_nosuper = pickle.load(file)

# Importing MultiLabelBinarizer
with open('model/mlb.pkl', 'rb') as file:
    mlb = pickle.load(file)

# Importing Tag_features
with open('model/tag_features.pkl', 'rb') as file:
    tag_features = pickle.load(file)


@app.route('/', methods=['POST'])

@app.route('/predict')
def predict():
    json_ = request.get_json(force=True)
    data = json_normalize(json_)
    # concat title and body
    data["document"] = data["title"] + " " + data["body"]
    data.drop(columns=["title", "body"], inplace=True)

    # Cleaning
    lemma = WordNetLemmatizer()
    token = ToktokTokenizer()
    data["document"] = data["document"].apply(lambda x: BeautifulSoup(x, 'lxml').get_text())
    data["document"] = data["document"].apply(lambda x: clean_contract(x))
    data["document"] = data["document"].apply(lambda x: clean_punct(x, token))
    data["document"] = data["document"].apply(lambda x: clean_stop_word(x, token))
    data["document"] = data["document"].apply(lambda x: lemitize_words(x, token, lemma))

    # Supervised approach
    if data["type"][0] == "supervised":
        # Prediction
        x_tfidf = vectorizer_super.transform(data["document"])
        prediction_inv = classifier_super.predict(x_tfidf)
        prediction = mlb.inverse_transform(prediction_inv)

        return jsonify({
            "prediction": str(prediction)
        })
    # UnSupervised approach
    elif data["type"][0] == "nosupervised":
        tfidf_target = vectorizer_nosuper.transform(data["document"])
        target_nmf_trans = classifier_nosuper.transform(tfidf_target)
        tags_scores = np.dot(target_nmf_trans, classifier_nosuper.components_).tolist()[0]
        dict_tags_score = dict(zip(vectorizer_nosuper.get_feature_names(), tags_scores))
        prediction = { k: v for k, v in dict_tags_score.items() if v >= 0.1}
        return jsonify({
            "prediction": str(prediction)
        })

def clean_contract(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    text = re.sub(r"\'\xa0", " ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']


def clean_punct(text, token):
    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punctuation))
    remove_punctuation = str.maketrans(' ', ' ', punctuation)
    for w in words:
        if w in tag_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))


def clean_stop_word(text, token):
    stop_words = set(stopwords.words("english"))
    words = token.tokenize(text)

    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))


def lemitize_words(text, token, lemma):
    words = token.tokenize(text)
    listLemma = []
    for w in words:
        x = lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))


if __name__ == "__main__":
    app.run()
