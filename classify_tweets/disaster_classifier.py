# import keras
# from keras.preprocessing import sequence, text
from typing import List, Any

import keras
import tensorflow as tf
from keras.preprocessing import sequence, text
import numpy as np
from keras.models import model_from_json, load_model

import pickle


def classify_disaster_tweets(tweet):

    if tweet == "":
        return "Please enter the tweet!", None

    max_len = 1500
    print(tweet)

    json_file = open('lstm_model_arch.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("lstm_model.h5")
    print("Loaded model from disk")

    with open('lstm_tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    seq = loaded_tokenizer.texts_to_sequences([tweet])
    padded = sequence.pad_sequences(seq, maxlen=max_len)
    pred = loaded_model.predict_classes(padded)

    classification = "DISASTER" if pred[0][0] == 1 else "NON_DISASTER"

    print("Tweet: ", tweet)
    print("Prediction: ", pred[0][0])
    print("Classification: ", classification )

    return tweet, classification


tweet = "Just happened a terrible car crash"
classify_disaster_tweets(tweet)
