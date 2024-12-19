# Library imports
import pandas as pd
import numpy as np
import re
import nltk
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load trained Pipeline
model = joblib.load('Intensity_Analysis.pkl')

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
from custom_tokenizer_function import CustomTokenizer


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    prediction = model.predict([text])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)



if __name__ == "__main__":
    app.run(debug=True)
