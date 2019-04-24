# tools.py
# module for data processing
import pandas as pd
import numpy as np
import os
import matplotlib as plt
import string
from stop_words import get_stop_words
import nltk
import ssl
from sklearn.feature_extraction.text import CountVectorizer

class tools:
    # Entrypoint for tools. Unique Identifier for a tools object is the
    # filepath to the dataset being observed by the tools.
    def __init__(self, filepath):
        self.filepath = None
        self.data = None
        self.data_shape = None
        self.comments_scores = None
        self.stop_words = None
        self.non_toxic_comments = None

        self.readData(filepath)
        self.processData()

    # Reads data from a given filepath into a pandas dataframe, then drops
    # entries from that dataframe that contain NaNs.
    def readData(self, filepath):
        df = pd.read_csv(filepath)
        df = df.dropna()
        self.data_shape = df.shape
        self.data = df
        self.filepath = filepath

    # Entrypoint for data processing.
    def processData(self):
        self.processTraining()
        self.buildStopWordDictionary()
        self.getGoodComments()

    # Extract comment and toxicity score from all entries in the dataframe that
    # is being observed. Save this extracted data in self.comments_scores
    def processTraining(self):
        cols = ['target', 'comment_text']
        comments_scores = df[cols]
        comments_scores['comment_text'] = self.processCommentList(comments_scores['comment_text'])
        self.comments_scores = comments_scores

    # Convert all comments from an iterable to lower case, remove punctuation
    def processCommentList(self, comments):
        punct = string.punctuation.replace('\'','')+"0123456789"
        outtab = "                                         "
        trantab = str.maketrans(punctuation_edit, outtab)
        for comment in comments:
            comment = comment.lower.translate(trantab)

    # Generate a stop word dictionary to be used by other methods. This uses
    # the english default stop words as well as all words from comments that
    # are identified as completely non-toxic
    def buildStopWordDictionary(self):
        stop_words = get_stop_words('english')
        stop_words.append('')
        for x in range(ord('b'), ord('z') + 1):
            stop_words.append(chr(x))

        stop_words += self.getGoodWords()
        self.stop_words = stop_words

    # Extract all comments that are completely non-toxic i.e. they have a
    # target score of 0.0
    def getGoodComments(self):
        good_comments = self.data[self.data['target'] == 0.0]
        self.non_toxic_comments = good_comments['comment_text']

    # Get every word in every good comment
    def getGoodWords(self):
        good_words = set()
        for comment in self.non_toxic_comments:
            for word in comment.split():
                good_words.append(word)
        return list(good_words)
