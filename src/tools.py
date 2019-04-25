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
        self.stop_words = None

        self.all_comments = None
        self.non_toxic_comments = None
        self.toxic_comments = None

        self.comments_scores = None
        self.non_toxic_comments_scores = None
        self.toxic_comments_scores = None

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
        self.getToxicComments()

    # Extract comment and toxicity score from all entries in the dataframe that
    # is being observed. Save this extracted data in self.comments_scores
    def processTraining(self):
        print("Processing training data...")
        cols = ['target', 'comment_text']
        comments_scores = self.data[cols]
        comments_scores['comment_text'] = self.processCommentList(comments_scores['comment_text'])
        self.comments_scores = comments_scores
        self.all_comments = comments_scores['comment_text']

    # Convert all comments from an iterable to lower case, remove punctuation
    def processCommentList(self, comments):
        print("Removing Punctuation...")
        punct = string.punctuation.replace('\'','')+"0123456789"
        outtab = "                                         "
        trantab = str.maketrans(punct, outtab)
        processed = []
        for comment in comments:
            comment = comment.lower().translate(trantab)
            comment = (comment.encode('ascii', 'ignore')).decode("utf-8")
            processed.append(comment)
        return processed

    # Generate a stop word dictionary to be used by other methods. This uses
    # the english default stop words as well as all words from comments that
    # are identified as completely non-toxic
    def buildStopWordDictionary(self):
        print("Building Stop Word Dictionary...")
        stop_words = get_stop_words('english')
        stop_words.append('')
        for x in range(ord('b'), ord('z') + 1):
            stop_words.append(chr(x))

        self.stop_words = stop_words

    # Extract all comments that are completely non-toxic i.e. they have a
    # target score of 0.0
    def getGoodComments(self):
        print("About to get all good comments")
        good_comments = self.comments_scores[self.comments_scores['target'] == 0.0]
        self.non_toxic_comments = good_comments['comment_text']
        self.non_toxic_comments_scores = good_comments

    # Get every word in every good comment
    def getGoodWords(self):
        good_words = set()
        for comment in self.non_toxic_comments:
            for word in comment.split():
                good_words.add(word)
        return list(good_words)

    # Retrieve all comments with a toxicity score above 0.5 i.e. the most toxic
    # comments
    def getToxicComments(self):
        toxic_comments = good_comments = self.comments_scores[self.comments_scores['target'] >= 0.5]
        self.toxic_comments = toxic_comments['comment_text']
        self.toxic_comments_scores = toxic_comments

    def getAllComments(self):
        return self.all_comments

    def getMyToxicComments(self):
        return self.toxic_comments

    def getMyNonToxicComments(self):
        return self.non_toxic_comments
    def getProcessedComments(self):
        return self.processCommentList(self.all_comments)
            
    def getStops(self):
        return self.stop_words
