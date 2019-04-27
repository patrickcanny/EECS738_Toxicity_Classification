import tools
import pandas as pd

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from nltk.corpus import sentiwordnet as swn
class CommentPredictor:

    def __init__(self, tools):
        self.tools = tools
        self.word_weight_dict = {}
        self.predictions = None

        self.initializeWeights()
        self.predictAll()

    def initializeWeights(self):
        self.word_weight_dict[' '] = 0.0
        words = self.getEveryWord()
        toxic_words = self.getToxicWords()
        non_toxic_words = self.getNonToxicWords()
        stop_words = self.tools.getStops()
        for word in words:
            weight = 0.0
            if word in toxic_words:
                weight = 1.0
                if word in non_toxic_words:
                    weight = .5
            if word in stop_words:
                weight = 0
            self.word_weight_dict[word] = weight

    def getPreds(self):
        return self.predictions

    def predictAll(self):
        scores = []
        comments = self.tools.getAllComments()
        predict = {}
        for comment in comments:
            scores.append(self.predictForComment(comment))
        predict = {'target':scores, 'comment_text':comments}
        self.predictions = pd.DataFrame(predict)

    def predictForComment(self, comment):
        score = 0.0
        for word in comment:
            try:
                score += self.word_weight_dict[word]
            except:
                score += 0
        return score

    def getEveryWord(self):
        comments = self.tools.getAllComments()
        return self.getAllWords(comments)

    def getToxicWords(self):
        comments = self.tools.getMyToxicComments()
        return self.getAllWords(comments)

    def getNonToxicWords(self):
        comments = self.tools.getMyNonToxicComments()
        return self.getAllWords(comments)

    def getAllWords(self, comments):
        s = set()
        for comment in comments:
            for word in comment.split():
                s.add(word)
        return s
    def buildNeuralNet(self, numWords, commentLen):
        #input layer for an arbitrary number of comments of length commentLen
        inputLayer = Input(shape=(commentLen, ))    
        #drop 1/10 nodes to improve generalization
        x = Dropout(rate=.9)(inputLayer)
        #standard nn
        x = Dense(50, activation='relu')(x)
        #drop again
        x = Dropout(rate=.9)(x)
        #standard nn, sigmoid for values between 0 and 1
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputLayer, outputs=x)
        adamOpt = optimizers.Adam(lr=.0001)
        model.compile(loss='binary_crossentropy', optimizer=adamOpt, metrics=['accuracy'])
        model.summary()
        return model

