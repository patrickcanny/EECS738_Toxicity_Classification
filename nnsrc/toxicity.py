#Based on: https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
#This is currently just a speculative approach we can talk about during our presentation/compare to ours
#If we go with this approach, potential improvement work includes:
#   -Getting training to work for the full set of 31 parameters in the new dataset
#    (the kernel above is for a previous, less complex kaggle competition on the same topic)
#   -Improving accuracy and training speed
#   -Experimenting with pre-trained embeddings and different NN architectures
import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping

def buildNeuralNet(numWords, commentLen):
    #input layer for an arbitrary number of comments of length commentLen
    inputLayer = Input(shape=(commentLen, ))
    #project words to coordinate vector space representing their co-appearance frequency
    x = Embedding(numWords, 128)(inputLayer)
    #stateful recurrent neural network
    x = LSTM(60, return_sequences = True, name = "lstm")(x)
    #reduce 3d vector space output to 2d
    x = GlobalMaxPool1D()(x)
    #drop 1/10 nodes to improve generalization
    x = Dropout(rate=.9)(x)
    #standard nn
    x = Dense(50, activation='relu')(x)
    #drop again
    x = Dropout(rate=.9)(x)
    #standard nn, sigmoid for values between 0 and 1
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inputLayer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def main(): 
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    subm = pd.read_csv('sample_submission.csv')

    label_cols = ['target',
                    'severe_toxicity',
                    'obscene',
                    'threat',
                    'insult',
                    'identity_attack']
                    # 'sexual_explicit',
                    # 'male',
                    # 'female',
                    # 'transgender',
                    # 'other_gender',
                    # 'heterosexual',
                    # 'homosexual_gay_or_lesbian',
                    # 'bisexual',
                    # 'other_sexual_orientation',
                    # 'christian',
                    # 'jewish',
                    # 'muslim',
                    # 'hindu',
                    # 'buddhist',
                    # 'atheist',
                    # 'other_religion',
                    # 'black',
                    # 'white',
                    # 'asian',
                    # 'latino',
                    # 'other_race_or_ethnicity',
                    # 'physical_disability',
                    # 'intellectual_or_learning_disability',
                    # 'psychiatric_or_mental_illness',
                    # 'other_disability']
    y = train[label_cols].values
    trainComments = train["comment_text"]
    testComments = test["comment_text"]

    uniqueWords = 20000
    tok = Tokenizer(num_words=uniqueWords)
    tok.fit_on_texts(list(trainComments))
    trainComments_tokenized = tok.texts_to_sequences(trainComments)
    testComments_tokenized = tok.texts_to_sequences(testComments)
    commentLen = 200
    X_t = pad_sequences(trainComments_tokenized, maxlen=commentLen)
    X_te = pad_sequences(testComments_tokenized, maxlen=commentLen)

    net = buildNeuralNet(uniqueWords, commentLen)
    earlyStoppingCallback = [EarlyStopping(monitor='val_loss', min_delta=0, mode='min', restore_best_weights=True)]
    net.fit(X_t, y, batch_size=32, epochs=2, validation_split=.1, callbacks=earlyStoppingCallback)
    netJson = net.to_json()
    with open("neuralNet.json", "w") as json_file:
        json_file.write(netJson)
        # serialize weights to HDF5
        net.save_weights("netWeights.h5")


main()