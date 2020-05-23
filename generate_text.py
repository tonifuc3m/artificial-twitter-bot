#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from load_text import load
from network import Network
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def train_network(filename, weights_path, log_path):
    ################# LOAD ###################### 
    #filename = 'data/lope.txt'
    X, y, _, _, n_vocab = load(filename)
    
    
    ####################### Train ####################
    ## Define network
    model = Network(n_vocab)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # define the checkpoint
    csvlogger = CSVLogger(log_path, separator=',', append=False)
    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint, csvlogger]
    
    ## Train
    model.fit(X, y, epochs=20, batch_size=256, callbacks=callbacks_list)

def generate_text(filename, weights_path):
    ################# LOAD ###################### 
    X, _, chars, dataX, n_vocab = load(filename)
    
    ################### Generate text ####################
    ## Reload network
    model = Network(n_vocab)
    model.build(X.shape)
    # load the network weights
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    ## Integer embedding
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    
    ## Generate text
    # pick a random seed
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    initial_sentence = ''.join([int_to_char[value] for value in pattern])
    print("Seed:")
    print("\"", initial_sentence, "\"")
    # generate characters
    tweet_len = 140
    result = ''
    for i in range(tweet_len):
      x = np.reshape(pattern, (1, len(pattern), 1))
      x = x / float(n_vocab)
      prediction = model.predict(x, verbose=0)
      index = np.argmax(prediction)
      result = result + int_to_char[index]
      pattern.append(index)
      pattern = pattern[1:len(pattern)]
    print(initial_sentence + result)

    return initial_sentence + result
