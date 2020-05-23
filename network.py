#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:04:14 2020

@author: antonio
"""

import tensorflow as tf

class Network(tf.keras.Model):
    def __init__(self, n_vocab):
        # call the parent constructor
        super(Network, self).__init__()
        
        # Define layers
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.d1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(256)
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(units=n_vocab)
        self.activation = tf.keras.layers.Activation('softmax')
        
    # Define forward pass
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.d1(x)
        x = self.lstm2(x)
        x = self.d2(x)
        x = self.out(x)
        return self.activation(x)