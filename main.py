#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:23:07 2020

@author: antonio
"""

from generate_text import train_network, generate_text
from post_tweets import start_connection, post_once
import argparse

def argparser():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument("-m", "--mode", required = True, dest = "mode", 
                        choices=['train', 'generate'])
    parser.add_argument("-f", "--filename", required = True, dest = "filename", 
                        help = "Route to train data")       
    args = parser.parse_args()
    return args.mode, args.filename


if __name__ == '__main__':
    # Parse arguments
    mode, filename = argparser()
    weights_path = "weights.hdf5"
    
    if mode == 'train':
        # Train network
        log_path = 'wModel-baseline-log.csv'
        train_network(filename, weights_path, log_path)
        
    if mode == 'generate':
        # Generate text with NN
        text = generate_text(filename, weights_path)
        
        # Connect to Twitter account
        api = start_connection
        
        # Post
        post_once(api, text)
        
