#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tweepy as tp

def start_connection():
    # credentials to login to twitter api
    filename = 'twitter-bot-details.txt'
    lines = open(filename, 'r').readlines()
    consumer_key = lines[2].split(':')[-1].strip()
    consumer_secret = lines[3].split(':')[-1].strip()
    access_token = lines[4].split(':')[-1].strip()
    access_secret = lines[5].split(':')[-1].strip()
    
    # login to twitter account api
    auth = tp.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tp.API(auth)
    
    return api
    
def post_once(api, sentence):
    # Post once
    api.update_status(sentence)
