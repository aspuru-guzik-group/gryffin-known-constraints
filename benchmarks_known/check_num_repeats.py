#! /usr/bin/env python

import pickle

with open('results.pkl', 'rb') as content:
    data = pickle.load(content)

print(len(data))
