#!/bin/bash

time python3 Word2Vec_train.py $1 $3
time python3 RNN_train.py $1 $2
