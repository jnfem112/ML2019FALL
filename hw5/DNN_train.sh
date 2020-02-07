#!/bin/bash

time python3 construct_vocabulary.py $1 $3
time python3 DNN_train.py $1 $2
