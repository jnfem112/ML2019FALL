#!/bin/bash

mkdir data
wget -O data/trainX.npy https://www.dropbox.com/s/vw3a7tkrpweoj19/trainX.npy?dl=0
wget -O data/trainY.npy https://www.dropbox.com/s/z9gzhx5l4b9f94w/trainY.npy?dl=0
wget -O data/testX.npy https://www.dropbox.com/s/xkahpd5a5ezjs7n/testX.npy?dl=0
mkdir model
wget -O model/generator.pkl https://www.dropbox.com/s/to4owiaaghztpwy/generator.pkl?dl=0
wget -O model/classifier_1.pkl https://www.dropbox.com/s/cc8ier909c6ylyn/classifier_1.pkl?dl=0
wget -O model/classifier_2.pkl https://www.dropbox.com/s/dtyybtelrkbmi2n/classifier_2.pkl?dl=0
