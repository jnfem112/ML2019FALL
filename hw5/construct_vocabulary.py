import sys
import pandas as pd
import numpy as np
import spacy

def construct_vocabulary():
	NLP = spacy.load('en_core_web_lg')
	vocabulary = set()

	df = pd.read_csv(sys.argv[1])
	data = df['comment'].values
	for comment in data:
		for token in NLP(comment):
			vocabulary.add(token.text)

	df = pd.read_csv(sys.argv[2])
	data = df['comment'].values
	for comment in data:
		for token in NLP(comment):
			vocabulary.add(token.text)

	vocabulary = np.array(list(vocabulary))

	return vocabulary

def main():
	vocabulary = construct_vocabulary()
	np.save('vocabulary.npy' , vocabulary)
	return

if (__name__ == '__main__'):
	main()
