import sys
import pandas as pd
import spacy
from emoji import UNICODE_EMOJI
from gensim.models import Word2Vec

def construct_vocabulary():
	NLP = spacy.load('en_core_web_lg')

	def valid(token):
		return not (token.is_punct or token.is_stop or (token.text in UNICODE_EMOJI) or any(not (character.isdigit() or character.isalpha()) for character in token.text))

	df = pd.read_csv(sys.argv[1])
	data = df['comment'].values
	train_x = [[token.lemma_ for token in NLP(comment) if valid(token)] for comment in data]

	df = pd.read_csv(sys.argv[2])
	data = df['comment'].values
	test_x = [[token.lemma_ for token in NLP(comment) if valid(token)] for comment in data]

	vocabulary = train_x + test_x
	return vocabulary

def main():
	vocabulary = construct_vocabulary()
	W2Vmodel = Word2Vec(vocabulary , size = 256 , sg = 1 , min_count = 5 , iter = 200)
	W2Vmodel.save('Word2Vec.model')
	return

if (__name__ == '__main__'):
	main()
