import sys
import os
import pandas as pd
import numpy as np
import spacy
from emoji import UNICODE_EMOJI
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset , DataLoader
import RNN

class Vocabulary():
	def __init__(self):
		W2Vmodel = Word2Vec.load('Word2Vec.model')
		self.word2vector = {token : W2Vmodel.wv[token] for token in W2Vmodel.wv.vocab}
		self.word2vector['<PAD>'] = np.zeros(256)
		self.word2vector['<UNK>'] = np.zeros(256)
		self.token2index = {token : index for (index , token) in enumerate(self.word2vector)}
		self.index2token = {index : token for (index , token) in enumerate(self.word2vector)}
		self.embedding = torch.FloatTensor([self.word2vector[token] for token in self.word2vector])
		return

def preprocess(data , vocabulary):
	def valid(token):
		return not (token.is_punct or token.is_stop or (token.text in UNICODE_EMOJI) or any(not (character.isdigit() or character.isalpha()) for character in token.text))

	def comment2index(comment , vocabulary):
		return [vocabulary.token2index[token] if (token in vocabulary.word2vector) else vocabulary.token2index['<UNK>'] for token in comment]

	def trim_and_pad(comment , max_length , vocabulary):
		comment = comment[ : min(len(comment) , max_length)]
		comment += (max_length - len(comment)) * [vocabulary.token2index['<PAD>']]
		return comment

	NLP = spacy.load('en_core_web_lg')

	data = [[token.lemma_ for token in NLP(comment) if valid(token)] for comment in data]
	data = [comment2index(comment , vocabulary) for comment in data]
	data = [trim_and_pad(comment , 16 , vocabulary) for comment in data]

	return np.array(data)

def load_data():
	vocabulary = Vocabulary()

	df = pd.read_csv(sys.argv[1])
	test_x = df['comment'].values
	test_x = preprocess(test_x , vocabulary)

	return (torch.tensor(test_x) , vocabulary)

def test(test_x , model , device):
	# Hyper-parameter.
	batch_size = 1024

	test_loader = DataLoader(test_x , batch_size = batch_size , shuffle = False)

	model.to(device)
	model.eval()
	predict = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			y = model(data)
			(_ , result) = torch.max(y , dim = 1)
			predict.append(result.cpu().detach().numpy())
	test_y = np.concatenate(predict , axis = 0)

	return test_y

def dump(test_y):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : test_y})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(test_x , vocabulary) = load_data()
	model = RNN.RNN_0(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model.load_state_dict(torch.load('RNN.pkl' , map_location = device))
	test_y = test(test_x , model , device)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()
