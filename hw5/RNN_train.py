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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
from time import time

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

class dataset(Dataset):
	def __init__(self , data , label):
		self.data = data
		self.label = label
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return (torch.tensor(self.data[index]) , self.label[index])

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
	train_x = df['comment'].values
	train_x = preprocess(train_x , vocabulary)

	df = pd.read_csv(sys.argv[2])
	train_y = df['label'].values

	number_of_data = train_x.shape[0]
	index = np.arange(number_of_data)
	np.random.shuffle(index)
	train_x = train_x[index]
	train_y = train_y[index]
	validation_x = train_x[ : 2000]
	validation_y = train_y[ : 2000]
	train_x = train_x[2000 : ]
	train_y = train_y[2000 : ]

	return (train_x , train_y , validation_x , validation_y , vocabulary)

def train(train_x , train_y , validation_x , validation_y , model , device):
	# Hyper-parameter.
	batch_size = 1024
	learning_rate = 0.0001
	epoch = 50

	train_dataset = dataset(train_x , train_y)
	validation_dataset = dataset(validation_x , validation_y)
	train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True)
	validation_loader = DataLoader(validation_dataset , batch_size = batch_size , shuffle = False)

	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	for i in range(epoch):
		start = time()
		model.train()
		total_loss = 0
		for (j , (data , label)) in enumerate(train_loader):
			(data , label) = (data.to(device) , label.to(device))
			optimizer.zero_grad()
			y = model(data)
			loss = F.cross_entropy(y , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()

			if (j < len(train_loader) - 1):
				n = (j + 1) * batch_size
				m = int(50 * n / train_x.shape[0])
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , train_x.shape[0] , bar) , end = '\r')
			else:
				n = train_x.shape[0]
				bar = 50 * '='
				end = time()
				print('epoch {}/{} ({}/{}) [{}] ({}s) loss : {:.8f}'.format(i + 1 , epoch , n , train_x.shape[0] , bar , int(end - start) , total_loss / train_x.shape[0]))

		if ((i + 1) % 10 == 0):
			start = time()
			model.eval()
			predict = list()
			with torch.no_grad():
				for (data , label) in validation_loader:
					data = data.to(device)
					y = model(data)
					(_ , result) = torch.max(y , dim = 1)
					predict.append(result.cpu().detach().numpy())
			predict = np.concatenate(predict , axis = 0)
			validation_score = f1_score(validation_y , predict , average = 'micro')
			end = time()
			print('evaluation ({}s) validation F1-score : {:.5f}'.format(int(end - start) , validation_score))

	return model

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , validation_x , validation_y , vocabulary) = load_data()
	model = RNN.RNN_0(vocabulary.embedding , vocabulary.token2index['<PAD>'])
	model = train(train_x , train_y , validation_x , validation_y , model , device)
	torch.save(model.state_dict() , 'RNN.pkl')
	return

if (__name__ == '__main__'):
	main()
