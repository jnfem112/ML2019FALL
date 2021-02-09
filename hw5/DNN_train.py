import sys
import os
import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset , DataLoader
import DNN
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
from time import time

class dataset(Dataset):
	def __init__(self , data , label):
		self.data = data
		self.label = label
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return (torch.FloatTensor(self.data[index]) , self.label[index])

def load_data():
	NLP = spacy.load('en_core_web_lg')
	vocabulary = np.load('vocabulary.npy')

	df = pd.read_csv(sys.argv[1])
	data = df['comment'].values
	train_x = list()
	for comment in data:
		BOW = np.zeros(vocabulary.shape[0])
		for token in NLP(comment):
			index = np.where(vocabulary == token.text)[0][0]
			BOW[index] = 1
		train_x.append(BOW)
	train_x = np.array(train_x)

	df = pd.read_csv(sys.argv[2])
	train_y = df['label'].values

	validation_x = train_x[ : 2000]
	validation_y = train_y[ : 2000]
	train_x = train_x[2000 : ]
	train_y = train_y[2000 : ]

	return (train_x , train_y , validation_x , validation_y)

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
	(train_x , train_y , validation_x , validation_y) = load_data()
	model = DNN.DNN_0(train_x.shape[1])
	model = train(train_x , train_y , validation_x , validation_y , model , device)
	torch.save(model.state_dict() , 'DNN.pkl')
	return

if (__name__ == '__main__'):
	main()
