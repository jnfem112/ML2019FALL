import sys
import os
import pandas as pd
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset , DataLoader
import DNN
from torch.optim import Adam

def load_data():
	NLP = spacy.load('en_core_web_lg')
	vocabulary = np.load('vocabulary.npy')

	df = pd.read_csv(sys.argv[1])
	data = df['comment'].values
	test_x = list()
	for comment in data:
		BOW = np.zeros(vocabulary.shape[0])
		for token in NLP(comment):
			index = np.where(vocabulary == token.text)[0][0]
			BOW[index] = 1
		test_x.append(BOW)
	test_x = np.array(test_x)

	return torch.FloatTensor(test_x)

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
	test_x = load_data()
	model = DNN.DNN_0(test_x.shape[1])
	model.load_state_dict(torch.load('DNN.pkl' , map_location = device))
	test_y = test(test_x , model , device)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()
