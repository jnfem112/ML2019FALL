import sys
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset , DataLoader
import MCD
import torch.nn as nn
import torch.nn.functional as F
from time import time

class dataset(Dataset):
	def __init__(self , data):
		self.data = data
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		return torch.FloatTensor(self.data[index])

def load_data():
	temp_x = np.load(sys.argv[1])
	test_x = list()
	number_of_data = temp_x.shape[0]
	for i in range(number_of_data):
		image = cv2.resize(temp_x[i] , (32 , 32) , cv2.INTER_LINEAR)
		test_x.append(image)
	test_x = np.array(test_x)
	test_x = np.expand_dims(test_x , axis = 1)
	test_x = test_x / 255
	return test_x

def test(test_x , generator , classifier_1 , classifier_2 , device):
	# Hyper-parameter.
	batch_size = 1024

	test_dataset = dataset(test_x)
	test_loader = DataLoader(test_dataset , batch_size = batch_size , shuffle = False)

	(generator , classifier_1 , classifier_2) = (generator.to(device) , classifier_1.to(device) , classifier_2.to(device))
	generator.eval()
	classifier_1.eval()
	classifier_2.eval()
	predict = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			feature = generator(data)
			y_1 = classifier_1(feature)
			y_2 = classifier_2(feature)
			y = (y_1 + y_2) / 2
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
	(generator , classifier_1 , classifier_2) = (MCD.generator() , MCD.classifier() , MCD.classifier())
	generator.load_state_dict(torch.load('model/generator.pkl' , map_location = device))
	classifier_1.load_state_dict(torch.load('model/classifier_1.pkl' , map_location = device))
	classifier_2.load_state_dict(torch.load('model/classifier_2.pkl' , map_location = device))
	test_y = test(test_x , generator , classifier_1 , classifier_2 , device)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()
