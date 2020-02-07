import sys
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset , DataLoader
import PyTorch_CNN as CNN

class dataset(Dataset):
	def __init__(self , directory , number_of_data):
		self.directory = directory
		self.number_of_data = number_of_data
		return

	def __len__(self):
		return self.number_of_data

	def __getitem__(self , index):
		image = cv2.imread(os.path.join(self.directory , '{:0>4d}.jpg'.format(index)) , cv2.IMREAD_GRAYSCALE)
		image = np.expand_dims(image , axis = 0)
		image = image / 255
		return torch.FloatTensor(image)

def test(directory , model , device):
	# Hyper-parameter.
	batch_size = 1024

	test_dataset = dataset(directory , len(os.listdir(directory)))
	test_loader = DataLoader(test_dataset , batch_size = batch_size , shuffle = False)

	model.to(device)
	model.eval()
	predict = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			y = model(data)
			(_ , result) = torch.max(y , dim = 1)
			predict.append(result.cpu().detach().numpy())
	predict = np.concatenate(predict , axis = 0)

	return predict

def dump(predict):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = CNN.CNN_2()
	model.load_state_dict(torch.load('PyTorch_CNN.pkl' , map_location = device))
	predict = test(sys.argv[1] , model , device)
	dump(predict)
	return

if (__name__ == '__main__'):
	main()
