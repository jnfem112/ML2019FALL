import sys
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset , DataLoader
import Autoencoder
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from time import time

class dataset(Dataset):
	def __init__(self , data , transform):
		self.data = data
		self.transform = transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		data = self.data[index]
		data = Image.fromarray(np.uint8(data))
		data = self.transform(data)
		data = np.asarray(data)
		data = np.transpose(data , (2 , 0 , 1))
		data = data / 255
		data = 2 * data - 1
		return torch.FloatTensor(data)

def load_data():
	train_x = np.load(sys.argv[1])
	return train_x

def train(train_x , model , device):
	# Hyper-parameter.
	batch_size = 256
	learning_rate = 0.0001
	epoch = 20

	transform = transforms.Compose([
		transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
		transforms.RandomHorizontalFlip()
	])

	train_dataset = dataset(train_x , transform)
	train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True)

	model.to(device)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.L1Loss()
	for i in range(epoch):
		start = time()
		model.train()
		total_loss = 0
		for (j , data) in enumerate(train_loader):
			data = data.to(device)
			optimizer.zero_grad()
			(encode , decode) = model(data)
			loss = criterion(data , decode)
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

	return model

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_x = load_data()
	model = Autoencoder.Autoencoder_1()
	model = train(train_x , model , device)
	torch.save(model.state_dict() , 'model.pkl')
	return

if (__name__ == '__main__'):
	main()