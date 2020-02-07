import sys
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset , DataLoader
import PyTorch_CNN as CNN
from torch.optim import Adam
import torch.nn.functional as F
from time import time

class dataset(Dataset):
	def __init__(self , directory , label , transform , apply_transform):
		self.directory = directory
		self.label = label
		self.transform = transform
		self.apply_transform = apply_transform
		return

	def __len__(self):
		return self.label.shape[0]

	def __getitem__(self , index):
		image = cv2.imread(os.path.join(self.directory , '{:0>5d}.jpg'.format(self.label[index][0])) , cv2.IMREAD_GRAYSCALE)
		if (self.apply_transform):
			image = Image.fromarray(np.uint8(image))
			image = self.transform(image)
			image = np.asarray(image)
		image = np.expand_dims(image , axis = 0)
		image = image / 255
		return (torch.FloatTensor(image) , self.label[index][1])

def load_data():
	df = pd.read_csv(sys.argv[2])
	label = df.values

	number_of_data = label.shape[0]
	index = np.arange(number_of_data)
	np.random.shuffle(index)
	label = label[index]
	validation_label = label[ : 2000]
	train_label = label[2000 : ]

	return (train_label , validation_label)

def train(directory , train_label , validation_label , model , device):
	# Hyper-parameter.
	batch_size = 1024
	learning_rate = 0.0001
	epoch = 2000

	transform = transforms.Compose([
		transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
		transforms.RandomHorizontalFlip()
	])

	train_dataset = dataset(directory , train_label , transform , True)
	validation_dataset = dataset(directory , validation_label , None , False)
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
				m = int(50 * n / train_label.shape[0])
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , train_label.shape[0] , bar) , end = '\r')
			else:
				n = train_label.shape[0]
				bar = 50 * '='
				end = time()
				print('epoch {}/{} ({}/{}) [{}] ({}s) loss : {:.8f}'.format(i + 1 , epoch , n , train_label.shape[0] , bar , int(end - start) , total_loss / train_label.shape[0]))

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
			validation_accuracy = accuracy(validation_label[ : , 1] , predict)
			end = time()
			print('evaluation ({}s) validation accuracy : {:.5f}'.format(int(end - start) , validation_accuracy))

	return model

def accuracy(label , predict):
	return np.sum(predict == label) / label.shape[0]

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_label , validation_label) = load_data()
	model = CNN.CNN_2()
	model = train(sys.argv[1] , train_label , validation_label , model , device)
	torch.save(model.state_dict() , 'PyTorch_CNN.pkl')
	return

if (__name__ == '__main__'):
	main()
