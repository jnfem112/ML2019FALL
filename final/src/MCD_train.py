import sys
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset , DataLoader
import MCD
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from time import time

class dataset(Dataset):
	def __init__(self , data , label , return_label , transform , apply_transform):
		self.data = data
		self.label = label
		self.return_label = return_label
		self.transform = transform
		self.apply_transform = apply_transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		data = self.data[index]
		if (self.apply_transform):
			data = Image.fromarray(np.uint8(data))
			data = self.transform(data)
			data = np.asarray(data)
		data = np.expand_dims(data , axis = 0)
		data = data / 255
		return (torch.FloatTensor(data) , self.label[index]) if (self.return_label) else torch.FloatTensor(data)

def load_data():
	temp_x = np.load(sys.argv[1])
	train_x = list()
	number_of_data = temp_x.shape[0]
	for i in range(number_of_data):
		image = cv2.cvtColor(temp_x[i] , cv2.COLOR_BGR2GRAY)
		image = cv2.Canny(image , 250 , 300)
		train_x.append(image)
	train_x = np.array(train_x)

	train_y = np.load(sys.argv[2])

	temp_x = np.load(sys.argv[3])
	test_x = list()
	number_of_data = temp_x.shape[0]
	for i in range(number_of_data):
		image = cv2.resize(temp_x[i] , (32 , 32) , cv2.INTER_LINEAR)
		test_x.append(image)
	test_x = np.array(test_x)

	return (train_x , train_y , test_x)

def discrepancy(output_1 , output_2):
	return torch.mean(torch.abs(F.softmax(output_1 , dim = 1) - F.softmax(output_2 , dim = 1)))

def train(train_x , train_y , test_x , generator , classifier_1 , classifier_2 , device):
	# Hyper-parameter.
	batch_size = 128
	learning_rate = 0.00002
	weight_decay = 0.0005
	epoch = 2000

	transform = transforms.Compose([
		transforms.RandomAffine(10 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
		transforms.RandomHorizontalFlip()
	])

	train_dataset = dataset(train_x , train_y , True , transform , True)
	test_dataset = dataset(test_x , None , False , transform , True)
	train_loader = DataLoader(train_dataset , batch_size = batch_size , shuffle = True)
	test_loader = DataLoader(test_dataset , batch_size = batch_size , shuffle = True)

	(generator , classifier_1 , classifier_2) = (generator.to(device) , classifier_1.to(device) , classifier_2.to(device))
	(optimizer_generator , optimizer_classifier_1 , optimizer_classifier_2) = (Adam(generator.parameters() , lr = learning_rate , weight_decay = weight_decay) , Adam(classifier_1.parameters() , lr = learning_rate , weight_decay = weight_decay) , Adam(classifier_2.parameters() , lr = learning_rate , weight_decay = weight_decay))
	for i in range(epoch):
		start = time()
		generator.train()
		classifier_1.train()
		classifier_2.train()
		for (j , ((data_source , label_source) , data_target)) in enumerate(zip(train_loader , test_loader)):
			(data_source , label_source , data_target) = (data_source.to(device) , label_source.to(device) , data_target.to(device))
			# Step 1
			optimizer_generator.zero_grad()
			optimizer_classifier_1.zero_grad()
			optimizer_classifier_2.zero_grad()
			feature = generator(data_source)
			y_1 = classifier_1(feature)
			y_2 = classifier_2(feature)
			loss = F.cross_entropy(y_1 , label_source) + F.cross_entropy(y_2 , label_source)
			loss.backward()
			optimizer_generator.step()
			optimizer_classifier_1.step()
			optimizer_classifier_2.step()
			# Step 2
			optimizer_generator.zero_grad()
			optimizer_classifier_1.zero_grad()
			optimizer_classifier_2.zero_grad()
			feature = generator(data_source)
			y_1 = classifier_1(feature)
			y_2 = classifier_2(feature)
			loss_1 = F.cross_entropy(y_1 , label_source) + F.cross_entropy(y_2 , label_source)
			feature = generator(data_target)
			y_1 = classifier_1(feature)
			y_2 = classifier_2(feature)
			loss_2 = discrepancy(y_1 , y_2)
			loss = loss_1 - loss_2
			loss.backward()
			optimizer_classifier_1.step()
			optimizer_classifier_2.step()
			# Step 3
			for k in range(4):
				feature = generator(data_target)
				y_1 = classifier_1(feature)
				y_2 = classifier_2(feature)
				loss = discrepancy(y_1 , y_2)
				loss.backward()
				optimizer_generator.step()

			if (j < min(len(train_loader) , len(test_loader)) - 1):
				m = int(50 * (j + 1) / min(len(train_loader) , len(test_loader)))
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} [{}]'.format(i + 1 , epoch , bar) , end = '\r')
			else:
				bar = 50 * '='
				end = time()
				print('epoch {}/{} [{}] ({}s)'.format(i + 1 , epoch , bar , int(end - start)))

	return (generator , classifier_1 , classifier_2)

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , test_x) = load_data()
	(generator , classifier_1 , classifier_2) = (MCD.generator() , MCD.classifier() , MCD.classifier())
	(generator , classifier_1 , classifier_2) = train(train_x , train_y , test_x , generator , classifier_1 , classifier_2 , device)
	torch.save(generator.state_dict() , 'model/generator.pkl')
	torch.save(classifier_1.state_dict() , 'model/classifier_1.pkl')
	torch.save(classifier_2.state_dict() , 'model/classifier_2.pkl')
	return

if (__name__ == '__main__'):
	main()
