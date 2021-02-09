import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import Autoencoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def load_data():
	test_x = np.load(sys.argv[1])
	test_x = np.transpose(test_x , (0 , 3 , 1 , 2))
	test_x = test_x / 255
	test_x = 2 * test_x - 1
	return torch.FloatTensor(test_x)

def cluster(test_x , model , device):
	# Hyper-parameter.
	batch_size = 256

	test_loader = DataLoader(test_x , batch_size = batch_size , shuffle = False)

	model.to(device)
	model.eval()
	new_x = list()
	with torch.no_grad():
		for data in test_loader:
			data = data.to(device)
			(encode , decode) = model(data)
			new_x.append(encode.cpu().detach().numpy())
	new_x = np.concatenate(new_x , axis = 0)
	new_x = new_x.reshape((test_x.shape[0] , -1))
	new_x = (new_x - np.mean(new_x , axis = 0)) / np.std(new_x , axis = 0)

	tsne = TSNE(n_components = 2)
	new_x = tsne.fit_transform(new_x)

	kmeans = KMeans(n_clusters = 2)
	kmeans.fit(new_x)
	test_y = kmeans.labels_

	if (np.sum(test_y[ : 5]) > 2):
		test_y = 1 - test_y

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
	model = Autoencoder.Autoencoder_1()
	model.load_state_dict(torch.load('model.pkl' , map_location = device))
	test_y = cluster(test_x , model , device)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()
