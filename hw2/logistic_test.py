import sys
import pandas as pd
import numpy as np

def load_model():
	mean = np.load('logistic_mean.npy')
	std = np.load('logistic_std.npy')
	model = np.load('logistic_model.npy')
	weight = model[ : -1]
	bias = model[-1]
	return (mean , std , weight , bias)

def load_data():
	with open(sys.argv[1] , 'r') as file:
		test_x = np.array([line.split(',') for line in file])
	test_x = test_x[1 : ].astype(np.float)

	for i in [0 , 1 , 3 , 4 , 5]:
		for j in range(2 , 11):
			test_x = np.hstack((test_x , (test_x[ : , i]**j).reshape((-1 , 1))))

	return test_x

def normalize(test_x , mean , std):
	test_x = (test_x - mean) / std
	return test_x

def sigmoid(x):
	return np.clip(1 / (1 + np.exp(-x)) , 1e-6 , 1 - 1e-6)

def predict(test_x , weight , bias):
	test_y = list()
	number_of_data = test_x.shape[0]
	for i in range(number_of_data):
		test_y.append(1 if (sigmoid(weight @ test_x[i] + bias) > 0.5) else 0)
	return test_y

def dump(test_y):
	number_of_data = len(test_y)
	df = pd.DataFrame({'id' : np.arange(1 , number_of_data + 1) , 'label' : test_y})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	(mean , std , weight , bias) = load_model()
	test_x = load_data()
	test_x = normalize(test_x , mean , std)
	test_y = predict(test_x , weight , bias)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()