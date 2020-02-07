import sys
import pandas as pd
import numpy as np
from sklearn import ensemble
import joblib

def load_model():
	mean = np.load('best_mean.npy')
	std = np.load('best_std.npy')
	model = joblib.load('best_model.joblib')
	return (mean , std , model)

def load_data():
	with open(sys.argv[1] , 'r') as file:
		test_x = np.array([line.split(',') for line in file])
	test_x = test_x[1 : ].astype(np.float)

	return test_x

def normalize(test_x , mean , std):
	test_x = (test_x - mean) / std
	return test_x

def predict(test_x , model):
	test_y = model.predict(test_x)
	return test_y

def dump(test_y):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : np.arange(1 , number_of_data + 1) , 'label' : test_y})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	(mean , std , model) = load_model()
	test_x = load_data()
	test_x = normalize(test_x , mean , std)

	test_y = predict(test_x , model)

	dump(test_y)

	return

if (__name__ == '__main__'):
	main()