import sys
import pandas as pd
import numpy as np

def load_model():
	mean = np.load('generative_mean.npy')
	std = np.load('generative_std.npy')
	prior_probability_1 = np.load('generative_prob_1.npy')
	prior_probability_2 = np.load('generative_prob_2.npy')
	mean_1 = np.load('generative_mean_1.npy')
	mean_2 = np.load('generative_mean_2.npy')
	covariance = np.load('generative_cov.npy')
	return (mean , std , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)

def load_data():
	with open(sys.argv[1] , 'r') as file:
		test_x = np.array([line.split(',') for line in file])
	test_x = test_x[1 : ].astype(np.float)

	for i in [0 , 1 , 3 , 4 , 5]:
		for j in range(2 , 4):
			test_x = np.hstack((test_x , (test_x[ : , i]**j).reshape((-1 , 1))))

	return test_x

def normalize(test_x , mean , std):
	test_x = (test_x - mean) / std
	return test_x

def gaussian(x , mean , inv_covariance):
	return np.exp(-0.5 * (((x - mean).T @ inv_covariance) @ (x - mean)))

def predict(test_x , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance):
	inv_covariance = np.linalg.inv(covariance)

	test_y = list()
	number_of_data = test_x.shape[0]
	for i in range(number_of_data):
		test_y.append(1 if (prior_probability_1 * gaussian(test_x[i].reshape((-1 , 1)) , mean_1 , inv_covariance) > prior_probability_2 * gaussian(test_x[i].reshape((-1 , 1)) , mean_2 , inv_covariance)) else 0)

	return test_y

def dump(test_y):
	number_of_data = len(test_y)
	df = pd.DataFrame({'id' : np.arange(1 , number_of_data + 1) , 'label' : test_y})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	(mean , std , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance) = load_model()
	test_x = load_data()
	test_x = normalize(test_x , mean , std)
	test_y = predict(test_x , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()