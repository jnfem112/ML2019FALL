import sys
import numpy as np

def load_data():
	with open(sys.argv[1] , 'r') as file:
		train_x = np.array([line.split(',') for line in file])
	train_x = train_x[1 : ].astype(np.float)

	for i in [0 , 1 , 3 , 4 , 5]:
		for j in range(2 , 4):
			train_x = np.hstack((train_x , (train_x[ : , i]**j).reshape((-1 , 1))))
	
	with open(sys.argv[2] , 'r') as file:
		train_y = np.array([line for line in file]).astype(np.int)

	number_of_data = train_x.shape[0]
	index = np.arange(number_of_data)
	np.random.shuffle(index)
	train_x = train_x[index]
	train_y = train_y[index]
	validation_x = train_x[ : 1000]
	validation_y = train_y[ : 1000]
	train_x = train_x[1000 : ]
	train_y = train_y[1000 : ]

	return (train_x , train_y , validation_x , validation_y)

def normalize(train_x , validation_x):
	mean = train_x.mean(axis = 0)
	std = train_x.std(axis = 0)
	train_x = (train_x - mean) / std
	validation_x = (validation_x - mean) / std
	return (train_x , validation_x , mean , std)

def cov(x):
	number_of_data = x.shape[1]
	mean = np.mean(x , axis = 1)
	return sum((x[ : , i] - mean).reshape((-1 , 1)) @ (x[ : , i] - mean).reshape((1 , -1)) for i in range(number_of_data)) / number_of_data

def generative_model(train_x , train_y):
	class_1 = train_x[train_y == 1].T
	class_2 = train_x[train_y == 0].T

	number_of_data = train_x.shape[0]
	number_of_data_1 = class_1.shape[1]
	number_of_data_2 = class_2.shape[1]

	prior_probability_1 = number_of_data_1 / number_of_data
	prior_probability_2 = number_of_data_2 / number_of_data

	mean_1 = np.mean(class_1 , axis = 1).reshape((-1 , 1))
	covariance_1 = cov(class_1)

	mean_2 = np.mean(class_2 , axis = 1).reshape((-1 , 1))
	covariance_2 = cov(class_2)

	covariance = prior_probability_1 * covariance_1 + prior_probability_2 * covariance_2

	return (prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)

def gaussian(x , mean , inv_covariance):
	return np.exp(-0.5 * (((x - mean).T @ inv_covariance) @ (x - mean)))

def accuracy(x , y , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance):
	inv_covariance = np.linalg.inv(covariance)

	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		probability_1 = prior_probability_1 * gaussian(x[i].reshape((-1 , 1)) , mean_1 , inv_covariance)
		probability_2 = prior_probability_2 * gaussian(x[i].reshape((-1 , 1)) , mean_2 , inv_covariance)
		if ((probability_1 > probability_2 and y[i] == 1) or (probability_1 < probability_2 and y[i] == 0)):
			count += 1

	return count / number_of_data

def save_model(mean , std , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance):
	np.save('generative_mean.npy' , mean)
	np.save('generative_std.npy' , std)
	np.save('generative_prob_1.npy' , prior_probability_1)
	np.save('generative_prob_2.npy' , prior_probability_2)
	np.save('generative_mean_1.npy' , mean_1)
	np.save('generative_mean_2.npy' , mean_2)
	np.save('generative_cov.npy' , covariance)
	return

def main():
	(train_x , train_y , validation_x , validation_y) = load_data()
	(train_x , validation_x , mean , std) = normalize(train_x , validation_x)
	(prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance) = generative_model(train_x , train_y)
	print('train accuracy : {:.5f}'.format(accuracy(train_x , train_y , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)))
	print('validation accuracy : {:.5f}'.format(accuracy(validation_x , validation_y , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)))
	save_model(mean , std , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance)
	return

if (__name__ == '__main__'):
	main()