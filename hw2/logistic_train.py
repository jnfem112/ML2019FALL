import sys
import numpy as np
from time import time

def load_data():
	with open(sys.argv[1] , 'r') as file:
		train_x = np.array([line.split(',') for line in file])
	train_x = train_x[1 : ].astype(np.float)

	for i in [0 , 1 , 3 , 4 , 5]:
		for j in range(2 , 11):
			train_x = np.hstack((train_x , (train_x[ : , i]**j).reshape((-1 , 1))))
	
	with open(sys.argv[2] , 'r') as file:
		train_y = np.array([line for line in file]).astype(np.int)

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

def sigmoid(x):
	return np.clip(1 / (1 + np.exp(-x)) , 1e-6 , 1 - 1e-6)

def logistic_regression(train_x , train_y , validation_x , validation_y):
	# Initialization.
	weight = np.zeros(train_x.shape[1])
	bias = 0.0

	# Hyper-parameter.
	epoch = 300
	lr = 0.05
	w_lr = np.ones(train_x.shape[1])
	b_lr = 0

	for i in range(epoch):
		start = time()

		z = np.dot(train_x , weight) + bias
		pred = sigmoid(z)
		loss = train_y - pred

		# Calculate gradient.
		w_grad = -1 * np.dot(loss , train_x)
		b_grad = -1 * np.sum(loss)

		# Update weight and bias.
		w_lr += w_grad**2
		b_lr += b_grad**2
		bias = bias - lr / np.sqrt(b_lr) * b_grad
		weight = weight - lr / np.sqrt(w_lr) * w_grad

		# Calculate loss and accuracy.
		loss = -1 * np.mean(train_y * np.log(pred + 1e-100) + (1 - train_y) * np.log(1 - pred + 1e-100))
		train_accuracy = accuracy(train_x , train_y , weight , bias)
		validation_accuracy = accuracy(validation_x , validation_y , weight , bias)

		end = time()
		print('({}s) [epoch {}/{}] loss : {:.5f} , train accuracy : {:.5f} , validation accuracy : {:.5f}'.format(int(end - start) , i + 1 , epoch , loss , train_accuracy , validation_accuracy))

	return (weight , bias)

def accuracy(x , y , weight , bias):
	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		probability = sigmoid(weight @ x[i] + bias)
		if ((probability > 0.5 and y[i] == 1) or (probability < 0.5 and y[i] == 0)):
			count += 1
	return count / number_of_data

def save_model(mean , std , weight , bias):
	model = np.hstack((weight , bias))
	np.save('logistic_mean.npy' , mean)
	np.save('logistic_std.npy' , std)
	np.save('logistic_model.npy' , model)
	return

def main():
	(train_x , train_y , validation_x , validation_y) = load_data()
	(train_x , validation_x , mean , std) = normalize(train_x , validation_x)
	(weight , bias) = logistic_regression(train_x , train_y , validation_x , validation_y)
	save_model(mean , std , weight , bias)
	return

if (__name__ == '__main__'):
	main()
