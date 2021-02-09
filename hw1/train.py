import sys
import pandas as pd
import numpy as np
from time import time

def load_data():
	df = pd.read_csv(sys.argv[1])

	for column in list(df.columns[2 : ]):
		df[column] = df[column].astype(str).map(lambda string : string.rstrip('#x*A'))

	df[df == ''] = 0
	df[df == 'nan'] = 0
	df[df == 'NR'] = 0

	data = df.values
	data = np.delete(data , [0 , 1] , axis = 1)
	data = data.astype(np.float)

	(number_of_row , number_of_column) = data.shape
	data = np.hstack([data[18 * i : 18 * (i + 1)] for i in range(number_of_row // 18)])

	data = data[[2 , 5 , 7 , 8 , 9 , 12]]

	def valid(x , y):
		for i in range(9):
			if (x[0 , i] < 0):
				return False
			if (x[1 , i] < 0):
				return False
			if (x[2 , i] < 0):
				return False
			if (x[3 , i] < 0 or x[3 , i] > 250):
				return False
			if (x[4 , i] < 0 or x[4 , i] > 100):
				return False
			if (x[5 , i] < 0 or x[5 , i] > 100):
				return False
		if (y < 0 or y > 100):
			return False
		return True

	train_x = list()
	train_y = list()
	(number_of_row , number_of_column) = data.shape
	for i in range(number_of_column - 9):
		x = data[ : , i : i + 9]
		y = data[4 , i + 9]
		if valid(x , y):
			train_x.append(x.reshape(-1))
			train_y.append(y)
	train_x = np.array(train_x)
	train_y = np.array(train_y)

	validation_x = train_x[ : 500]
	validation_y = train_y[ : 500]
	train_x = train_x[500 : ]
	train_y = train_y[500 : ]

	return (train_x , train_y , validation_x , validation_y)

def RMSE(x , y , weight , bias):
	return np.sqrt(np.mean(((x @ weight + bias) - y.reshape((-1 , 1)))**2))

def minibatch(train_x , train_y , validation_x , validation_y):
	# Shuffle training data.
	(number_of_data , dimension) = train_x.shape
	index = np.arange(number_of_data)
	np.random.shuffle(index)
	train_x = train_x[index]
	train_y = train_y[index]

	# Initialization.
	weight = np.full(dimension , 0.1).reshape((-1 , 1))
	bias = 0.1
	
	# Hyper-parameter.
	epoch = 20
	batch_size = 64
	learning_rate = 1e-3
	lambd = 0.001
	beta_1 = np.full(dimension , 0.9).reshape((-1 , 1))
	beta_2 = np.full(dimension , 0.99).reshape((-1 , 1))
	m_t = np.full(dimension , 0).reshape((-1 , 1))
	v_t = np.full(dimension , 0).reshape((-1 , 1))
	m_t_b = 0
	v_t_b = 0
	t = 0
	epsilon = 1e-8
	
	for i in range(epoch):
		start = time()
		for j in range(int(number_of_data / batch_size)):
			t += 1
			x_batch = train_x[j * batch_size : (j + 1) * batch_size]
			y_batch = train_y[j * batch_size : (j + 1) * batch_size].reshape((-1 , 1))
			loss = y_batch - np.dot(x_batch , weight) - bias
			
			# Calculate gradient.
			g_t = -2 * np.dot(x_batch.transpose() , loss) +  2 * lambd * np.sum(weight)
			g_t_b = 2 * loss.sum(axis = 0)
			m_t = beta_1 * m_t + (1 - beta_1) * g_t 
			v_t = beta_2 * v_t + (1 - beta_2) * np.multiply(g_t , g_t)
			m_cap = m_t / (1 - beta_1**t)
			v_cap = v_t / (1 - beta_2**t)
			m_t_b = 0.9 * m_t_b + (1 - 0.9) * g_t_b
			v_t_b = 0.99 * v_t_b + (1 - 0.99) * (g_t_b * g_t_b) 
			m_cap_b = m_t_b / (1 - 0.9**t)
			v_cap_b = v_t_b / (1 - 0.99**t)
			w_0 = np.copy(weight)
			
			# Update weight and bias.
			weight -= ((learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)).reshape((-1 , 1))
			bias -= (learning_rate * m_cap_b) / (np.sqrt(v_cap_b) + epsilon)

			if (j < int(number_of_data / batch_size) - 1):
				n = (j + 1) * batch_size
				m = int(50 * n / number_of_data)
				bar = m * '=' + '>' + (49 - m) * ' '
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , number_of_data , bar) , end = '\r')
			else:
				n = number_of_data
				bar = 50 * '='
				print('epoch {}/{} ({}/{}) [{}]'.format(i + 1 , epoch , n , number_of_data , bar) , end = ' ')

		# Calculate loss.
		train_RMSE = RMSE(train_x , train_y , weight , bias)
		validation_RMSE = RMSE(validation_x , validation_y , weight , bias)

		end = time()
		print('({}s) train RMSE : {:.5f} , validation RMSE : {:.5f}'.format(int(end - start) , train_RMSE , validation_RMSE))

	return (weight , bias)

def save_model(weight , bias):
	model = np.hstack((weight.reshape(-1) , bias))
	np.save('model.npy' , model)
	return

def main():
	(train_x , train_y , validation_x , validation_y) = load_data()
	(weight , bias) = minibatch(train_x , train_y , validation_x , validation_y)
	save_model(weight , bias)
	return

if (__name__ == '__main__'):
	main()
