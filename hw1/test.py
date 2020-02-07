import sys
import pandas as pd
import numpy as np

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

	def valid(x):
		def replace(x , index , bound):
			left = 0
			right = 0
			for i in range(index - 1 , -1 , -1):
				if (bound[0] <= x[i] <= bound[1]):
					left = x[i]
					break
			for i in range(index + 1 , 9 , 1):
				if (bound[0] <= x[i] <= bound[1]):
					right = x[i]
					break
			return (left + right) / 2

		for i in range(9):
			if (x[0 , i] < 0):
				x[0 , i] = replace(x[0] , i , [0 , np.inf])
			if (x[1 , i] < 0):
				x[1 , i] = replace(x[1] , i , [0 , np.inf])
			if (x[2 , i] < 0):
				x[2 , i] = replace(x[2] , i , [0 , np.inf])
			if (x[3 , i] < 0 or x[3 , i] > 250):
				x[3 , i] = replace(x[3] , i , [0 , 250])
			if (x[4 , i] < 0 or x[4 , i] > 100):
				x[4 , i] = replace(x[4] , i , [0 , 100])
			if (x[5 , i] < 0 or x[5 , i] > 100):
				x[5 , i] = replace(x[5] , i , [0 , 100])
		return x

	test_x = list()
	(number_of_row , number_of_column) = data.shape
	for i in range(number_of_column // 9):
		x = valid(data[ : , 9 * i : 9 * (i + 1)])
		test_x.append(x.reshape(-1))
	test_x = np.array(test_x)

	return test_x

def load_model():
	model = np.load('model.npy')
	weight = model[ : -1].reshape((-1 , 1))
	bias = model[-1]
	return (weight , bias)

def predict(test_x , weight , bias):
	return test_x @ weight + bias

def dump(test_y):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : ['id_{}'.format(i) for i in range(number_of_data)] , 'value' : test_y.reshape(-1)})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	test_x = load_data()
	(weight , bias) = load_model()
	test_y = predict(test_x , weight , bias)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()