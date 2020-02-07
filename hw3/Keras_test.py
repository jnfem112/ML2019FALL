import sys
import os
import pandas as pd
import numpy as np
import cv2
import Keras_CNN as CNN

def load_data():
	test_x = list()
	number_of_data = len(os.listdir(sys.argv[1]))
	for i in range(number_of_data):
		image = cv2.imread(os.path.join(sys.argv[1] , '{:0>4d}.jpg'.format(i)) , cv2.IMREAD_GRAYSCALE)
		image = np.expand_dims(image , axis = 2)
		test_x.append(image)
	test_x = np.array(test_x)
	test_x = test_x / 255
	return test_x

def test(test_x , model):
	test_y = model.predict(test_x)
	test_y = np.argmax(test_y , axis = 1)
	return test_y

def dump(test_y):
	number_of_data = test_y.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : test_y})
	df.to_csv(sys.argv[2] , index = False)
	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	test_x = load_data()
	model = CNN.CNN_0()
	model.load_weights('Keras_CNN.h5')
	test_y = test(test_x , model)
	dump(test_y)
	return

if (__name__ == '__main__'):
	main()