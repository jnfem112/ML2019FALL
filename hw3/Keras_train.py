import sys
import os
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import Keras_CNN as CNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_data():
	train_x = list()
	number_of_data = len(os.listdir(sys.argv[1]))
	for i in range(number_of_data):
		image = cv2.imread(os.path.join(sys.argv[1] , '{:0>5d}.jpg'.format(i)) , cv2.IMREAD_GRAYSCALE)
		image = np.expand_dims(image , axis = 2)
		train_x.append(image)
	train_x = np.array(train_x)
	train_x = train_x / 255

	df = pd.read_csv(sys.argv[2])
	train_y = df['label'].values
	train_y = to_categorical(train_y)

	number_of_data = train_x.shape[0]
	index = np.arange(number_of_data)
	np.random.shuffle(index)
	train_x = train_x[index]
	train_y = train_y[index]
	validation_x = train_x[ : 2000]
	validation_y = train_y[ : 2000]
	train_x = train_x[2000 : ]
	train_y = train_y[2000 : ]

	return (train_x , train_y , validation_x , validation_y)

def train(train_x , train_y , validation_x , validation_y , model):
	# Hyper-parameter.
	batch_size = 1024
	learning_rate = 0.0001
	epoch = 2000

	datagen = ImageDataGenerator(rotation_range = 10 , height_shift_range = 0.1  , width_shift_range = 0.1 , zoom_range = 0.1 , horizontal_flip = True)
	datagen.fit(train_x)

	model.compile(loss = 'categorical_crossentropy' , optimizer = Adam(lr = learning_rate) , metrics = ['accuracy'])
	history = model.fit_generator(datagen.flow(train_x , train_y , batch_size = batch_size) , steps_per_epoch = train_x.shape[0] / batch_size , epochs = epoch , validation_data = (validation_x , validation_y))

	result = model.evaluate(train_x , train_y)
	print('train accuracy : {:.5f}'.format(result[1]))
	result = model.evaluate(validation_x , validation_y)
	print('validation accuracy : {:.5f}'.format(result[1]))

	plot(history)

	return model

def plot(history):
	plt.title('Training Process' , fontsize = 24)
	plt.plot(history.history['loss'] , label = 'training loss')
	plt.plot(history.history['val_loss'] , label = 'validation loss')
	plt.legend(loc = 'upper right' , fontsize = 18)
	plt.xlabel('epoch' , fontsize = 18)
	plt.ylabel('loss' , fontsize = 18)
	plt.show()

	plt.title('Training Process' , fontsize = 24)
	plt.plot(history.history['accuracy'] , label = 'training accuracy')
	plt.plot(history.history['val_accuracy'] , label = 'validation accuracy')
	plt.legend(loc = 'lower right' , fontsize = 18)
	plt.xlabel('epoch' , fontsize = 18)
	plt.ylabel('accuracy' , fontsize = 18)
	plt.show()

	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	(train_x , train_y , validation_x , validation_y) = load_data()
	model = CNN.CNN_0()
	model = train(train_x , train_y , validation_x , validation_y , model)
	model.save_weights('Keras_CNN.h5')
	return

if (__name__ == '__main__'):
	main()
