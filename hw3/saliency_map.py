# sudo pip3 install git+https://github.com/raghakot/keras-vis.git -U
# https://raghakot.github.io/keras-vis/visualizations/saliency/

import sys
import os
from random import randint
import pandas as pd
import numpy as np
import cv2
import Keras_CNN as CNN
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_saliency
from vis.visualization import overlay
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_image(i):
	df = pd.read_csv(sys.argv[2])
	label = df['label'].values
	index = np.where(label == i)[0]
	index = index[randint(0 , index.shape[0] - 1)]
	image_name = '{:0>5d}.jpg'.format(index)
	image_1 = cv2.imread(os.path.join(sys.argv[1] , image_name) , cv2.IMREAD_COLOR)
	image_2 = cv2.imread(os.path.join(sys.argv[1] , image_name) , cv2.IMREAD_GRAYSCALE)
	image_2 = np.expand_dims(image_2 , axis = 2)
	image_2 = image_2 / 255
	return (image_name , image_1 , image_2)

def saliency_map(model):
	index = utils.find_layer_idx(model , 'output')
	model.layers[index].activation = activations.linear
	model = utils.apply_modifications(model)

	for i in range(7):
		(image_name , image_1 , image_2) = load_image(i)
		gradient = visualize_saliency(model , index , filter_indices = i , seed_input = image_2 , backprop_modifier = 'guided')

		fig = plt.figure()

		ax1 = fig.add_subplot(1 , 3 , 1)
		ax1.set_title(image_name)
		ax1.set_axis_off()
		ax1.imshow(image_1)

		ax2 = fig.add_subplot(1 , 3 , 2)
		ax2.set_title('saliency map')
		ax2.set_axis_off()
		ax2.imshow(gradient , cmap = 'jet')

		jet_heatmap = np.uint8(255 * cm.jet(gradient)[... , : 3])
		ax3 = fig.add_subplot(1 , 3 , 3)
		ax3.set_title('overlay')
		ax3.set_axis_off()
		ax3.imshow(overlay(jet_heatmap , image_1))

		plt.show()

	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = CNN.CNN_0()
	model.load_weights('Keras_CNN.h5')
	saliency_map(model)
	return

if (__name__ == '__main__'):
	main()
