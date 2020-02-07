# sudo pip3 install git+https://github.com/raghakot/keras-vis.git -U
# https://raghakot.github.io/keras-vis/vis.visualization/

import os
import numpy as np
import cv2
import Keras_CNN as CNN
from keras import activations
from vis.utils import utils
from vis.visualization import get_num_filters
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
import matplotlib.pyplot as plt

def feature_visualization(model):
	index = utils.find_layer_idx(model , 'output')
	model.layers[index].activation = activations.linear
	model = utils.apply_modifications(model)

	for (i , layer) in enumerate(['convolution_1' , 'convolution_2' , 'convolution_3']):
		fig = plt.figure(figsize = (18 , 2))

		index = utils.find_layer_idx(model , layer)
		number_of_filter = get_num_filters(model.layers[index])
		for j in range(20):
			image = visualize_activation(model , index , filter_indices = j , max_iter = 2000 , input_modifiers = [Jitter(15)])
			image = image[ : , : , 0]

			ax = fig.add_subplot(1 , 20 , j + 1)
			ax.set_axis_off()
			ax.imshow(image , cmap = 'gray')

		fig.suptitle('convolution layer {}'.format(i + 1) , fontsize = 24)
		plt.show()

	return

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	model = CNN.CNN_0()
	model.load_weights('Keras_CNN.h5')
	feature_visualization(model)
	return

if (__name__ == '__main__'):
	main()
