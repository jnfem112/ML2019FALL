from keras.models import Sequential
from keras.layers import ZeroPadding2D , Conv2D , MaxPooling2D , Flatten , Dense , Dropout , Activation
from keras.layers.normalization import BatchNormalization

def CNN_0():
	model = Sequential()
	model.add(Conv2D(64 , (3 , 3) , input_shape = (48 , 48 , 1) , name = 'convolution_1'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Conv2D(128 , (3 , 3) , name = 'convolution_2'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Conv2D(512 , (3 , 3) , name = 'convolution_3'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Flatten())
	for i in range(3):
		model.add(Dense(units = 100 , activation = 'relu'))
	model.add(Dense(units = 7 , activation = 'softmax' , name = 'output'))

	return model

def CNN_1():
	model = Sequential()
	model.add(ZeroPadding2D((2 , 2) , input_shape = (48 , 48 , 1)))
	model.add(Conv2D(32 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(ZeroPadding2D((2 , 2)))
	model.add(Conv2D(64 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(ZeroPadding2D((2 , 2)))
	model.add(Conv2D(128 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(ZeroPadding2D((2 , 2)))
	model.add(Conv2D(256 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(ZeroPadding2D((2 , 2)))
	model.add(Conv2D(512 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(ZeroPadding2D((2 , 2)))
	model.add(Conv2D(1024 , (3 , 3) , padding = 'valid'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D((2 , 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(units = 100))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units = 100))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units = 7))
	model.add(Activation('softmax'))

	return model
