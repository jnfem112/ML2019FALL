import torch.nn as nn

class CNN_0(nn.Module):
	def __init__(self):
		super(CNN_0 , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(1 , 32 , kernel_size = (3 , 3) , stride = (1 , 1)) ,
			nn.BatchNorm2d(32) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(32 , 64 , kernel_size = (3 , 3) , stride = (1 , 1)) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(64 , 128 , kernel_size = (3 , 3) , stride = (1 , 1)) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(128 , 256 , kernel_size = (3 , 3) , stride = (1 , 1)) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d()
		)

		self.linear = nn.Sequential(
			nn.Linear(256 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 7 , bias = True)
		)

		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(0) , -1)
		x = self.linear(x)
		return x

class CNN_1(nn.Module):
	def __init__(self):
		super(CNN_1 , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(1 , 32 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(32) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(32 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(64 , 128 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(128 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(256 , 512 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(512) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d()
		)

		self.linear = nn.Sequential(
			nn.Linear(4608 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 7 , bias = True)
		)

		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(0) , -1)
		x = self.linear(x)
		return x

class CNN_2(nn.Module):
	def __init__(self):
		super(CNN_2 , self).__init__()

		self.convolution = nn.Sequential(
			nn.Conv2d(1 , 32 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(32) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(32 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(64) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(64 , 128 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(128) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(128 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(256) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(256 , 512 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(512) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d() ,
			nn.Conv2d(512 , 1024 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (2 , 2)) ,
			nn.BatchNorm2d(1024) ,
			nn.ReLU() ,
			nn.MaxPool2d(2) ,
			nn.Dropout2d()
		)

		self.linear = nn.Sequential(
			nn.Linear(4096 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 7 , bias = True)
		)

		return

	def forward(self , x):
		x = self.convolution(x)
		x = x.view(x.size(0) , -1)
		x = self.linear(x)
		return x
