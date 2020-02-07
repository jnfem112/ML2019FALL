import torch.nn as nn

class DNN_0(nn.Module):
	def __init__(self , dimension):
		super(DNN_0 , self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(dimension , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 100 , bias = True) ,
			nn.BatchNorm1d(100) ,
			nn.ReLU() ,
			nn.Dropout() ,
			nn.Linear(100 , 2 , bias = True)
		)

		return

	def forward(self , x):
		x = self.linear(x)
		return x