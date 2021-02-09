import torch
import torch.nn as nn

class RNN_0(nn.Module):
	def __init__(self , embedding , padding_index):
		super(RNN_0 , self).__init__()

		self.embedding = nn.Embedding(embedding.size(0) , embedding.size(1) , padding_idx = padding_index)
		self.embedding.weight = nn.Parameter(embedding)
		self.embedding.weight.requires_grad = True

		self.recurrent = nn.LSTM(embedding.size(1) , 128 , batch_first = True , bias = True , num_layers = 2 , dropout = 0.3 , bidirectional = True)
		
		self.linear = nn.Sequential(
			nn.Linear(768 , 100 , bias = True) ,
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
		x = self.embedding(x)
		(x , _) = self.recurrent(x)
		x = torch.cat([x.min(dim = 1).values , x.max(dim = 1).values , x.mean(dim = 1)] , dim = 1)
		x = self.linear(x)
		return x
