import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
	def __init__(self , lambd):
		self.lambd = lambd
		return

	def forward(self , x):
		return x.view_as(x)

	def backward(self , grad_output):
		return -self.lambd * grad_output

def grad_reverse(x , lambd = 1.0):
	return GradReverse(lambd)(x)

class generator(nn.Module):
	def __init__(self):
		super(generator , self).__init__()
		self.conv1 = nn.Conv2d(1 , 64 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64 , 64 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64 , 128 , kernel_size = (5 , 5) , stride = (1 , 1) , padding = (2 , 2))
		self.bn3 = nn.BatchNorm2d(128)
		self.fc1 = nn.Linear(8192 , 3072)
		self.bn4 = nn.BatchNorm1d(3072)
		return

	def forward(self , x):
		x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))) , stride = (2 , 2) , kernel_size = (3 , 3) , padding = (1 , 1))
		x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))) , stride = (2 , 2) , kernel_size = (3 , 3) , padding = (1 , 1))
		x = F.relu(self.bn3(self.conv3(x)))
		x = x.view(x.size(0) , 8192)
		x = F.relu(self.bn4(self.fc1(x)))
		x = F.dropout(x , training = self.training)
		return x

class classifier(nn.Module):
	def __init__(self):
		super(classifier , self).__init__()
		self.fc1 = nn.Linear(3072 , 2048)
		self.bn1 = nn.BatchNorm1d(2048)
		self.fc2 = nn.Linear(2048 , 10)
		return

	def set_lambda(self , lambd):
		self.lambd = lambd
		return

	def forward(self , x , reverse = False):
		if (reverse):
			x = grad_reverse(x , self.lambd)
		x = F.relu(self.bn1(self.fc1(x)))
		x = self.fc2(x)
		return x
