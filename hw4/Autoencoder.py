import torch.nn as nn

class Autoencoder_0(nn.Module):
	def __init__(self):
		super(Autoencoder_0 , self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(3 , 8 , kernel_size = (3 , 3) , stride = (2 , 2) , padding = (1 , 1)) ,
			nn.Conv2d(8 , 16 , kernel_size = (3 , 3) , stride = (2 , 2) , padding = (1 , 1))
		)

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(16 , 8 , kernel_size = (2 , 2) , stride = (2 , 2)) ,
			nn.ConvTranspose2d(8 , 3 , kernel_size = (2 , 2) , stride = (2 , 2)) ,
			nn.Tanh()
		)

		return

	def forward(self , x):
		encode_x = self.encoder(x)
		decode_x = self.decoder(encode_x)
		return (encode_x , decode_x)

class Autoencoder_1(nn.Module):
	def __init__(self):
		super().__init__()
		self.convolution_1 = nn.Conv2d(3 , 1024 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxpool_1 = nn.MaxPool2d(2 , return_indices = True)
		self.convolution_2 = nn.Conv2d(1024 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxpool_2 = nn.MaxPool2d(2 , return_indices = True)
		self.convolution_3 = nn.Conv2d(256 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxpool_3 = nn.MaxPool2d(2 , return_indices = True)
		self.convolution_4 = nn.Conv2d(64 , 16 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxpool_4 = nn.MaxPool2d(2 , return_indices = True)
		self.maxunpool_1 = nn.MaxUnpool2d(2)
		self.deconvolution_1 = nn.ConvTranspose2d(16 , 64 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxunpool_2 = nn.MaxUnpool2d(2)
		self.deconvolution_2 = nn.ConvTranspose2d(64 , 256 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxunpool_3 = nn.MaxUnpool2d(2)
		self.deconvolution_3 = nn.ConvTranspose2d(256 , 1024 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.maxunpool_4 = nn.MaxUnpool2d(2)
		self.deconvolution_4 = nn.ConvTranspose2d(1024 , 3 , kernel_size = (3 , 3) , stride = (1 , 1) , padding = (1 , 1))
		self.activate = nn.Tanh()
		return

	def forward(self , x):
		encode_x = self.convolution_1(x)
		(encode_x , indice_1) = self.maxpool_1(encode_x)
		encode_x = self.convolution_2(encode_x)
		(encode_x , indice_2) = self.maxpool_2(encode_x)
		encode_x = self.convolution_3(encode_x)
		(encode_x , indice_3) = self.maxpool_3(encode_x)
		encode_x = self.convolution_4(encode_x)
		(encode_x , indice_4) = self.maxpool_4(encode_x)
		decode_x = self.maxunpool_1(encode_x , indice_4)
		decode_x = self.deconvolution_1(decode_x)
		decode_x = self.maxunpool_2(decode_x , indice_3)
		decode_x = self.deconvolution_2(decode_x)
		decode_x = self.maxunpool_3(decode_x , indice_2)
		decode_x = self.deconvolution_3(decode_x)
		decode_x = self.maxunpool_4(decode_x , indice_1)
		decode_x = self.deconvolution_4(decode_x)
		decode_x = self.activate(decode_x)
		return (encode_x , decode_x)