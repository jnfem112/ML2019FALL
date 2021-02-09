import sys
import numpy as np
from sklearn import ensemble
import joblib

def load_data():
	with open(sys.argv[1] , 'r') as file:
		train_x = np.array([line.split(',') for line in file])
	train_x = train_x[1 : ].astype(np.float)
	
	with open(sys.argv[2] , 'r') as file:
		train_y = np.array([line for line in file]).astype(np.int)

	validation_x = train_x[ : 1000]
	validation_y = train_y[ : 1000]
	train_x = train_x[1000 : ]
	train_y = train_y[1000 : ]

	return (train_x , train_y , validation_x , validation_y)

def normalize(train_x , validation_x):
	mean = train_x.mean(axis = 0)
	std = train_x.std(axis = 0)
	train_x = (train_x - mean) / std
	validation_x = (validation_x - mean) / std
	return (train_x , validation_x , mean , std)

def train(train_x , train_y , validation_x , validation_y , model):
	model.fit(train_x , train_y)
	print('train accuracy : {:.5f}'.format(model.score(train_x , train_y)))
	print('validation accuracy : {:.5f}'.format(model.score(validation_x , validation_y)))
	return model

def save_model(mean , std , model):
	np.save('best_mean.npy' , mean)
	np.save('best_std.npy' , std)
	joblib.dump(model , 'best_model.joblib')
	return

def main():
	(train_x , train_y , validation_x , validation_y) = load_data()
	(train_x , validation_x , mean , std) = normalize(train_x , validation_x)

	model = ensemble.GradientBoostingClassifier(loss = 'deviance' , learning_rate = 0.1 , n_estimators = 100 , validation_fraction = 0.1 , n_iter_no_change = 10 , tol = 0.0001)
	model = train(train_x , train_y , validation_x , validation_y , model)

	save_model(mean , std , model)

	return

if (__name__ == '__main__'):
	main()
