import torch
import pickle
from numpy import savetxt, loadtxt, array

class Dataset(torch.utils.data.Dataset):
	def __init__(self, list_IDs):
		self.list_IDs = list_IDs

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		X = torch.load('data/X/' + ID + '.pt')
		y = torch.load('data/y/' + ID + '.pt')
		return X, y

class Datasaver():
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

		self.id_prefix = 'IDNR'

		print('Creating IDs and storing them...')

		self.create_IDs()
		self.store_IDs()
		#self.store_training_test_lengths()

		print('Saving samples to file tagged with their IDs...')

		self.save_samples_to_file()

		print('Samples stored in data folder.')

	def create_IDs(self):
		file_names = []
		self.dataset_length = len(self.X_train)+len(self.X_test)
		for i in range(self.dataset_length):
			max_id_number_length = len(str(self.dataset_length))
			id_number = str(i)
			while len(id_number) < max_id_number_length:
				id_number = '0' + id_number
			current_id = self.id_prefix + id_number
			file_names.append(f'{current_id}')
		self.IDs = file_names
	
	def store_IDs(self):
		pickle_out = open("data/IDs.pickle","wb")
		pickle.dump(self.IDs, pickle_out)
		pickle_out.close()

	def save_samples_to_file(self):
		for i in range(len(self.X_train)):
			xtrain = self.X_train[i].detach().clone()
			ytrain = self.y_train[i].detach().clone()
			torch.save(xtrain, f'data/X/{self.IDs[i]}.pt')
			torch.save(ytrain, f'data/y/{self.IDs[i]}.pt')
		for j in range(len(self.X_test)):
			xtest = self.X_test[j].detach().clone()
			ytest = self.y_test[j].detach().clone()
			torch.save(xtest, f'data/X/{self.IDs[j+len(self.X_train)]}.pt')
			torch.save(ytest, f'data/y/{self.IDs[j+len(self.X_train)]}.pt')

def get_IDs():
	try:
		pickle_in = open("data/IDs.pickle","rb")
	except:
		print('List of IDs has not been created yet.')
	IDs = pickle.load(pickle_in)
	pickle_in.close()
	return IDs

def get_predef_hyperparams():
	try:
		pickle_in = open("data/hyperparameters.pickle","rb")
	except:
		print('List of hyperparameters has not been created yet.')
	hyper_params = pickle.load(pickle_in)
	pickle_in.close()
	return hyper_params

def get_partition_dict(split_lengths):
	IDs = get_IDs()
	return {'train': IDs[:split_lengths[0]], 'test': IDs[-split_lengths[1]:]}