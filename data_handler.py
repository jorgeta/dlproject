import torch
import pickle
from numpy import savetxt, loadtxt, array
import os

class Dataset(torch.utils.data.Dataset):
	def __init__(self, list_IDs):
		self.list_IDs = list_IDs

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		X = torch.load('data/X/' + ID + '.pt', map_location=torch.device('cpu'))
		y = torch.load('data/y/' + ID + '.pt', map_location=torch.device('cpu'))
		return X, y

class Datasaver():
	def __init__(
		self,
		name, 
		X_train, 
		y_train, 
		X_test, 
		y_test, 
		boarding_data,
		difference_length,
		sequence_length
		):

		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.boarding_data = boarding_data
		self.difference_length = difference_length
		self.sequence_length = sequence_length

		self.path = 'data'
		dirs = [f'{self.path}/X', f'{self.path}/y', f'{self.path}/y_unscaled']
		for this_dir in dirs:
			try:
				for f in os.listdir(this_dir):
					os.remove(os.path.join(this_dir, f))
			except:
				pass
			os.makedirs(this_dir, exist_ok=True)

		self.name = name
		self.storage_path = f'storage/{self.name}'
		os.makedirs(self.storage_path, exist_ok=True)

		self.id_prefix = 'IDNR'

		print('Creating IDs and storing them...')

		self.IDs = self.create_IDs()
		self.store_IDs()

		print('Saving samples to file tagged with their IDs...')

		self.save_samples_to_file()
		self.save_unscaled_boarding_data_to_file()

		print('Samples stored in data folder.')

	def create_IDs(self, unscaled_data=False):
		file_names = []
		self.dataset_length = len(self.X_train)+len(self.X_test)
		if unscaled_data:
			range_start = 0
		else:
			range_start = self.difference_length+self.sequence_length
		range_stop = self.dataset_length+self.difference_length+self.sequence_length
		for i in range(range_start, range_stop):
			max_id_number_length = len(str(self.dataset_length+self.difference_length+self.sequence_length))
			id_number = str(i)
			while len(id_number) < max_id_number_length:
				id_number = '0' + id_number
			current_id = self.id_prefix + id_number
			file_names.append(f'{current_id}')
		return file_names
	
	def store_IDs(self):
		pickle_out = open(f"{self.storage_path}/IDs.pickle","wb")
		pickle.dump(self.IDs, pickle_out)
		pickle_out.close()

	def save_samples_to_file(self):
		for i in range(len(self.X_train)):
			xtrain = self.X_train[i].detach().clone()
			ytrain = self.y_train[i].detach().clone()
			torch.save(xtrain, f'{self.path}/X/{self.IDs[i]}.pt')
			torch.save(ytrain, f'{self.path}/y/{self.IDs[i]}.pt')
		for j in range(len(self.X_test)):
			xtest = self.X_test[j].detach().clone()
			ytest = self.y_test[j].detach().clone()
			torch.save(xtest, f'{self.path}/X/{self.IDs[j+len(self.X_train)]}.pt')
			torch.save(ytest, f'{self.path}/y/{self.IDs[j+len(self.X_train)]}.pt')
	
	def save_unscaled_boarding_data_to_file(self):
		IDs = self.create_IDs(unscaled_data=True)
		for i in range(len(self.boarding_data)):
			bd = torch.Tensor(self.boarding_data[i])
			torch.save(bd, f'{self.path}/y_unscaled/{IDs[i]}.pt')

class Utils():
	def __init__(self, storage_name):
		self.storage_path = f'storage/{storage_name}'
		self.IDs = self.get_IDs()
		self.hyper_params = self.get_predef_hyperparams()
		self.partition = self.get_partition_dict(
			[self.hyper_params['train_set_length'], self.hyper_params['test_set_length']])

	def get_IDs(self):
		try:
			pickle_in = open(f'{self.storage_path}/IDs.pickle',"rb")
		except:
			print('List of IDs has not been created yet.')
		IDs = pickle.load(pickle_in)
		pickle_in.close()
		return IDs

	def get_predef_hyperparams(self):
		try:
			pickle_in = open(f"{self.storage_path}/hyperparameters.pickle","rb")
		except:
			print('List of hyperparameters has not been created yet.')
		hyper_params = pickle.load(pickle_in)
		pickle_in.close()
		return hyper_params

	def get_partition_dict(self, split_lengths):
		IDs = self.get_IDs()
		return {'train': IDs[:split_lengths[0]], 'test': IDs[-split_lengths[1]:]}

class StorageOps():
	def __init__(self, name: str):
		self.name = name
		self.path = f'storage/{name}'
		os.makedirs(self.path, exist_ok=True)
	
	def store_model(
		self, 
		model, 
		train_losses, 
		test_losses,
		input_dimension,
		hidden_dimension,
    	linear_dimension,
    	output_dimension
		):

		pickle_out = open(f"{self.path}/model.pickle","wb")
		pickle.dump(model, pickle_out)
		pickle_out.close()
		savetxt(f'{self.path}/train_losses.npy', array(train_losses), delimiter=',')
		savetxt(f'{self.path}/test_losses.npy', array(test_losses), delimiter=',')
		savetxt(f'{self.path}/dimensions.npy', array([
			input_dimension, hidden_dimension, linear_dimension, output_dimension
		]), delimiter=',')
		

	def load_model(self):
		try:
			pickle_in = open(f"{self.path}/model.pickle","rb")
			model = pickle.load(pickle_in)
			pickle_in.close()
		except:
			print('Model has not been created yet.')
			model = None
		try:
			train_losses = loadtxt(f'{self.path}/train_losses.npy', delimiter=',')
			test_losses = loadtxt(f'{self.path}/test_losses.npy', delimiter=',')
			input_dimension, hidden_dimension, linear_dimension, output_dimension = loadtxt(
				f'{self.path}/dimensions.npy', delimiter=','
			)
		except:
			train_losses = None
			test_losses = None
			print('Train and test loss arrays have not yet been created.')
		return model, train_losses, test_losses, input_dimension, hidden_dimension, linear_dimension, output_dimension

	def save_targets_means_stds(self, targets, means, stds):
		savetxt(f'{self.path}/targets.npy', targets.detach().numpy(), delimiter=',')
		savetxt(f'{self.path}/means.npy', means.detach().numpy(), delimiter=',')
		savetxt(f'{self.path}/stds.npy', stds.detach().numpy(), delimiter=',')

	def save_mse_and_accuracy(self, mse, acc):
		savetxt(f'{self.path}/mse.npy', mse, delimiter=',')
		savetxt(f'{self.path}/acc.npy', acc, delimiter=',')

	def load_targets_means_stds(self):
		targets = torch.from_numpy(loadtxt(f'{self.path}/targets.npy', delimiter=','))
		means = torch.from_numpy(loadtxt(f'{self.path}/means.npy', delimiter=','))
		stds = torch.from_numpy(loadtxt(f'{self.path}/stds.npy', delimiter=','))
		return targets, means, stds
