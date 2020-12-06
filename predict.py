import torch
from torch import nn
import numpy as np
import pickle
import os
from data_handler import Dataset
from torch import nn

class Predict():
    def __init__(
        self, 
        name,
        trained_model, 
        difference_length, 
        test_set_IDs,
        params,
        n_samples,
        n_std,
        passenger_amount
        ):

        self.model = trained_model
        self.difference_length = difference_length
        self.test_set_IDs = test_set_IDs
        self.params = params
        self.n_samples = n_samples
        self.n_std = n_std
        self.passenger_amount = passenger_amount

        self.name = name
        self.path = f'storage/{self.name}'
        os.makedirs(self.path, exist_ok=True)

    def predict(self):
        self.get_scaler()
        self.samplePredictions()
        self.get_test_targets()

    def get_scaler(self):
        print('Getting scaler...')
        try:
            pickle_in = open(f"{self.path}/scaler.pickle","rb")
        except:
            print('Scaler has not been created yet.')
        scaler = pickle.load(pickle_in)
        pickle_in.close()
        self.scaler = scaler

    def inverseScale(self, output):
        return torch.from_numpy(self.scaler.inverse_transform(output.cpu().detach().numpy()))

    def inverseDifference(self, output, IDs):
        for i in range(len(output)):
            difference_ID = self.get_ID_minus_difference(IDs[i])
            y_difference_ago = torch.load('data/y_unscaled/' + difference_ID + '.pt')
            output[i] = output[i].add(y_difference_ago)
        if self.passenger_amount:
            output[output < 0] = 0
        return output

    def get_ID_minus_difference(self, ID):
        id_number = ID[4:]
        id_prefix = ID[:4]
        differenced_id_number = str(int(id_number)-self.difference_length)
        while len(id_number) > len(differenced_id_number):
            differenced_id_number = '0' + differenced_id_number
        ID = id_prefix + differenced_id_number
        return ID

    def samplePredictions(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        validation_set = Dataset(self.test_set_IDs)
        validation_generator = torch.utils.data.DataLoader(validation_set, **self.params)

        with torch.set_grad_enabled(False):
            counter = 0
            for local_batch, local_labels in validation_generator:
                if counter % 10 == 0:
                    print(f'Batch prediction number {counter+1}...')
                outputs = []
                for sample in range(self.n_samples):
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                    # Model computations
                    output = self.model(local_batch.float())

                    outputs.append(output)
                
                # do inverse transform before taking mean and std
                preds = [self.inverseScale(outputs[k]) for k in range(len(outputs))]

                if self.difference_length > 0:
                    IDs = self.test_set_IDs[
                        counter*self.params['batch_size']:(counter+1)*self.params['batch_size']]
                    preds = [self.inverseDifference(preds[k], IDs) for k in range(len(outputs))]
                
                preds = torch.stack(preds)

                # find mean and std for each observation, store these in a list
                if counter==0:
                    means = preds.mean(axis=0)
                    stds = preds.std(axis=0)
                else:
                    means = torch.cat((means, preds.mean(axis=0)), 0)
                    stds = torch.cat((stds, preds.std(axis=0)), 0)
                counter += 1
        
        self.means = means
        self.stds = stds

    def get_test_targets(self):
        targets = []
        for ID in self.test_set_IDs:
            targets.append(torch.load('data/y_unscaled/' + ID + '.pt'))
        self.targets = torch.stack(targets)
    
    def get_MSE(self, means, targets):
        '''Compare targets and means'''
        criterion = nn.MSELoss()
        mse = criterion(means, targets)
        return np.array([mse.cpu().detach().numpy()])
    
    def get_upper_and_lower_confidence_interval(self, means, stds):
        upper = means + (self.n_std * stds)
        lower = means - (self.n_std * stds)
        if self.passenger_amount:
            upper[upper < 0] = 0
            lower[lower < 0] = 0
        return lower, upper

    def get_accuracy(self, targets, means, stds):
        lower, upper = self.get_upper_and_lower_confidence_interval(means, stds)
        total_elements = torch.numel(lower)
        elements_in_condfidence_interval = 0
        lower_flattened = torch.flatten(lower)
        upper_flattened = torch.flatten(upper)
        targets_flattened = torch.flatten(targets)
        for i in range(total_elements):
            if (targets_flattened[i]-lower_flattened[i] > 0) and (targets_flattened[i]-upper_flattened[i] < 0):
                elements_in_condfidence_interval += 1
        return np.array([elements_in_condfidence_interval/total_elements])