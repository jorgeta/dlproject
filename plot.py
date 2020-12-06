import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
from numpy import savetxt, loadtxt

class Results():
    def __init__(self, storage_name, passenger_amount):
        self.path = f'storage/{storage_name}'
        self.passenger_amount = passenger_amount

        self.train_losses = loadtxt(f'{self.path}/train_losses.npy', delimiter=',')
        self.test_losses = loadtxt(f'{self.path}/test_losses.npy', delimiter=',')
        self.targets = loadtxt(f'{self.path}/targets.npy', delimiter=',')
        self.means = loadtxt(f'{self.path}/means.npy', delimiter=',')
        self.stds = loadtxt(f'{self.path}/stds.npy', delimiter=',')
        self.acc = loadtxt(f'{self.path}/acc.npy', delimiter=',')
        self.mse = loadtxt(f'{self.path}/mse.npy', delimiter=',')

        self.n_std = 2

        self.lower, self.upper = self.get_upper_and_lower_confidence_interval(self.means, self.stds)

    def plot_training(self):
        print(self.test_losses)
        print(self.train_losses)
        plt.plot(self.test_losses, label='test')
        plt.plot(self.train_losses, label='train')
        plt.legend()
        plt.show()
    
    def plot_predictions_all_stops(self, hour):
        plt.plot(self.means[hour], label='prediction')
        plt.plot(self.upper[hour], label='upper CI', linestyle='--')
        plt.plot(self.lower[hour], label='lower CI', linestyle='--')
        plt.plot(self.targets[hour], label='target')
        plt.xlabel('stops')
        plt.ylabel('on- and off-boardings')
        plt.legend()
        plt.title(f'hour number {hour}')
        plt.show()
    
    def plot_predictions_given_stop(self, stop: int):
        plt.plot(self.means.T[stop], label='prediction')
        plt.plot(self.upper.T[stop], label='upper CI', linestyle='--')
        plt.plot(self.lower.T[stop], label='lower CI', linestyle='--')
        plt.plot(self.targets.T[stop], label='target')
        plt.xlabel('time')
        plt.ylabel('on- and off-boardings')
        plt.legend()
        plt.title(f'stop number {stop}')
        plt.show()
    
    def print_performance_measures(self):
        print(f'MSE: {self.get_MSE(self.means, self.targets)}')
        print(f'Accuracy: {self.get_accuracy(self.targets, self.means, self.stds)*100} %')

    def get_MSE(self, means, targets):
        '''Compare targets and means'''
        criterion = nn.MSELoss()
        mse = criterion(torch.Tensor(means).float(), torch.Tensor(targets).float())
        return np.array([mse.cpu().detach().numpy()])
    
    def get_upper_and_lower_confidence_interval(self, means, stds):
        upper = means + (self.n_std * stds)
        lower = means - (self.n_std * stds)
        if self.passenger_amount:
            upper[upper < 0] = 0
            lower[lower < 0] = 0
        return torch.from_numpy(lower), torch.from_numpy(upper)

    def get_accuracy(self, targets, means, stds):
        lower, upper = self.get_upper_and_lower_confidence_interval(means, stds)
        total_elements = torch.numel(lower)
        elements_in_condfidence_interval = 0
        lower_flattened = torch.flatten(lower)
        upper_flattened = torch.flatten(upper)
        targets_flattened = torch.flatten(torch.from_numpy(targets))
        lower_flattened[lower_flattened<0] = 0
        upper_flattened[upper_flattened<0] = 0
        targets_flattened[targets_flattened<0] = 0
        for i in range(total_elements):
            if (targets_flattened[i]-lower_flattened[i] > 0) and (targets_flattened[i]-upper_flattened[i] < 0):
                elements_in_condfidence_interval += 1
        return np.array([elements_in_condfidence_interval/total_elements])