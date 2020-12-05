import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import savetxt, loadtxt

class Results():
    def __init__(self, storage_name):
        self.path = f'storage/{storage_name}'

        self.train_losses = loadtxt(f'{self.path}/train_losses.npy', delimiter=',')
        self.test_losses = loadtxt(f'{self.path}/test_losses.npy', delimiter=',')
        self.targets = loadtxt(f'{self.path}/targets.npy', delimiter=',')
        self.means = loadtxt(f'{self.path}/means.npy', delimiter=',')
        self.stds = loadtxt(f'{self.path}/stds.npy', delimiter=',')
        self.acc = loadtxt(f'{self.path}/acc.npy', delimiter=',')
        self.mse = loadtxt(f'{self.path}/mse.npy', delimiter=',')

        self.n_std = 2

        self.upper = self.means + (self.n_std * self.stds)
        self.lower = self.means - (self.n_std * self.stds)
    
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
        print(f'MSE: {self.mse}')
        print(f'Accuracy: {self.acc*100} %')