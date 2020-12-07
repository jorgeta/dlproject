import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
from numpy import savetxt, loadtxt
import os

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

        self.plot_path = f'plots/{storage_name}'
        os.makedirs(self.plot_path, exist_ok=True)

        self.mse = self.get_MSE(self.means, self.targets)
        self.acc = self.get_accuracy(self.targets, self.means, self.stds)

        np.set_printoptions(precision=3)

    def plot_training(self):
        print(self.test_losses)
        print(self.train_losses)
        plt.plot(self.test_losses, label='Testing score')
        plt.plot(self.train_losses, label='Training score')
        plt.legend()
        plt.savefig(f'{self.plot_path}/TRAINING.png')
        plt.close()
    
    def plot_predictions_all_stops(self, hour):
        plt.plot(self.means[hour], label='Prediction')
        plt.plot(self.upper[hour], label='Predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower[hour], label='Predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets[hour], label='True value')
        plt.xlabel('Different stops')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        plt.title(f'Hour number: {hour}, time of day: {hour % 24}:00')
        plt.savefig(f'{self.plot_path}/HOUR{hour}.png')
        plt.close()
    
    def plot_predictions_given_stop(self, stop: int):
        mse_given_stop = np.mean(np.square(self.means.T[stop]-self.targets.T[stop]))
        acc_given_stop = self.get_accuracy(self.targets.T[stop], self.means.T[stop], self.stds.T[stop])

        plt.plot(self.means.T[stop], label='Prediction')
        plt.plot(self.upper.T[stop], label='Predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower.T[stop], label='Predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets.T[stop], label='True value')
        plt.xlabel('Time')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        mse_str = '{:.2f}'.format(self.mse[0])
        mse_gs_str = '{:.2f}'.format(mse_given_stop)
        acc_str = '{:.2f}'.format(self.acc[0]*100)
        acc_gs_str = '{:.2f}'.format(acc_given_stop[0]*100)
        plt.title(f'Stop number {stop}\nMSE (total, stop): ({mse_str}, {mse_gs_str})\nACC (total, stop): ({acc_str} %, {acc_gs_str} %)')
        plt.savefig(f'{self.plot_path}/STOPNR{stop}.png')
        plt.close()

    def plot_predictions_given_stop_48h(self, stop: int):
        mse_given_stop = np.mean(np.square(
            self.means.T[stop][24*7:24*9]-self.targets.T[stop][24*7:24*9]))
        acc_given_stop = self.get_accuracy(
            self.targets.T[stop][24*7:24*9], self.means.T[stop][24*7:24*9], self.stds.T[stop][24*7:24*9])

        plt.plot(self.means.T[stop][24*7:24*9], label='Prediction')
        plt.plot(self.upper.T[stop][24*7:24*9], label='Predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower.T[stop][24*7:24*9], label='Predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets.T[stop][24*7:24*9], label='True value')
        plt.xlabel('Time')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        mse_str = '{:.2f}'.format(self.mse[0])
        mse_gs_str = '{:.2f}'.format(mse_given_stop)
        acc_str = '{:.2f}'.format(self.acc[0]*100)
        acc_gs_str = '{:.2f}'.format(acc_given_stop[0]*100)
        plt.title(f'Stop number {stop}\nMSE (total, stop): ({mse_str}, {mse_gs_str})\nACC (total, stop): ({acc_str} %, {acc_gs_str} %)')
        plt.savefig(f'{self.plot_path}/STOPNR{stop}_48H.png')
        plt.close()
    
    def plot_trend_given_stop(self, stop):
        mean_of_means = np.zeros(24)
        mean_of_upper = np.zeros(24)
        mean_of_lower = np.zeros(24)
        mean_of_targets = np.zeros(24)

        for i in range(24):
            mean_of_means[i] = self.means.T[stop][i::24].mean()
            mean_of_upper[i] = self.upper.T[stop][i::24].mean()
            mean_of_lower[i] = self.lower.T[stop][i::24].mean()
            mean_of_targets[i] = self.targets.T[stop][i::24].mean()
        
        plt.plot(mean_of_means, label='Mean of predictions')
        plt.plot(mean_of_upper, label='Mean of predicted upper 95 % CI', linestyle='--')
        plt.plot(mean_of_lower, label='Mean of predicted lower 95 % CI', linestyle='--')
        plt.plot(mean_of_targets, label='Mean of true value')
        plt.xlabel('Time')
        plt.ylabel('Number of people on the bus')
        plt.title(f'Mean of data for each time of day at stop ({stop})')
        plt.legend()
        plt.savefig(f'{self.plot_path}/STOPMEAN{stop}.png')
        plt.close()


    def plot_trend_given_time(self, hour):
        mean_of_means = self.means[hour::24].mean(axis=0)
        mean_of_upper = self.upper[hour::24].mean(axis=0)
        mean_of_lower = self.lower[hour::24].mean(axis=0)
        mean_of_targets = self.targets[hour::24].mean(axis=0)
        
        plt.plot(mean_of_means, label='Mean of predictions')
        plt.plot(mean_of_upper, label='Mean of predicted upper 95 % CI', linestyle='--')
        plt.plot(mean_of_lower, label='Mean of predicted lower 95 % CI', linestyle='--')
        plt.plot(mean_of_targets, label='Mean of true value')
        plt.xlabel('Stops')
        plt.ylabel('Number of people on the bus')
        plt.title(f'Mean of data for each stop at {hour}:00')
        plt.legend()
        plt.savefig(f'{self.plot_path}/HOURMEAN{hour}.png')
        plt.close()
    
    def print_performance_measures(self):
        print(f'MSE: {self.mse}')
        print(f'Accuracy: {self.acc*100} %')

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

    def get_accuracy(self, targets, means, stds):#, given_stop=False, stop=0):
        lower, upper = self.get_upper_and_lower_confidence_interval(means, stds)
        total_elements = torch.numel(lower)
        elements_in_condfidence_interval = 0
        lower_flattened = torch.flatten(lower)
        upper_flattened = torch.flatten(upper)
        try:
            targets_flattened = torch.flatten(targets)
        except:
            targets_flattened = torch.flatten(torch.from_numpy(targets))
        lower_flattened[lower_flattened<0] = 0
        upper_flattened[upper_flattened<0] = 0
        targets_flattened[targets_flattened<0] = 0
        for i in range(total_elements):
            if (targets_flattened[i]-lower_flattened[i] > -0.01) and (targets_flattened[i]-upper_flattened[i] < 0.01):
                elements_in_condfidence_interval += 1
        return np.array([elements_in_condfidence_interval/total_elements])