import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
import torch
import os
from preprocessing import preprocess

class Baseline():

    def __init__(self, storage_name):

        pp = preprocess(storage_name)
        self.train = np.array(pp.boarding_data[24:-pp.test_set_length]) # 13 weeks exactly
        self.targets = np.array(pp.boarding_data[-pp.test_set_length:]) # 4 weeks exactly
        del pp

        self.plot_path = f'plots/{storage_name}'
        os.makedirs(self.plot_path, exist_ok=True)
    
    def get_means_and_stds(self):
        # split into data for each hour in a week
        # take mean and std values for each hour of the week for each stop

        HIW = 24*7

        means = np.zeros((HIW, 10))
        stds = np.zeros((HIW, 10))
        
        for hour in range(HIW):
            means[hour] = self.train[hour::HIW].mean(axis=0)
            stds[hour] = self.train[hour::HIW].std(axis=0)
        
        return means, stds
    
    def get_predictions_lower_upper(self):
        means, stds = self.get_means_and_stds()
        self.stds = np.array(np.concatenate((stds,stds,stds,stds), axis=0))
        predictions = np.concatenate((means, means, means, means), axis=0)

        lower_short = means-2*stds
        lower_short = lower_short.clip(min=0)

        lower = np.concatenate((lower_short,lower_short,lower_short,lower_short), axis=0)

        upper_short = means+2*stds
        upper_short = upper_short.clip(min=0)
        upper = np.concatenate((upper_short,upper_short,upper_short,upper_short), axis=0)

        self.predictions = np.array(predictions)
        self.upper = np.array(upper)
        self.lower = np.array(lower)
    
    def get_mse_and_acc(self):
        self.mse = np.mean(np.square(self.predictions-self.targets))
        self.acc = self.get_acc(self.targets, self.predictions, self.stds)
    
    def get_acc(self, targets, predictions, stds):
        predictions_flat = predictions.flatten()
        targets_flat = targets.flatten()
        lower_flat = (predictions-2*stds).flatten()
        upper_flat = (predictions+2*stds).flatten()
        lower_flat = lower_flat.clip(min=0)
        upper_flat = upper_flat.clip(min=0)
        total_elements = len(predictions_flat)
        elements_in_condfidence_interval = 0
        for i in range(total_elements):
            if (targets_flat[i]-lower_flat[i] > -0.01) and (targets_flat[i]-upper_flat[i] < 0.01):
                elements_in_condfidence_interval += 1
        return elements_in_condfidence_interval/total_elements
    
    def plot_predictions_all_stops(self, hour: int):
        plt.plot(self.predictions[hour], label='Baseline prediction')
        plt.plot(self.upper[hour], label='Baseline predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower[hour], label='Baseline predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets[hour], label='True value')
        plt.xlabel('Different stops')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        plt.title(f'Hour number: {hour}, time of day: {hour % 24}:00')
        plt.savefig(f'{self.plot_path}/baselineHOUR{hour}.png')
        plt.close()
    
    def plot_predictions_given_stop(self, stop: int):
        mse_given_stop = np.mean(np.square(self.predictions.T[stop]-self.targets.T[stop]))
        acc_given_stop = self.get_acc(self.targets.T[stop], self.predictions.T[stop], self.stds.T[stop])

        plt.plot(self.predictions.T[stop], label='Baseline prediction')
        plt.plot(self.upper.T[stop], label='Baseline predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower.T[stop], label='Baseline predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets.T[stop], label='True value')
        plt.xlabel('Time')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        mse_str = '{:.2f}'.format(self.mse)
        mse_gs_str = '{:.2f}'.format(mse_given_stop)
        acc_str = '{:.2f}'.format(self.acc*100)
        acc_gs_str = '{:.2f}'.format(acc_given_stop*100)
        plt.title(f'Stop number {stop}\nMSE (total, stop): ({mse_str}, {mse_gs_str})\nACC (total, stop): ({acc_str} %, {acc_gs_str} %)')
        plt.savefig(f'{self.plot_path}/baselineSTOPNR{stop}.png')
        plt.close()

    def plot_predictions_given_stop_48h(self, stop: int):
        mse_given_stop = np.mean(np.square(
            self.predictions.T[stop][24*7:24*9]-self.targets.T[stop][24*7:24*9]))
        acc_given_stop = self.get_acc(
            self.targets.T[stop][24*7:24*9], self.predictions.T[stop][24*7:24*9], self.stds.T[stop][24*7:24*9])

        plt.plot(self.predictions.T[stop][24*7:24*9], label='Baseline prediction')
        plt.plot(self.upper.T[stop][24*7:24*9], label='Baseline predicted upper 95 % CI', linestyle='--')
        plt.plot(self.lower.T[stop][24*7:24*9], label='Baseline predicted lower 95 % CI', linestyle='--')
        plt.plot(self.targets.T[stop][24*7:24*9], label='True value')
        plt.xlabel('Time')
        plt.ylabel('Number of people on the bus')
        plt.legend()
        mse_str = '{:.2f}'.format(self.mse)
        mse_gs_str = '{:.2f}'.format(mse_given_stop)
        acc_str = '{:.2f}'.format(self.acc*100)
        acc_gs_str = '{:.2f}'.format(acc_given_stop*100)
        plt.title(f'Stop number {stop}\nMSE (total, stop): ({mse_str}, {mse_gs_str})\nACC (total, stop): ({acc_str} %, {acc_gs_str} %)')
        plt.savefig(f'{self.plot_path}/baselineSTOPNR{stop}_48H.png')
        plt.close()


        

        # predict this mean and std value through whole test set

        # calculate mse and accuracy
    