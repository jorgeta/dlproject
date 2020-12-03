from models import LSTM_Net
from data_handler import Dataset, get_partition_dict, get_predef_hyperparams
import torch
import matplotlib.pyplot as plt
from train import train_model

def main():

    # Parameters
    params = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = 5

    predef_params = get_predef_hyperparams()
    print(predef_params)
    partition = get_partition_dict([predef_params['train_set_length'], predef_params['test_set_length']])

    input_dimension = 41 # number of features
    hidden_dimension = 256 # dimension of hidden lstm layer
    linear_dimension = 256 # dimension of output of first linear layer
    output_dimension = 10 # output dimension of network
    sequence_length = predef_params['sequence_length']
    samples = 20 # number of samples when sampling the predictions
    n_std = 2 # number of standard deviations away from the mean to set confidence interval
    elbo_sample_nbr=3 # The number of times of the weight-sampling and predictions done to gather the loss

    net = LSTM_Net(
        input_dim=input_dimension, 
        hidden_dim=hidden_dimension, 
        linear_dim=linear_dimension,
        sequence_length=sequence_length, 
    )

    print(net)

    trained_model, train_losses, test_losses = train_model(
        model = net, 
        max_epochs = max_epochs,
        params = params,
        partition = partition,
        elbo_sample_nbr = elbo_sample_nbr,
    )

main()