from model import LSTM_Net
from data_handler import Dataset, Utils, StorageOps
import torch
import matplotlib.pyplot as plt
from train import train_model
from predict import Predict
from preprocessing import preprocess
from plot import Results

def main():
    do_preprocessing = True
    train_new_model = True
    storage_name = '003'
    do_prediction = True
    plot_results = True

    if do_preprocessing:
        sequence_length = 24*7
        test_set_length = 24*28
        path_to_data_folder = 'raw_data/'
        difference_length = 24
        use_difference = True
        bus_nr = '150'
        bus_direction = True
        passenger_amount = False
        preprocess(
            storage_name,
            sequence_length,
            test_set_length,
            path_to_data_folder,
            difference_length,
            use_difference,
            bus_nr,
            bus_direction,
            passenger_amount
        )

    # get and print some preprocessorially defined hyperparameters
    utils = Utils(storage_name)
    #print(utils.partition['test'])
    
    for key, value in utils.hyper_params.items():
        print(f'{key}:\t{value}')

    # Parameters
    train_params = {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 0
    }
    max_epochs = 10

    test_params = {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 0
    }
    n_samples = 16 # number of samples when sampling the predictions
    n_std = 2 # number of standard deviations away from the mean to set confidence interval
    input_dimension = 41 # number of features
    hidden_dimension = 256 # dimension of hidden lstm layer
    linear_dimension = 256 # dimension of output of first linear layer
    output_dimension = 10 # output dimension of network
    sequence_length = utils.hyper_params['sequence_length']
    difference_length = utils.hyper_params['difference_length']
    elbo_sample_nbr=3 # The number of times of the weight-sampling and predictions done to gather the loss

    net = LSTM_Net(
        input_dim=input_dimension, 
        hidden_dim=hidden_dimension, 
        linear_dim=linear_dimension,
        sequence_length=sequence_length, 
        output_dim=output_dimension
    )

    print(net)

    stOps = StorageOps(storage_name)

    if train_new_model:
        trained_model, train_losses, test_losses = train_model(
            model = net, 
            max_epochs = max_epochs,
            params = train_params,
            partition = utils.partition,
            elbo_sample_nbr = elbo_sample_nbr,
        )
        stOps.store_model(trained_model, train_losses, test_losses)
    else:
        trained_model, train_losses, test_losses = stOps.load_model()
    
    predicting = Predict(
        name = storage_name,
        trained_model = trained_model, 
        difference_length = difference_length, 
        test_set_IDs = utils.partition['test'],
        params = test_params,
        n_samples = n_samples,
        n_std = n_std
    )
    if do_prediction:
        predicting.predict()

        stOps.save_targets_means_stds(predicting.targets, predicting.means, predicting.stds)
        stOps.save_mse_and_accuracy(
            predicting.get_MSE(predicting.targets, predicting.means), 
            predicting.get_accuracy(predicting.targets, predicting.means, predicting.stds)
        )
    else:
        targets, means, stds = stOps.load_targets_means_stds()
        stOps.save_mse_and_accuracy(
            predicting.get_MSE(targets, means), 
            predicting.get_accuracy(targets, means, stds)
        )
    
    if plot_results:
        res = Results(storage_name)
        res.plot_predictions_all_stops(0)
        res.plot_predictions_given_stop(0)
        res.plot_training()
        res.print_performance_measures()




    

main()