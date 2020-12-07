from model import LSTM_Net
from data_handler import Dataset, Utils, StorageOps
import torch
import matplotlib.pyplot as plt
from train import train_model
from predict import Predict
from preprocessing import preprocess
from plot import Results

def main():
    do_preprocessing = False
    train_new_model = False
    storage_name = '023'
    do_prediction = False
    plot_results = False

    use_temporal_features = True
    passenger_amount = True
    test_set_length = 24*28
    if do_preprocessing:
        sequence_length = 24*7
        path_to_data_folder = 'raw_data/'
        difference_length = 24*7
        use_difference = True
        bus_nr = '150'
        bus_direction = True
        

        preprocess(
            storage_name,
            sequence_length,
            test_set_length,
            path_to_data_folder,
            difference_length,
            use_difference,
            bus_nr,
            bus_direction,
            passenger_amount,
            use_temporal_features
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
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
    }
    n_samples = 32 # number of samples when sampling the predictions
    n_std = 2 # number of standard deviations away from the mean to set confidence interval
    if use_temporal_features:
        input_dimension = 41 # number of features
    else:
        input_dimension = 10
    hidden_dimension = 512 # dimension of hidden lstm layer
    linear_dimension = 512 # dimension of output of first linear layer
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
        stOps.store_model(trained_model, train_losses, test_losses, input_dimension, hidden_dimension, linear_dimension, output_dimension)
    else:
        trained_model, train_losses, test_losses, input_dimension, hidden_dimension, linear_dimension, output_dimension = stOps.load_model()
        print(input_dimension, hidden_dimension, linear_dimension, output_dimension)
    predicting = Predict(
        name = storage_name,
        trained_model = trained_model, 
        difference_length = difference_length, 
        test_set_IDs = utils.partition['test'],
        params = test_params,
        n_samples = n_samples,
        n_std = n_std,
        passenger_amount=passenger_amount
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
        res = Results(storage_name, passenger_amount)
        for i in range(24):
            res.plot_trend_given_time(i)
        for i in range(10):
            res.plot_trend_given_stop(i)
        '''
        for i in range(test_set_length):
            res.plot_predictions_all_stops(i)
        for i in range(10):
            res.plot_predictions_given_stop(i)
            res.plot_predictions_given_stop_48h(i)
        res.plot_training()
        res.print_performance_measures()'''

main()
