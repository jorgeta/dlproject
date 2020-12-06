# Deep Learning Project

## TODO

- Make a baseline where an as similar as possible preprocessing strategy is used.
- Discuss how the results relates to the project description.
- Ask supervisor about prediction horizon.
- Figure out which plots to include in the results section in the poster.
- Agree on further work; what we can do during Christmas to improve the model.

## Project description

Around the world, the Covid-19 is spreading again, and many countries have imposed new “corona” restrictions in an attempt to contain the infection (Hjelskov, 2020). Also, Denmark has new restrictions to try and limit Covid-19 (Københavns Vestegns Politi, 2020). The outburst and talks about a second wave of the Covid-19 (Gøttske, 2020) social distance is becoming important to the Danish citizens and people around the world. Reason being it has been shown that the effect of social distancing after six weeks amounted to a reduction in infection rate by 25% to 45% for most countries in Europe (Cot et al., 2020). Thereby, it is crucial to ensure enough capacity in public transport, as it helps the citizens to keep social distance and thereby helps to reduce the spread of the virus. 

A way to avoid full public transport, and thereby ensuring enough capacity, is by sharing information with passengers, allowing those who can, to choose alternative routes or modes. This can be accomplished by predicting whether a given bus will be at capacity or not and allowing passengers access to this information.   

This project will be conducted in collaboration with Movia and seeks to explore the confidence of such a prediction, with the goal of determining whether this solution would add value to the user experience. The model used in the project will be a Bayesian Recurrent Neural Network (BRNN) (Fortunato et al., 2017). Consequently, the research question will be:

“Utilising a BRNN to predict whether or not a bus will be at full capacity and the uncertainty of it happening, in order to provide the passenger with such information.”

## Preprocessing

### Main data transformation
Started out with data telling when people checked in and out of the bus, we made this into an array containing the net amount of people going into the bus per stop.

### Scaling of the data
Different types of scaling and differencing were tested:
- Max-min scaling
- Standard scaling
- Gathering all data points with the same stop, hour of the day, and day of the week, then subtracting their mean and dividing by their standard deviation
- Gathering all data points with the same stop, hour of the day, and subtracting their mean
- Differencing, that is, not predicting the actual number of passengers going on and of the bus at each stop, but predicting the difference from either one hour ago, one day ago or one week ago at that very stop

### Train-test split
Differencing and sequencing causes loss of number of samples, that is, a higher difference length and/or a higher sequence length provides less data. This happens, however, only in the training set, as the differencing in the test set can be done on the basis of the training set. In the same way, a sequence in the test set can include observations from the training set as part of the sequence.

## Structure of the data

### Simplifying the task
- The original model is designed to be able to predict the number of people going in and out of the bus at any given time at any of the stops in the data.
- Tested whether the model would perform better if we limited the data into 24 datasets, each containing only data from one time of the day, for example only from 07:00 in the morning.
- Tested whether the model would perform better if we limited the data into 10 (number of stops) datasets, each containing only data from a specific stop.

### Sequence creation
The sequence length, how many hours before the time predicted should be feeded into the model (tested up to two weeks).

### Temporal features
- Added numbers to each sample containing information on the day of the week and the time of day. Features representing rush hour, weekend and night time were also tried.
- One hot enconding of these temporal features were eventually applied (only for the day of the week and the time of day)

## Network

### Structure of the network

The Bayesian LSTM based neural network has the goal to provide a confidence interval for each prediction. Therefore, some layers are Bayesian, i.e. for each forward pass, the weight is sampled from a normal distribution. In these layers, the weights themselves aren't the parameters who are trained, but instead the mean and standard deviation of the normal distribution from which the weights are sampled.

In every case, the network starts by using a Bayesian LSTM. Two versions of the output of this have been tried:
- taking the last output of the LSTM, i.e. what the LSTM predicts for the given prediction time point, and using only this output further in the network.
- taking all the outputs from the LSTM, and using them further in the network.

The output of the Bayesian LSTM thereafter goes through a dense Bayesian linear layer. An important point here is that it seemed difficult for this layer to do accurate predictions with its means. It functions more like a layer for determining the uncertainty. A non-Bayesian layer is needed afterwards in order to capture the actual likely values.

Adding one more Bayesian LSTM layer and one more Bayesian linear layer were tested without any signs of improvement.

At the end, a linear layer is provided to capture the actual values. Adding another linear layer did not improve the score.

The seemily best structure of the network is therefore a Bayesian LSTM, a Bayesian linear layer, and a linear layer. 

### Temporal features in the network

Where the temporal features becomes a part of the net is something we tested. One can argue that the LSTM should be able to capture trends itself without temporal features. However, the temporal features could also be a part of the LSTM, and help it learn patterns better. Therefore, two options were tested:
- temporal features are part of the sample all the way through the network, also in the LSTM.
- temporal features only becomes a part of the network in the last linear layer.

### Other hyperparameters and functions

Both the Adam optimizer and the SGD optimizer were tested. Without a doubt, the Adam optimizer with a learning rate of around 0.001 reached the lowest scores.

Changing the batch size from 64 didn't yeild any noticable improvements.

Networks with a high number of nodes per layer gave, as expected, a slower training, but also provided better results. However, there is an upper limit to how many nodes there can be in total, as one does not want a model with more parameters than there are samples, so overfitting is avoided.

## Predicting and sampling

Predictions are made by forwarding a sample through the network several times. Using the fact that the weights in the Bayesian layers of the network are sampled randomly for each forward pass, the network gives different outputs. Taking the mean and standard deviations of these outputs for a given sample, provides a prediction condifence interval.

There are two scores associated with how well a given trained model is performing. Both are applied after the scaling and differencing of the data is undone, so that different models with different preprocessing have comparable scores. The scoring types are:
- Mean squared error, calculated using the target values in the test set for each stop and the mean of the sampling of the predictions.
- Accuracy, which is the percentage of targets in the test set that are within the prediction intervals.

## Data handling

In the beginning, there was a limit to how large the sequence length could be, how many weights the network could contain, and how many times an observation in the test set was sampled. This was due to the fact that we loaded all the data one time. This problem was solved with a dataloader, where finished preprocessed data was stored in disc, and samples were only loaded in batches when used in the training.

## Training on GPU

As the dataloader made training take considerably more time on a CPU, there was no choice but to start training on a GPU. Here data can be loaded at the same time in batches to go through the network in parallel.