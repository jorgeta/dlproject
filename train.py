import torch
from torch import nn, optim
from torch.autograd import Variable
from data_handler import Dataset
import numpy as np
from torch.optim.lr_scheduler import StepLR

def train_model(
    model, 
    max_epochs,
    params,
    partition,
    elbo_sample_nbr,
    ):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model.to(device)

    # Generators
    training_set = Dataset(partition['train'])
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['test'])
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # define criterion and optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1.0)
    #optimiser = optim.SGD(model.parameters(), lr=1e-3)
    
    # store losses per epoch
    train_losses = np.zeros(max_epochs)
    test_losses = np.zeros(max_epochs)

    scheduler = StepLR(optimiser, step_size=15)

    for epoch in range(max_epochs):
        # Training
        counter = 0
        current_loss_sum = 0
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            optimiser.zero_grad()

            # forward and backward propagation
            batch_loss_elbo = model.sample_elbo(
                inputs=local_batch.float(),
                labels=local_labels.float(),
                criterion=criterion,
                sample_nbr=elbo_sample_nbr
            )

            batch_loss_elbo.backward()

            optimiser.step()

            output = model(local_batch)

            current_train_loss = criterion(output, local_labels.float())
            current_loss_sum += current_train_loss.item()
            
            counter += 1
        train_losses[epoch] = current_loss_sum / counter

        # Validation
        counter = 0
        current_loss_sum = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # Model computations
                output = model(local_batch.float())
                
                current_test_loss = criterion(output, local_labels.float())
                current_loss_sum += current_test_loss.item()

                counter += 1
            test_losses[epoch] = current_loss_sum / counter
        
        scheduler.step()

        # Output losses after each epoch
        print(f'Epoch {epoch+1} train loss: {train_losses[epoch]}, test loss: {test_losses[epoch]}')

    return model, train_losses, test_losses