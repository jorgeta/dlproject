from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import os
from data_handler import Datasaver

class Preprocessor():
    def __init__(
        self, 
        name: str,
        sequence_length: int, # length of sequence of data per sample
        test_set_length: int, # size of test set in hours
        difference_length: int, # predicting (data at time t) - data at time (t-difference_length)
        use_difference = True, # whether to difference the data
        path_to_data_folder: str = 'raw_data/', # path to the raw data
        bus_nr: str = '150', # 10, 15, 150, or 300
        bus_direction: bool = True, # for bus_nr=150: northwards (True), southwards (False)
        passenger_amount: bool = False, # predict amount (True) or change in amount (False) of passengers
        use_temporal_features: bool = True
    ):
        self.sequence_length = sequence_length
        self.test_set_length = test_set_length
        self.difference_length = difference_length
        self.use_difference = use_difference
        self.path_to_data_folder = path_to_data_folder
        self.bus_nr = bus_nr
        self.bus_direction = bus_direction
        self.passenger_amount = passenger_amount
        self.use_temporal_features = use_temporal_features

        self.name = name
        self.path = f'storage/{name}'
        os.makedirs(self.path, exist_ok=True)

        if not self.use_difference:
            self.difference_length = 0

        # raw data
        self.od_demand = None
        self.od_time = None
        self.od_stop = None

        # data through steps
        self.boarding_data = None
        self.temporal_data = None

        # keeping track
        self.raw_data_is_fetched = False
        self.temporal_data_is_initialized = False
        self.boarding_data_is_initialized = False
        self.data_is_splitted = False
        self.data_is_concatenated = False

        # do basic first step preprocessing
        self.get_raw_data()
        self.create_person_density_array()
        if self.use_temporal_features:
            self.temporal_one_hot_encoder()
        self.train_test_split()

    def get_raw_data(self):
        print("Fetching raw data...")

        od_demand = np.load(f'{self.path_to_data_folder}od-{self.bus_nr}-1H-20201125-120-demand.npy')
        od_time = np.load(f'{self.path_to_data_folder}od-{self.bus_nr}-1H-20201125-120-time.npy')
        od_time = pd.to_datetime(od_time)
        od_time = od_time[24:]
        od_demand = od_demand[24:]
        od_stop = np.load(f'{self.path_to_data_folder}od-{self.bus_nr}-1H-20201125-120-stop.npy', allow_pickle=True)
        od_stop[9] = 'NPST1'

        self.od_demand = od_demand
        self.od_time = od_time
        self.od_stop = od_stop

        self.raw_data_is_fetched = True

    def create_person_density_array(self):
        if not self.raw_data_is_fetched:
            print("Warning: Raw data needs to be initialied first.")
            self.get_raw_data()
        
        print("Creating boarding data using od_demand...")

        person_density_array_all_hours = []
        for hour in range(len(self.od_demand)):
            person_density_array = np.zeros(len(self.od_stop))
            if self.bus_direction:
                for i in range(len(self.od_stop)):
                    people_on = np.sum(self.od_demand[hour][-1-i][:len(self.od_stop)-i])
                    people_off = np.sum(self.od_demand[hour].T[-1-i][len(self.od_stop)-i-1:])
                    net_addition_of_people = people_on - people_off
                    if not self.passenger_amount:
                        person_density_array[i] = net_addition_of_people
                    else:
                        if i == 0:
                            person_density_array[i] = net_addition_of_people
                        else:
                            person_density_array[i] = person_density_array[i-1] + net_addition_of_people
            else:
                for i in range(len(self.od_stop)):
                    people_on = np.sum(self.od_demand[hour][i][i:len(self.od_stop)])
                    people_off = np.sum(self.od_demand[hour].T[i][:i+1])
                    net_addition_of_people = people_on - people_off
                    if not self.passenger_amount:
                        person_density_array[i] = net_addition_of_people
                    else:
                        if i == 0:
                            person_density_array[i] = net_addition_of_people
                        else:
                            person_density_array[i] = person_density_array[i-1] + net_addition_of_people
            person_density_array_all_hours.append(person_density_array)
        
        self.boarding_data = np.array(person_density_array_all_hours)
        self.boarding_data_is_initialized = True

    def temporal_one_hot_encoder(self):
        if not self.raw_data_is_fetched:
            print("Warning: Raw data needs to be initialied first.")
            self.get_raw_data()
        
        print("Creating temporal data using od_time...")
        if self.use_temporal_features:
            temporal_features = []
            for i in range(len(self.od_time)):
                temporal_features.append([self.od_time[i].dayofweek, self.od_time[i].hour])
            
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(temporal_features)

            self.temporal_data = np.array(enc.transform(temporal_features).toarray())
        self.temporal_data_is_initialized = True

    def train_test_split(self):
        if not self.temporal_data_is_initialized or not self.boarding_data_is_initialized:
            print("Warning: both boarding data and temporal data needs to be initialized.")
            self.create_person_density_array()
            if self.use_temporal_features:
                self.temporal_one_hot_encoder()
        if not self.data_is_splitted:
            print("Splitting data into a training and a test set...")

            self.train_set_length = len(self.boarding_data)-self.test_set_length

            self.train_boarding_data = self.boarding_data[:self.train_set_length]
            self.test_boarding_data = self.boarding_data[-self.test_set_length:]

            if self.use_temporal_features:
                self.train_temporal_data = self.temporal_data[:self.train_set_length]
                self.test_temporal_data = self.temporal_data[-self.test_set_length:]

            self.data_is_splitted = True
        else:
            print("Data is already splitted.")

    def difference(self):
        if not self.data_is_splitted:
            print("Warning: data is not splitted yet.")
            self.train_test_split()
        if self.use_difference:
            print(f"Differencing data using difference length {self.difference_length}...")

            cat = np.concatenate((self.train_boarding_data, self.test_boarding_data))

            data_diffed = np.zeros(cat.shape)
            for stop in range(cat.shape[1]):
                for timepoint in range(self.difference_length, cat.shape[0]):
                    data_diffed[timepoint][stop] = cat[timepoint][stop]
                    data_diffed[timepoint][stop] -= cat[timepoint-self.difference_length][stop]

            self.train_boarding_data = data_diffed[self.difference_length:len(self.train_boarding_data)]
            self.test_boarding_data = data_diffed[-len(self.test_boarding_data):]

            if self.use_temporal_features:
                if len(self.train_temporal_data) > len(self.train_boarding_data):
                    self.train_temporal_data = self.train_temporal_data[self.difference_length:]
        else:
            print("Skipping differencing.")

    def save_scaler(self, scaler):
        try:
            print('Saving scaler to file...')
            pickle_out = open(f"{self.path}/scaler.pickle", "wb")
            pickle.dump(scaler, pickle_out)
            pickle_out.close()
        except:
            print('Scaler is not yet defined.')

    def standard_scale(self):
        '''Scale data by subtracting mean and dividing by standard deviation.'''
        if not self.data_is_splitted:
            print("Warning: data is not splitted yet.")
            self.train_test_split()

        print("Scaling data using StandardScaler...")

        # initiate scaler
        scaler = StandardScaler()
        scaler = scaler.fit(np.expand_dims(self.train_boarding_data.flatten(), axis=1))

        # scale data
        train_data_scaled = scaler.transform(np.expand_dims(self.train_boarding_data.flatten(), axis=1))
        test_data_scaled = scaler.transform(np.expand_dims(self.test_boarding_data.flatten(), axis=1))

        # reshape to original form
        train_data_scaled = train_data_scaled.reshape(len(self.train_boarding_data), -1)
        test_data_scaled = test_data_scaled.reshape(len(self.test_boarding_data), -1)

        self.train_boarding_data = train_data_scaled
        self.test_boarding_data = test_data_scaled

        self.save_scaler(scaler)

    def concatenate(self):
        if not self.data_is_concatenated:
            if self.use_temporal_features:
                self.train_data = np.concatenate((self.train_boarding_data, self.train_temporal_data), axis=1)
                self.test_data = np.concatenate((self.test_boarding_data, self.test_temporal_data), axis=1)
            else:
                self.train_data = self.train_boarding_data
                self.test_data = self.test_boarding_data
            self.data_is_concatenated = True
        else:
            print("Data is already concatenated.")

    def generate_sequences(self):
        '''Generates sequences of the data.'''

        print(f"Generating sequences of the data with sequence length {self.sequence_length}...")

        cat = np.concatenate((self.train_data, self.test_data))

        inputs = []
        targets = []
        for i in range(len(cat) - self.sequence_length):
            x = cat[i:(i+self.sequence_length)]
            y = cat[i+self.sequence_length]
            inputs.append(x)
            targets.append(y)
        
        if self.use_temporal_features:
            temporal_parts_of_targets = [i for i in range(10, cat.shape[1])]
            targets = np.delete(np.array(targets), temporal_parts_of_targets, axis=1)

        self.train_set_length = len(targets)-self.test_set_length
        
        self.X_train = torch.tensor(inputs[:self.train_set_length])
        self.y_train = torch.tensor(targets[:self.train_set_length])
        self.X_test = torch.tensor(inputs[self.train_set_length:])
        self.y_test = torch.tensor(targets[self.train_set_length:])

    def store_hyperparameters(self):
        hyper_params = {
            'sequence_length': self.sequence_length,
            'train_set_length': len(self.X_train),
            'test_set_length': self.test_set_length,
            'difference_length': self.difference_length,
            'bus_nr': self.bus_nr,
            'bus_direction': self.bus_direction,
            'passenger_amount': self.passenger_amount
        }
        pickle_out = open(f"{self.path}/hyperparameters.pickle", "wb")
        pickle.dump(hyper_params, pickle_out)
        pickle_out.close()

def preprocess(
    storage_name,
    sequence_length = 24*7,
    test_set_length = 24*28,
    path_to_data_folder = 'raw_data/',
    difference_length = 24*7,
    use_difference = True,
    bus_nr = '150',
    bus_direction = True,
    passenger_amount = True,
    use_temporal_features = True
    ):

    # preprocessing defining variables
    

    # initialize the preprocessor
    pp = Preprocessor(
        storage_name,
        sequence_length,
        test_set_length,
        difference_length,
        use_difference,
        path_to_data_folder,
        bus_nr,
        bus_direction,
        passenger_amount,
        use_temporal_features
    )

    # difference the data
    pp.difference()

    # use standard scaling on the data
    pp.standard_scale()

    # concatenate borading data and temporal data
    pp.concatenate()

    # create sequences for the LSTM network
    pp.generate_sequences()

    # store hyperparameters for later use
    pp.store_hyperparameters()

    # save data so dataloader can use it
    datasaver = Datasaver(
        pp.name,
        pp.X_train, 
        pp.y_train, 
        pp.X_test, 
        pp.y_test,
        pp.boarding_data,
        pp.difference_length,
        pp.sequence_length
    )

    return pp