from blitz.modules import BayesianLSTM
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

import torch
from torch import nn, optim
from torch.nn.functional import relu

@variational_estimator
class LSTM_Net(nn.Module):
	def __init__(
		self,
		input_dim, 
		hidden_dim, 
		linear_dim, 
		sequence_length,
		output_dim
		):

		'''
			input_dim: input dimension (number of stops + dimension of temporal features)
			hidden_dim: hidden dimension
			linear_dim: linear layer dimension
			sequence_length: sequence_length (default: 24*14 hours i.e. two weeks)
			output_dim: output dimension (number of stops, default: 10)
		'''

		super(LSTM_Net, self).__init__()
		
		# define network constants
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.linear_dim = linear_dim
		self.sequence_length = sequence_length
		self.output_dim = output_dim

		# bayesian LSTM layer
		self.lstm1=BayesianLSTM(
			in_features=input_dim, 
			out_features=hidden_dim, 
			bias=True
		)

		# bayesian linear layer
		self.blinear1 = BayesianLinear(
			in_features=hidden_dim, 
			out_features=linear_dim,
			bias=True
		)

		# linear layer
		self.linear1 = nn.Linear(
			in_features = linear_dim,
			out_features = output_dim,
			bias = False
		)

		# dropout function
		self.dropout = nn.Dropout(p=0.5)

	def forward(self, x):
		# bayesian lstm layer
		lstm_out, _ = self.lstm1(self.dropout(x.float()))

		# reshape lstm output for linear layer
		x = lstm_out.view(self.sequence_length, len(x), self.hidden_dim)[-1]
		
		# bayesian linear layer
		x = relu(self.blinear1(self.dropout(x)))

		# linear layer
		x = self.linear1(self.dropout(x))

		return x