
"""Model class for sorting numbers."""

import torch.nn as nn

class Features(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        This Feature extractor class takes an input and constructs a feature vector. It can be applied independently to all elements of the input sequence

        in_flattened_vector: input flattened vector
        latent_dim: number of neurons in latent layer
        output_dim: dimension of log alpha square matrix
        """
        super().__init__()

        # net: output of the first neural network that connects numbers to a
        # 'latent' representation.
        # activation_fn: ReLU is default hence it is specified here
        # dropout p – probability of an element to be zeroed
        self.linear1 = nn.Linear(1, latent_dim)
        self.relu1 = nn.ReLU()
        self.d1 = nn.Dropout(p = dropout_prob)
        # now those latent representation are connected to rows of the matrix
        # log_alpha.
        self.linear2 = nn.Linear(latent_dim, output_dim)
        self.d2 = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.

        x: Tensor of shape (batch_size, 1)
        """
        # activation_fn: ReLU
        x = self.d1(self.relu1(self.linear1(x)))
        # no activation function is enabled
        x = self.d2(self.linear2(x))
        return x

class Sinkhorn_Net(nn.Module):

    def __init__(self, latent_dim, output_dim, dropout_prob):
        super().__init__()
        self.output_dim = output_dim
        self.features = Features(latent_dim, output_dim, dropout_prob)
    def forward(self, x):
        """
        x: Tensor of length (batch, sequence_length)
        Note that output_dim should correspond to the intended sequence length
        """
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
        x = x.view(-1, 1)
        x = self.features(x)
        #reshape to cubic for sinkhorn operation
        x = x.reshape(-1, self.output_dim, self.output_dim)
        return x
