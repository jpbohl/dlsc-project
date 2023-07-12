import torch
import torch.nn as nn

total_params = lambda model: sum(p.numel() for p in model.parameters())

class NeuralNet(nn.Module):

    def __init__(self, n_hidden_layers, neurons, input_dimension=1, output_dimension=1, regularization_param=0, regularization_exp=2, retrain_seed=0, dropout=0.0):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

        d1,d2,h,l = input_dimension, output_dimension, neurons, n_hidden_layers
        self.size =           d1*h + h         + (l-1)*(  h*h + h)        +   h*d2 + d2
        self._single_flop = 2*d1*h + h + 5*h   + (l-1)*(2*h*h + h + 5*h)  + 2*h*d2 + d2 # assumes Tanh uses 5 FLOPS
        self.flops = lambda BATCH_SIZE: BATCH_SIZE*self._single_flop
        assert self.size == total_params(self)


    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            if self.dropout > 0.0: 
                x = self.dropout_layer(l(x))
                x = self.activation(x)
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss