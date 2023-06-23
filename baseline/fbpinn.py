import torch
from torch.nn import Module

from nn import NeuralNet as NN

def partition_domain(domain):
    """
    Given an interval, splits it into evenly sized
    overlapping subintervals. 
    Input: 
        domain (tensor) : start and end point of domain
    Return:
        subdomains (tensor) : k x 2 tensor containing 
            the start and end points of the subdomains
    """

    #TODO Implement domain partioning

    raise NotImplementedError


class FBPinn(Module):

    def __init__(self, nwindows, domain, hidden, neurons):
        super(FBPinn, self).__init__()

        self.nwindows = nwindows
        self.domain = domain
        self.subdomains = partition_domain(self.domain)
        self.means = self.get_midpoints(self.subdomains)
        
        self.models = [NN(hidden, neurons) for _ in range(self.nwindows)]
        
        raise NotImplementedError

    def get_midpoints(self, subdomains):
        """
        Gets the midpoint of each subdomain for subdomain
        normalization. 
        """
        return (subdomains[:, 1] - subdomains[:, 0]) / 2

    def compute_window(self, input, domain):
        """
        Computes window function given input points and domain
        """

        #TODO Implement window function

        raise NotImplementedError

    def forward(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        pred = torch.zeros_like(input)
        for i in range(self.nwindows):

            model = self.models[i] # get model i
            
            # normalize data to given subdomain
            input_norm = (input - self.means[i]) / self.std 
            
            # model i prediction
            output = model(input_norm) 

            # TODO: Implement unnormalizing data
            
            # compute window function for subdomain i
            subdomain = self.subdomains[i, :]
            window = self.compute_window(input, subdomain)

            # add prediction to total output
            pred += window * output

        return pred