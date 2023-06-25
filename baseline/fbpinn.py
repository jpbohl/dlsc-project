import torch
from torch.nn import Module

from nn import NeuralNet as NN

def partition_domain(domain):
    """
    Given an interval, splits it into evenly sized
    overlapping subintervals. 

    First just focus on 1 dimension.

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


    def get_midpoints_overlap(self, subdomains):
        """
        Gets the midpoint of left and right overlapping domain 
        for window function later.
        """

        raise NotImplementedError

    def compute_window(self, input, domain):
        """
        Computes window function given input points and domain
        """

        #TODO Implement window function
        #given by sigmoid function 
        #needs midpoint of right and left overlapping domain

        raise NotImplementedError

    def un_norm(self):
        """
        
        Get common function for all unnormalization of neural networks
        such that all outputs stay within [-1,1]
        """

        raise NotImplementedError

    def forward(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        pred = torch.zeros_like(input)
        for i in range(self.nwindows):

            model = self.models[i] # get model i

            #TODO: get input data of subdomain
            
            # normalize data to given subdomain
            # normalize such that input lies in [-1,1]
            input_norm = (input - self.means[i]) / self.std 
            
            # model i prediction
            output = model(input_norm) 

            # TODO: Implement unnormalizing data
            # use function un_norm as we need common unnormalizing for all NN
            
            # compute window function for subdomain i
            subdomain = self.subdomains[i, :]
            window = self.compute_window(input, subdomain)

            # add prediction to total output
            pred += window * output


        #TODO: sum neural networks in overlapping regions

        return pred
    