import torch
from torch.nn import Module

from nn import NeuralNet as NN




class FBPinn(Module):

    def __init__(self, nwindows, domain, hidden, neurons, u_mean=0, u_sd=1):
        super(FBPinn, self).__init__()

        self.nwindows = nwindows
        self.domain = domain
        self.subdomains = self.partition_domain()
        self.means = self.get_midpoints()

        self.u_mean = u_mean
        self.u_sd = u_sd
        
        self.models = [NN(hidden, neurons) for _ in range(self.nwindows)]
        
        raise NotImplementedError


    ###  Task Allebasi
    def partition_domain(self):
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


    def get_midpoints(self):
        """
        Gets the midpoint of each subdomain for subdomain
        normalization. 
        """
        self.subdomain

        return (subdomains[:, 1] - subdomains[:, 0]) / 2


    def get_midpoints_overlap(self):
        """
        Gets the midpoint of left and right overlapping domain 
        for window function later.
        """

        raise NotImplementedError

    def compute_window(self, input):
        """
        Computes window function given input points and domain
        """
        self.domain

        #TODO Implement window function
        #given by sigmoid function 
        #needs midpoint of right and left overlapping domain
        # + set of parameters (see paper first)

        raise NotImplementedError
    

    def forward(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        pred = torch.zeros_like(input)
        for i in range(self.nwindows):

            model = self.models[i] # get model i
            
            # normalize data to given subdomain
            # normalize such that input lies in [-1,1]
            input_norm = (input - self.means[i]) / self.std 
            
            # model i prediction
            output = model(input_norm) 

            output = output * self.u_sd + self.u_mean

            # compute window function for subdomain i
            subdomain = self.subdomains[i, :]
            window = self.compute_window(input, subdomain)

            # add prediction to total output
            # sum neural networks in overlapping regions
            pred += window * output

        return pred