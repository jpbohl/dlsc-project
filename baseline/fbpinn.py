import torch
from torch.nn import Module

from nn import NeuralNet as NN




class FBPinn(Module):

    def __init__(self, nwindows, domain, hidden, neurons, overlap, sigma, u_mean=0, u_sd=1):
        super(FBPinn, self).__init__()

        self.nwindows = nwindows
        self.domain = domain # domain of the form torch.tensor([[a, b], [c, d], ...]) depending on dimension
        self.overlap = overlap # percentage of overlap of subdomains
        self.sigma = sigma # parameter (set) for window function
        self.subdomains = self.partition_domain()
        self.means = self.get_midpoints()
        self.std = (self.subdomains[:, 1] - self.subdomains[:, 0]) / 2

        self.u_mean = u_mean
        self.u_sd = u_sd
        self.models = [NN(hidden, neurons) for _ in range(self.nwindows)]

        #set the parameters to optimize for later
        params=[]
        for i in range(len(self.models)):
            params+=list(self.models[i].parameters())


        self.params= params
        
        #raise NotImplementedError


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
                the start and end points of the subdomains with equal overlap on all sides
        """
        subdomains = torch.zeros(self.nwindows, 2)
        
        #problem for 1d: width = (self.domain[0][1]-self.domain[0][0]) / self.nwindows
        width = (self.domain[1]-self.domain[0]) / self.nwindows
        for i in range(self.nwindows):
            #subdomains[i][0] = self.domain[0][0] + (i-self.overlap/2) * width if i != 0 else self.domain[0][0]
            #subdomains[i][1] = self.domain[0][0] + (i+1+self.overlap/2) * width if i != (self.nwindows-1) else self.domain[0][1]
            subdomains[i][0] = self.domain[0] + (i-self.overlap/2) * width if i != 0 else self.domain[0]
            subdomains[i][1] = self.domain[0] + (i+1+self.overlap/2) * width if i != (self.nwindows-1) else self.domain[1]
        
        return subdomains

        #raise NotImplementedError


    def get_midpoints(self):
        """
        Gets the midpoint of each subdomain for subdomain
        normalization. 
        """

        return torch.tensor((self.subdomains[:, 1] + self.subdomains[:, 0]) / 2 )


    def get_midpoints_overlap(self):
        """
        Gets the midpoint of left and right overlapping domain 
        for window function later.
        """

        #initialize midpoints, edges of domain are not overlapping
        midpoints = torch.zeros(self.nwindows+1)
        midpoints[0] = self.subdomains[0][0]
        #midpoints[self.nwindows] = self.subdomains[self.nwindows][1]
        midpoints[self.nwindows] = self.subdomains[self.nwindows-1][1]

        #compute midpoints of overlapping interior domains
        # we have self.nwindows -1 overlapping regions 
        #begin with 0 end with end of domain
        for i in range(1,self.nwindows):
            midpoints[i] = (self.subdomains[i-1][1] + self.subdomains[i][0]) / 2 

        return torch.tensor(midpoints)
        #raise NotImplementedError

    def compute_window(self, input, iteration):
        """
        Computes window function given input points and domain and parameter sigma
        """
        # 1D case
        
        x_left = (torch.sub(input, self.get_midpoints_overlap()[iteration]))/self.sigma
        x_right = (torch.sub(input, self.get_midpoints_overlap()[iteration+1]))/self.sigma
        # x_left = (input - self.subdomain[subdomain][0])/self.sigma
        # x_right = (input - self.subdomain[subdomain][1])/self.sigma
        
        window = 1/(1+torch.exp(x_left)) * 1/(1+torch.exp(-x_right))
        
        
        return window

        #raise NotImplementedError
    

    def forward(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        pred = torch.zeros_like(input)
        for i in range(self.nwindows):

            model = self.models[i] # get model i
            
            # normalize data to given subdomain
            # normalize such that input lies in [-1,1]
            input_norm = torch.sub(input,self.means[i]) / self.std[i] 
            
            #caro: woher kommt self.std ?

            # model i prediction
            output = model(input_norm.reshape(-1,1)) 

            output = output * self.u_sd + self.u_mean

            # compute window function for subdomain i
            subdomain = self.subdomains[i, :]
            window = self.compute_window(input, i)

            # add prediction to total output
            # sum neural networks in overlapping regions
            pred += window * output

        return pred


class Pinn(Module):

    def __init__(self, domain, hidden, neurons, u_mean=0, u_sd=1):

        self.domain = domain # domain of the form torch.tensor([a, b])

        #parameter for normalize
        self.mean= (domain[1] + domain[0])/2
        self.std = (domain[1] - domain[0])/2
        
        #parameters for unnormalize
        self.u_mean= u_mean
        self.u_sd=u_sd

        super().__init__()
        self.model= NN(hidden, neurons)


    def forward(self, input):


        pred = torch.zeros_like(self.domain)

        model = self.model
        
        # normalize data to given subdomain
        # normalize such that input lies in [-1,1]
        input_norm = (input - self.mean) / self.std 
        
        # model prediction
        output = model(input_norm) 

        output = output * self.u_sd + self.u_mean

        return pred