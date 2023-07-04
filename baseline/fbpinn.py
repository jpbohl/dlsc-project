import torch
from torch.nn import Module, ModuleList
from torch.optim import Adam, LBFGS

from nn import NeuralNet as NN


class FBPinn(Module):

    def __init__(self, problem, nwindows, domain, hidden, neurons, overlap, sigma):
        super(FBPinn, self).__init__()

        self.nwindows = nwindows
        self.problem = problem
        self.domain = domain # domain of the form torch.tensor([[a, b], [c, d], ...]) depending on dimension
        self.overlap = overlap # percentage of overlap of subdomains
        self.sigma = sigma # parameter (set) for window function
        self.subdomains = self.partition_domain()
        self.means = self.get_midpoints()
        self.std = (self.subdomains[:, 1] - self.subdomains[:, 0]) / 2

        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd
        self.models = ModuleList([NN(hidden, neurons) for _ in range(self.nwindows)])



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
        # error when overlap is 0 or smaller
        if self.overlap <= 0:
            raise ValueError("Overlap must be greater than 0.")
        
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

        return (self.subdomains[:, 1] + self.subdomains[:, 0]) / 2


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

        return midpoints

    def compute_window(self, input, iteration):
        """
        Computes window function given input points and domain and parameter sigma
        """
        # 1D case
        #tol = 1e-5
        x_left = (torch.sub(input, self.get_midpoints_overlap()[iteration])) / self.sigma
        x_right = (torch.sub(input, self.get_midpoints_overlap()[iteration+1])) / self.sigma
        # x_left = (input - self.subdomain[subdomain][0])/self.sigma
        # x_right = (input - self.subdomain[subdomain][1])/self.sigma
        
        window = torch.sigmoid(x_left) * torch.sigmoid(-x_right)
        
        #window = torch.clamp(torch.clamp(1/(1+torch.exp(x_left)), min = tol )* torch.clamp(1/(1+torch.exp(-x_right)), min = tol), min = tol)
        #window = 1/(1+torch.exp(x_left))* 1/(1+torch.exp(-x_right))
        
        return window

    def forward(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        #output for every subdomain: dimension  nwindows*input
        fbpinn_output = torch.zeros(self.nwindows,input.size(0))
        window_output = torch.zeros(self.nwindows,input.size(0))
        pred = torch.zeros_like(input)
        for i in range(self.nwindows):

            model = self.models[i] # get model i
            
            # normalize data to given subdomain
            # normalize such that input lies in [-1,1]
            input_norm = (input - self.means[i]) / self.std[i] 

            # check whether we normalised to (-1, 1)
            in_subdomain = (self.subdomains[i][0] <= input) & (input <= self.subdomains[i][1])

            assert((input_norm[in_subdomain] <= 1).all().item())
            assert((-1 <= input_norm[in_subdomain]).all().item())

            # model i prediction
            output = model(input_norm.reshape(-1,1))

            output = output * self.u_sd + self.u_mean

            # compute window function for subdomain i
            window = self.compute_window(input, i)

            ind_pred = window * output

            # add prediction to total output
            # sum neural networks in overlapping regions
            pred += ind_pred

            # add it to output tensor in row i
            # used for different plots after training
            ind_pred = self.problem.hard_constraint(input, ind_pred)
            window_output[i,] = window.reshape(1,-1)[0]
            fbpinn_output[i,] = ind_pred.reshape(1,-1)[0]

        pred = self.problem.hard_constraint(pred, input)
        
        return pred, fbpinn_output, window_output


class FBPINNTrainer:

    def __init__(self, fbpinn, lr, problem):

        self.fbpinn = fbpinn
        self.optimizer = Adam(fbpinn.parameters(),
                            lr=lr)
        self.problem = problem
        

    def train(self, nepochs, trainset): 
        print("Training FBPINN")
        
        history = list()
        
        for i in range(nepochs):
            
            for input, in trainset:

                def closure():
                    self.optimizer.zero_grad()
                    input.requires_grad_(True) # allow gradients wrt to input for pde loss
                    pred, _, __ = self.fbpinn(input)
                    loss = self.problem.compute_loss(pred, input)
                    loss.backward(retain_graph=True)
                    history.append(loss.item())

                    print(f"Loss : {loss.item()}")
                    return loss
                
            self.optimizer.step(closure=closure)

            input = next(iter(trainset))[0]
            pred, fbpinn_output, window_output = self.fbpinn(input)
        
        return pred, fbpinn_output, window_output, history
    
    def test(self):

        domain = self.problem.domain
        ntest = 1000
        points = torch.rand(ntest).reshape(-1, 1)
        points = points * (domain[1] - domain[0]) + domain[0]

        self.fbpinn.eval()
        pred, _, __ = self.fbpinn(points)
        true = self.problem.exact_solution(points)

        # check that no unwanted broadcasting occured
        assert (pred - true).numel() == ntest

        relative_L2 = torch.sqrt(torch.sum((pred - true) ** 2) / torch.sum(true ** 2))

        return relative_L2.item()

###############################################################################

# PINNs

###############################################################################


class Pinn(Module):

    def __init__(self, problem, domain, hidden, neurons):

        super(Pinn, self).__init__()
        self.domain = domain # domain of the form torch.tensor([a, b])

        self.problem = problem

        #parameter for normalize
        self.mean = (domain[1] + domain[0])/2
        self.std = (domain[1] - domain[0])/2
        
        #parameters for unnormalize
        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd

        self.model= NN(hidden, neurons)


    def forward(self, input):

        # normalize data to given subdomain
        # normalize such that input lies in [-1,1]
        input_norm = (input - self.mean) / self.std 
        
        # model prediction
        output = self.model(input_norm) 

        output = output * self.u_sd + self.u_mean
        
        output = self.problem.hard_constraint(output, input)

        return output

class PINNTrainer:

    def __init__(self, pinn, lr, problem):

        self.pinn = pinn
        self.optimizer = Adam(pinn.parameters(),
                            lr=lr)
        self.problem = problem
        

    def train(self, nepochs, trainset): 
        
        print("Training PINN")
        
        history = list()
        
        for i in range(nepochs):
            
            for input, in trainset:
                
                def closure():
                    self.optimizer.zero_grad()
                    input.requires_grad_(True) # allow gradients wrt to input for pde loss
                    pred = self.pinn(input)
                    loss = self.problem.compute_loss(pred, input)
                    loss.backward(retain_graph=True)
                    history.append(loss.item())

                    print(f"Epoch {i} // Total Loss : {loss.item()}")
                    return loss
                
                self.optimizer.step(closure=closure)
                
            


            input = next(iter(trainset))[0]
            pred = self.pinn(input)

        return pred, history

    def test(self):

        domain = self.problem.domain
        ntest = 1000
        points = torch.rand(ntest).reshape(-1, 1)
        points = points * (domain[1] - domain[0]) + domain[0]

        self.pinn.eval()
        pred = self.pinn(points)
        true = self.problem.exact_solution(points)

        # check that no unwanted broadcasting occured
        assert (pred - true).numel() == ntest

        relative_L2 = torch.sqrt(torch.sum((pred - true) ** 2) / torch.sum(true ** 2))

        return relative_L2.item()