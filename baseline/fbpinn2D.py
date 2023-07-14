import torch
from torch.nn import Module, ModuleList
from torch.optim import Adam, LBFGS

import numpy as np

from multiprocessing import Pool

from nn import NeuralNet as NN

class FBPinn(Module):

    def __init__(self, problem, nwindows, domain, hidden, neurons, overlap, sigma, manual_part=[]):
        super(FBPinn, self).__init__()

        self.nwindows = nwindows
        self.problem = problem
        self.domain = domain # domain of the form torch.tensor([[a, b], [c, d], ...]) depending on dimension
        self.overlap = overlap # percentage of overlap of subdomains
        self.sigma = sigma # parameter (set) for window function

        self.manual_part=manual_part
        if len(self.manual_part)==0:
            self.subdomains = self.partition_domain()
        else:
            self.subdomains = self.manual_partition()     

        self.means = self.get_midpoints()
        self.subdomains_row, self.subdomains_col = torch.split(self.subdomains, self.nwindows)
        self.stdrow = (self.subdomains_row[:, 1] - self.subdomains_row[:, 0]) / 2
        self.stdcol = (self.subdomains_col[:, 1] - self.subdomains_col[:, 0]) / 2
        self.std = torch.cartesian_prod(self.stdrow, self.stdcol)

        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd
        self.models = ModuleList([NN(hidden, neurons, 2) for _ in range(self.nwindows[0]*self.nwindows[1])])

    def manual_partition(self):
        """
        Given a list of subinterval endpoints, creates 
        overlapping subintervals. 

        Works in 1 dimension. Note that the overlap is given in absolute values.

        Input: 
            manual partition (tensor) : list with midpoints of the desired subintervals
        Return:
            subdomains (tensor) : k x 2 tensor containing 
                the start and end points of the subdomains with equal overlap on all sides
        """
        assert self.nwindows == len(self.manual_part) + 1

        partition= self.manual_part.copy()

        partition.insert(0,self.domain[0].item())
        partition.insert(len(partition),self.domain[1].item())

        width= torch.zeros(self.nwindows, 1)
        for i in range(self.nwindows):
            width[i]= partition[i+1]-partition[i]

        subdomains = torch.zeros(self.nwindows, 2)
        for i in range(self.nwindows):
            subdomains[i][0] = partition[i]- self.overlap/2 if i != 0 else partition[0]
            subdomains[i][1] = partition[i+1]+ self.overlap/2 if i != (self.nwindows-1) else partition[-1]
        #do not need to run midpoints (should be the same)
        
        return subdomains

    def partition_domain(self):
        """
        Given an interval, splits it into evenly sized
        overlapping subintervals. 

        First just focus on 1 dimension. Note that the overlap is given in relative values.

        Input: 
            domain (tensor) : start and end point of domain
        Return:
            subdomains (tensor) : k x 2 tensor containing 
                the start and end points of the subdomains with equal overlap on all sides
        """
        # error when overlap is 0 or smaller
        if self.overlap <= 0:
            raise ValueError("Overlap must be greater than 0.")
        #check if nwindows is not int or tuple otherwise raise error
        if not isinstance(self.nwindows, tuple):
            raise ValueError("nwindows must be an integer or a tuple of integers.")
            
        subdomains = torch.zeros(sum(self.nwindows), 2)
        width = ((self.domain[0][1]-self.domain[0][0]) / self.nwindows[0] , (self.domain[1][1]-self.domain[1][0] )/ self.nwindows[1]) 
        for i in range(self.nwindows[0]):
            subdomains[i][0] = self.domain[0][0] + (i-self.overlap/2) * width[0] if i != 0 else self.domain[0][0]
            subdomains[i][1] = self.domain[0][0] + (i+1+self.overlap/2) * width[0] if i != (self.nwindows[0]-1) else self.domain[0][1]
        for j in range(self.nwindows[1]):
            subdomains[j+self.nwindows[0]][0] = self.domain[1][0] + (j-self.overlap/2) * width[1] if j != 0 else self.domain[1][0]
            subdomains[j+self.nwindows[0]][1] = self.domain[1][0] + (j+1+self.overlap/2) * width[1] if j != (self.nwindows[1]-1) else self.domain[1][1]
        
        return subdomains

    def get_midpoints(self):
        """
        Gets the midpoint of each subdomain for subdomain
        normalization. 
        """
        subdomains_row, subdomains_col = torch.split(self.subdomains, self.nwindows)
        midpoints_row = (subdomains_row[:, 1] + subdomains_row[:, 0]) / 2
        midpoints_col = (subdomains_col[:, 1] + subdomains_col[:, 0]) / 2
        midpoints = torch.cartesian_prod( midpoints_row, midpoints_col)
       
        return midpoints

    def get_midpoints_overlap(self):
        """
        Gets the midpoint of left and right overlapping domain 
        for window function later.
        """
       
        #initialize midpoints, edges of domain are not overlapping
        midpoints_row = torch.zeros(self.nwindows[0]+1)
        midpoints_col = torch.zeros(self.nwindows[1]+1)
        midpoints_row[0] = self.subdomains[0][0]
        midpoints_row[self.nwindows[0]] = self.subdomains[self.nwindows[0]-1][1]
        midpoints_col[0] = self.subdomains[0][0]
        midpoints_col[self.nwindows[1]] = self.subdomains[self.nwindows[0]+self.nwindows[1]-1][1]
            
        for i in range(1,self.nwindows[0]):
            midpoints_row[i] = (self.subdomains[i-1][1] + self.subdomains[i][0]) / 2
        for j in range(1,self.nwindows[1]):
            midpoints_col[j] = (self.subdomains[self.nwindows[0]+j-1][1] + self.subdomains[self.nwindows[0]+j][0]) / 2
        midpoints = torch.cartesian_prod(midpoints_row, midpoints_col)
            
        return midpoints

    def compute_window(self, input, iteration):
        """
        Computes window function given input points and domain and parameter sigma
        """
        
        r, c = divmod(iteration, self.nwindows[1])
        r = r * (self.nwindows[0]+1)
            
        #print(self.get_midpoints_overlap(), self.subdomains, self.get_midpoints())
        x_left = (torch.sub(input[:,0], self.get_midpoints_overlap()[r+c][0])) / self.sigma
        x_right = (torch.sub(input[:,0], self.get_midpoints_overlap()[r+c+1+self.nwindows[1]][0])) / self.sigma
        y_left = (torch.sub(input[:,1], self.get_midpoints_overlap()[r+c][1])) / self.sigma
        y_right = (torch.sub(input[:,1], self.get_midpoints_overlap()[r+c+1][1])) / self.sigma
        window = torch.sigmoid(x_left) * torch.sigmoid(-x_right) * torch.sigmoid(y_left) * torch.sigmoid(-y_right)
        # print("check", self.get_midpoints_overlap()[r+c][0], self.get_midpoints_overlap()[r+c+1+self.nwindows[1]][0], 
        #       self.get_midpoints_overlap()[r+c][1], self.get_midpoints_overlap()[r+c+1][1], iteration, r, c )
        #print("midpoints", self.get_midpoints_overlap()) 
        #print("subdomains", self.subdomains) 
        assert self.get_midpoints_overlap()[r+c][0] < self.get_midpoints_overlap()[r+c+1+self.nwindows[1]][0]
        assert self.get_midpoints_overlap()[r+c][1] < self.get_midpoints_overlap()[r+c+1][1]    
        #window = torch.clamp(torch.clamp(1/(1+torch.exp(x_left)), min = tol )* torch.clamp(1/(1+torch.exp(-x_right)), min = tol), min = tol)
        #window = 1/(1+torch.exp(x_left))* 1/(1+torch.exp(-x_right))
        
        return window


    def get_active_inputs(self, input, active_models):
        """
        Gets inputs relevant to training the currently
        active models.
        """

        active_inputs = torch.zeros_like(input, dtype=bool)
       
        for i in active_models:
            row, col = divmod(i, self.nwindows[0])
            col = col + self.nwindows[0]
                
            subdomain = torch.cat(( self.subdomains[row, :], self.subdomains[col, :]), dim = 0)

            # turn inputs in subdomain i to active
            active_inputs |= (input[:,0] <= subdomain[0][1]) & (subdomain[0][0] <= input[:,0]) & (input[:,1] <= subdomain[1][1]) & (subdomain[1][0] <= input[:,1])

        return input[active_inputs].reshape(-1, 2)

    def forward(self, input, active_models=None):
        """
        Computes forward pass of FBPinn model 
        """
        
        # differentiate between 1d and 2d case
        if active_models is None:
            active_models = np.arange(self.nwindows[0] * self.nwindows[1]).tolist()
        
        pred = torch.zeros_like(input[:,0])
            
        flops = 0
        
        for i in active_models:

            model = self.models[i] # get model i
            window = self.compute_window(input, i)
            
            row, col = divmod(i, self.nwindows[1])
            col = col + self.nwindows[0]         
           
            # get index for points which are in model i subdomain
            in_subdomain = (self.subdomains[row][0] < input[:,0]) & (input[:,0] < self.subdomains[row][1]) & (self.subdomains[col][0] < input[:,1]) & (input[:,1] < self.subdomains[col][1])
            
            # normalize data to given subdomain and extract relevant points
            input_norm = ((input[in_subdomain] - self.means[i]) / self.std[i]).reshape(-1, 2)

            # model i prediction
            output = model(input_norm).reshape(-1)            
            output = output * self.u_sd + self.u_mean
            
            # add prediction to total output
            pred[in_subdomain] += window[in_subdomain] * output
            
            #add the number of flops for each trained network on subdomain
            flops += model.flops(input_norm.shape[0])
            
        pred = self.problem.hard_constraint(pred, input)
        
        return pred, flops
    
    
    def plotting_data(self, input):
        """
        Computes forward pass of FBPinn model such that individual predictions can
        be easily plotted
        """

        #output for every subdomain: dimension  nwindows*input
        fbpinn_output = torch.zeros(self.nwindows[0] * self.nwindows[1], input.size(0))
        window_output = torch.zeros(self.nwindows[0] * self.nwindows[1], input.size(0))
        pred = torch.zeros_like(input[:,0])
        flops = 0
            
        for i in range(self.nwindows[0] * self.nwindows[1]):

            model = self.models[i]
                
            # normalize data to given subdomain and extract relevant points
            input_norm = ((input - self.means[i]) / self.std[i]).reshape(-1, 2)
            output = model(input_norm)
            output = output.squeeze()
            output = output * self.u_sd + self.u_mean
           
            window = self.compute_window(input, i)
            ind_pred = window * (output.squeeze())
            pred += ind_pred
            
            ind_pred = self.problem.hard_constraint(ind_pred, input)
            window_output[i,] = window.reshape(1,-1)[0]
            fbpinn_output[i,] = ind_pred.reshape(1,-1)[0]
            flops += model.flops(input_norm.shape[0])
        
        pred = self.problem.hard_constraint(pred, input)

        return pred, fbpinn_output, window_output, flops

class FBPINNTrainer:

    def __init__(self, fbpinn, lr, problem, optim='adam'):
        self.fbpinn = fbpinn
        self.lr = lr
        if optim == 'adam':
            self.optimizer = Adam(fbpinn.parameters(),
                                lr=lr)
        else:
            self.optimizer = LBFGS(fbpinn.parameters(),
                                        lr=float(0.5),
                                        max_iter=20,
                                        max_eval=50000,
                                        history_size=150,
                                        line_search_fn="strong_wolfe",
                                        tolerance_change=1.0 * np.finfo(float).eps)
        
        self.problem = problem
        
    def train(self, nepochs, trainset, active_models=None): 
        '''
        Implements training of the FBPiNN by optimizing according
        to the log L2 loss of the pde_residual 
        '''

        print("Training FBPINN")
        
        history = list()
        flops_history = list() 
        for i in range(nepochs):
            
            for input_, in trainset:

                def closure():
                    self.optimizer.zero_grad()
                    input_.requires_grad_(True) # allow gradients wrt to input for pde loss
                    pred, flops = self.fbpinn(input_, active_models=active_models)
                    loss = self.problem.compute_loss(pred, input_)
                    loss.backward(retain_graph=True)

                    flops_history.append(flops)
                    history.append(self.test())

                    print(f"Epoch {i} // Total Loss : {loss.item()}")
                    return loss

                self.optimizer.step(closure=closure)

            input = next(iter(trainset))[0]
            pred, _ = self.fbpinn(input)

        flops_history = np.cumsum(flops_history)
        flops_history = flops_history.tolist()
        
        return pred, history, flops_history 

    def test(self):
        '''
        Calculate testing error of predictions compared to true solution
        in terms of relative L2 error. 
        '''

        domain = self.problem.domain
        ntest = 1000   
        
        points = torch.rand(ntest, 2)
        points = points * (domain[:,1] - domain[:,0]) + domain[:,0]

        self.fbpinn.eval()
        pred, flops = self.fbpinn(points)
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
        self.mean = (domain[:,1] + domain[:,0])/2
        self.std = (domain[:,1] - domain[:,0])/2
        
        #parameters for unnormalize
        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd

        self.model= NN(hidden, neurons, 2)


    def forward(self, input):
        '''
        Calculate general forward pass for the PiNN models.
        (Generally the same as for FBPiNNs)
        '''
        
        # normalize data to given subdomain
        # normalize such that input lies in [-1,1]
        input_norm = (input - self.mean) / self.std 
        
        # model prediction and unnormalization
        pred = self.model(input_norm) 
        pred = pred.squeeze()
        pred = pred * self.u_sd + self.u_mean
        
        
        # apply hard constraint
        output = self.problem.hard_constraint(pred, input)

        # compute flops of forward pass
        flops = self.model.flops(input_norm.shape[0])

        return output, flops


class PINNTrainer:

    def __init__(self, pinn, lr, problem, optim = 'adam'):
        self.pinn = pinn
        if optim== 'adam':
            self.optimizer = Adam(pinn.parameters(),
                                lr=lr)
        else:           
            self.optimizer = LBFGS(pinn.parameters(),
                                lr=float(0.5),
                                max_iter=50000,
                                max_eval=50000,
                                history_size=150,
                                line_search_fn="strong_wolfe",
                                tolerance_change=0.5 * np.finfo(float).eps)
        
        self.problem = problem
        
    def train(self, nepochs, trainset): 
        '''
        Implements training of the PiNN by optimizing according
        to the log L2 loss of the pde_residual 
        '''
        print("Training PINN")
        
        history = list()
        flops_history = list() 

        for i in range(nepochs):
            
            for input, in trainset:
                
                def closure():
                    self.optimizer.zero_grad()
                    input.requires_grad_(True) # allow gradients wrt to input for pde loss
                    pred, flops = self.pinn(input)
                    #print("pred pinn", pred.shape)
                    loss = self.problem.compute_loss(pred, input)
                    loss.backward(retain_graph=True)

                    flops_history.append(flops)

                    history.append(self.test())

                    print(f"Epoch {i} // Total Loss : {loss.item()}")
                    return loss
                
                self.optimizer.step(closure=closure)
                
            input = next(iter(trainset))[0]
            pred, flops = self.pinn(input)

        flops_history = np.cumsum(flops_history)
        flops_history = flops_history.tolist()

        return pred, history, flops_history

    def test(self):
        '''
        Calculate testing error of predictions compared to true solution
        in terms of relative L2 error. 
        '''

        domain = self.problem.domain
        ntest = 1000
        points = torch.rand(ntest, 2)
        points = points * (domain[:,1] - domain[:,0]) + domain[:,0] 
        
        self.pinn.eval()
        pred, flops = self.pinn(points)
        true = self.problem.exact_solution(points)

        # check that no unwanted broadcasting occured
        assert (pred - true).numel() == ntest

        relative_L2 = torch.sqrt(torch.sum((pred - true) ** 2) / torch.sum(true ** 2))

        return relative_L2.item()