import torch
from torch.nn import Module, ModuleList
from torch.optim import Adam, LBFGS
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

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
        if isinstance(self.nwindows, int):
            self.std = (self.subdomains[:, 1] - self.subdomains[:, 0]) / 2
        else:
             
             self.subdomains_row, self.subdomains_col = torch.split(self.subdomains, self.nwindows)
             self.stdrow = (self.subdomains_row[:, 1] - self.subdomains_row[:, 0]) / 2
             self.stdcol = (self.subdomains_col[:, 1] - self.subdomains_col[:, 0]) / 2
             self.std = torch.cartesian_prod(self.stdrow, self.stdcol)
             

        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd
        # case 1D
        if isinstance(self.nwindows, int):
            self.models = ModuleList([NN(hidden, neurons) for _ in range(self.nwindows)])
        # case 2D    
        else:
            self.models = ModuleList([NN(hidden, neurons) for _ in range(self.nwindows[0]*self.nwindows[1])])


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
        #check if nwindows is not int or tuple otherwise raise error
        if not isinstance(self.nwindows, int) and not isinstance(self.nwindows, tuple):
            raise ValueError("nwindows must be an integer or a tuple of integers.")
            
        #problem for 1d: width = (self.domain[0][1]-self.domain[0][0]) / self.nwindows
        if isinstance(self.nwindows, int):
            subdomains = torch.zeros(self.nwindows, 2)
            width = (self.domain[1]-self.domain[0]) / self.nwindows
            for i in range(self.nwindows):
                #subdomains[i][0] = self.domain[0][0] + (i-self.overlap/2) * width if i != 0 else self.domain[0][0]
                #subdomains[i][1] = self.domain[0][0] + (i+1+self.overlap/2) * width if i != (self.nwindows-1) else self.domain[0][1]
                subdomains[i][0] = self.domain[0] + (i-self.overlap/2) * width if i != 0 else self.domain[0]
                subdomains[i][1] = self.domain[0] + (i+1+self.overlap/2) * width if i != (self.nwindows-1) else self.domain[1]
        else:
            subdomains = torch.zeros(sum(self.nwindows), 2)
            width = ((self.domain[0][1]-self.domain[0][0]) / self.nwindows[0] , (self.domain[1][1]-self.domain[1][0] )/ self.nwindows[1]) 
            for i in range(self.nwindows[0]):
                subdomains[i][0] = self.domain[0][0] + (i-self.overlap/2) * width[0] if i != 0 else self.domain[0][0]
                subdomains[i][1] = self.domain[0][0] + (i+1+self.overlap/2) * width[0] if i != (self.nwindows[0]-1) else self.domain[0][1]
            for j in range(self.nwindows[1]):
                subdomains[j+self.nwindows[0]][0] = self.domain[1][0] + (j-self.overlap/2) * width[1] if j != 0 else self.domain[1][0]
                subdomains[j+self.nwindows[0]][1] = self.domain[1][0] + (j+1+self.overlap/2) * width[1] if j != (self.nwindows[1]-1) else self.domain[1][1]
        #print(subdomains, subdomains.shape)
        return subdomains

        #raise NotImplementedError


    def get_midpoints(self):
        """
        Gets the midpoint of each subdomain for subdomain
        normalization. 
        """
        if isinstance(self.nwindows, int):
            midpoints = (self.subdomains[:, 1] + self.subdomains[:, 0]) / 2
        else:
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
        # 1D case
        if isinstance(self.nwindows, int):
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
        # 2D case
        else: 
            #initialize midpoints, edges of domain are not overlapping
            midpoints_row = torch.zeros(self.nwindows[0]+1)
            midpoints_col = torch.zeros(self.nwindows[1]+1)
            midpoints_row[0] = self.subdomains[0][0]
            midpoints_row[self.nwindows[0]] = self.subdomains[self.nwindows[0]-1][1]
            midpoints_col[0] = self.subdomains[0][0]
            midpoints_col[self.nwindows[1]] = self.subdomains[self.nwindows[1]-1][1]
            
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
        # 1D case
        if isinstance(self.nwindows, int):
            x_left = (torch.sub(input, self.get_midpoints_overlap()[iteration])) / self.sigma
            x_right = (torch.sub(input, self.get_midpoints_overlap()[iteration+1])) / self.sigma
            window = torch.sigmoid(x_left) * torch.sigmoid(-x_right)
        # 2D case
        else:
            x_left = (torch.sub(input[:,0], self.get_midpoints_overlap()[iteration][0])) / self.sigma
            x_right = (torch.sub(input[:,0], self.get_midpoints_overlap()[iteration+1][0])) / self.sigma
            y_left = (torch.sub(input[:,1], self.get_midpoints_overlap()[iteration][1])) / self.sigma
            y_right = (torch.sub(input[:,1], self.get_midpoints_overlap()[iteration+1][1])) / self.sigma
            window = torch.sigmoid(x_left) * torch.sigmoid(-x_right) * torch.sigmoid(y_left) * torch.sigmoid(-y_right)
                  
        #window = torch.clamp(torch.clamp(1/(1+torch.exp(x_left)), min = tol )* torch.clamp(1/(1+torch.exp(-x_right)), min = tol), min = tol)
        #window = 1/(1+torch.exp(x_left))* 1/(1+torch.exp(-x_right))
        
        return window


    def get_active_inputs(self, input, active_models):
        """
        Gets inputs relevant to training the currently
        active models.
        """

        active_inputs = torch.zeros_like(input, dtype=bool)
        if isinstance(self.nwindows, int):
            for i in active_models:
                subdomain = self.subdomains[i, :]

                # turn inputs in subdomain i to active
                active_inputs |= (input <= subdomain[1]) & (subdomain[0] <= input)

            return input[active_inputs].reshape(-1, 1)
        else:
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
        
        if active_models is None:
            active_models = range(self.nwindows)
        
        pred = torch.zeros_like(input)
            
        #pred = torch.zeros_like(input)
        flops = 0
        
        for i in active_models:

            model = self.models[i] # get model i
            
            # get index for points which are in model i subdomain and normalize
            in_subdomain = (self.subdomains[i][0] < input) & (input < self.subdomains[i][1])
            input_norm = ((input[in_subdomain] - self.means[i]) / self.std[i]).reshape(-1, 1)

            # check whether we normalised to (-1, 1)
            assert((input_norm <= 1).all().item())
            assert((-1 <= input_norm).all().item())

            # model i prediction and unnormalization
            output = model(input_norm).reshape(-1)
            output = output * self.u_sd + self.u_mean
            
            # compute window function for subdomain i and model out to predictions
            window = self.compute_window(input[in_subdomain], i)
            pred[in_subdomain] += window * output
            
            #add the number of flops for each trained network on subdomain
            flops += model.flops(input_norm.shape[0])
            
        #pred = self.problem.hard_constraint(pred, input)

        return pred, flops
    
    
    def plotting_data(self, input):
        """
        Computes forward pass of FBPinn model 
        """

        #output for every subdomain: dimension  nwindows*input
        fbpinn_output = torch.zeros(self.nwindows, input.size(0))
        window_output = torch.zeros(self.nwindows, input.size(0))
        pred = torch.zeros_like(input)
        flops = 0
        
        for i in range(self.nwindows):

            model = self.models[i] # get model i
            
            # normalize data to given subdomain and extract relevant points
            input_norm = ((input - self.means[i]) / self.std[i]).reshape(-1, 1)

            # model i prediction
            output = model(input_norm)

            output = output * self.u_sd + self.u_mean

            # compute window function for subdomain i
            window = self.compute_window(input, i)

            # prediction of individual network times window function
            ind_pred = window * output

            # add prediction to total output
            # sum neural networks in overlapping regions
            pred += ind_pred

            ind_pred = self.problem.hard_constraint(ind_pred, input)
            window_output[i,] = window.reshape(1,-1)[0]
            fbpinn_output[i,] = ind_pred.reshape(1,-1)[0]

            #add the number of flops for each trained network on subdomain
            flops += model.flops(input_norm.shape[0])
            #print("Number of FLOPS:", model.flops(input_norm.shape[0]))
        
        pred = self.problem.hard_constraint(pred, input)

        return pred, fbpinn_output, window_output, flops

class FBPINNTrainer:

    def __init__(self, run, fbpinn, lr, problem, optimizer="ADAM", debug_loss=False):

        self.run = run 
        self.fbpinn = fbpinn
        self.lr = lr
        
        match optimizer:
            case "ADAM":
                self.optimizer = Adam(fbpinn.parameters(), lr=lr)
            case "LBFGS":
                self.optimizer = LBFGS(fbpinn.parameters(),
                                    lr=float(0.5),
                                    max_iter=50000,
                                    max_eval=50000,
                                    history_size=150,
                                    line_search_fn="strong_wolfe",
                                    tolerance_change=1.0 * np.finfo(float).eps)

        self.problem = problem
        
        if debug_loss:
            self.loss = problem.debug_loss
        else:
            self.loss = problem.compute_loss
        
    def train(self, nepochs, trainset, active_models=None): 
        print("Training FBPINN")
        
        history = list()
        flops_history = list() 
        for i in range(nepochs):
            
            for input_, in trainset:

                def closure():
                    self.optimizer.zero_grad()
                   

                    input_.requires_grad_(True) # allow gradients wrt to input for pde loss
                    pred, flops = self.fbpinn(input_, active_models=active_models)
                    loss = self.loss(pred, input_)
                    loss.backward(retain_graph=True)

                    flops_history.append(flops)
                    
                    test_loss = self.test()
                    history.append(test_loss)

                    self.run.log({"train loss fbpinn" : loss.item(),
                                  "test loss fbpinn" : test_loss})


                    print(f"Epoch {i} // Total Loss : {loss.item()}")
                    return loss

                
            self.optimizer.step(closure=closure)

        input = next(iter(trainset))[0]
        pred, _ = self.fbpinn(input)

        flops_history = np.cumsum(flops_history)
        flops_history = flops_history.tolist()
        
        return pred, history, flops_history 

    def train_outward(self, nepochs, trainset):
        """
        Training outward from the inital condition for
        1D problems. The models wh
        """

        # get initial active models:
        #       l_active is the active model left of the initial condition
        #       r_active is the active model right of the initial condition
        l_active, r_active = round(self.fbpinn.nwindows / 2) - 1, round(self.fbpinn.nwindows / 2) 

        input = next(iter(trainset))[0]
        
        # get parameters of currently training models
        l_parameters = list(self.fbpinn.models[l_active].parameters())
        r_parameters = list(self.fbpinn.models[r_active].parameters())
        self.optimizer = Adam(l_parameters + r_parameters, lr=self.lr)
            
        active_models = (l_active, r_active)
        prev_flops = 0 
            
        # get input points in active domains and train
        active_inputs = self.fbpinn.get_active_inputs(input, active_models)
        dataset = TensorDataset(active_inputs)
        dataloader = DataLoader(dataset, batch_size=active_inputs.shape[0], shuffle=False)
        out = self.train(nepochs, trainset, active_models)

        # update histories
        history = out[-2]
        flops_history = out[-1]

        while l_active > 0:
        
            # move active models outward 
            l_active -= 1
            r_active += 1
            
            # get parameters of currently training models
            l_parameters = list(self.fbpinn.models[l_active].parameters())
            r_parameters = list(self.fbpinn.models[r_active].parameters())
            self.optimizer = Adam(l_parameters + r_parameters, lr=self.lr)
            
            # after most inward models are trained we also need to evaluate
            # their inward neighbours to get a matching solution on the 
            # overlaps
            active_models = (l_active, r_active)
            fixed_models = (l_active + 1, r_active -1)
            
            # get input points in active domains and train
            active_inputs = self.fbpinn.get_active_inputs(input, active_models)
            dataset = TensorDataset(active_inputs)
            dataloader = DataLoader(dataset, batch_size=active_inputs.shape[0], shuffle=False)
            out = self.train(nepochs, dataloader, active_models + fixed_models)

            # update histories
            history += out[-2]

            # number of flops from previous iterations
            prev_flops = flops_history[-1]

            # add flops from last iterations to flops from current active models
            flops_history_iteration = np.array(out[-1]) + prev_flops
            flops_history += flops_history_iteration.tolist()
            
        out = self.fbpinn(input)

        return out[0], history, flops_history
    
    def plot_windows(self, input):

        pred_fbpinn, fbpinn_output, window_output, flops = self.fbpinn.plotting_data(input)
        for i in range(self.fbpinn.nwindows):
            plt.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

        plt.show()
    
    def test(self):

        domain = self.problem.domain
        ntest = 1000   
        
        if domain.ndim == 1:
            points = torch.rand(ntest).reshape(-1, 1)
            points = points * (domain[1] - domain[0]) + domain[0]
        else:
        #2D case
            points = torch.rand(ntest, 2)
            points = points * (domain[:,1] - domain[:,0]) + domain[:,0]

        self.fbpinn.eval()
        pred, flops = self.fbpinn(points)
        pred = self.problem.hard_constraint(pred, points)
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
        if domain.ndim == 1:
            self.mean = (domain[1] + domain[0])/2
            self.std = (domain[1] - domain[0])/2
        else:
            self.mean = (domain[:,1] + domain[:,0])/2
            self.std = (domain[:,1] - domain[:,0])/2
        
        #parameters for unnormalize
        self.u_mean = problem.u_mean
        self.u_sd = problem.u_sd

        self.model= NN(hidden, neurons)


    def forward(self, input):
        
        # normalize data to given subdomain
        input_norm = (input - self.mean) / self.std 
        
        # model prediction and unnormalization
        pred = self.model(input_norm) 
        output = pred * self.u_sd + self.u_mean
        
        # apply hard constraint
        #output = self.problem.hard_constraint(pred, input)

        # compute flops of forward pass
        flops = self.model.flops(input_norm.shape[0])

        return output, flops

class PINNTrainer:

    def __init__(self, run, pinn, lr, problem, optimizer="ADAM", debug_loss=False):

        self.run = run 
        self.pinn = pinn

        match optimizer:
            case "ADAM":
                self.optimizer = Adam(pinn.parameters(),
                            lr=lr)
            case "LBFGS":
                self.optimizer = LBFGS(pinn.parameters(),
                              lr=float(0.5),
                              max_iter=50,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=0.5 * np.finfo(float).eps)
        
        self.problem = problem
        

        if debug_loss:
            self.loss = problem.debug_loss
        else:
            self.loss = problem.compute_loss
        

    def train(self, nepochs, trainset): 
        
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
                    loss = self.loss(pred, input)
                    loss.backward(retain_graph=True)

                    flops_history.append(flops)

                    test_loss = self.test()
                    history.append(test_loss)

                    self.run.log({"train loss pinn" : loss.item(),
                                  "test loss pinn" : test_loss})

                    print(f"Epoch {i} // Total Loss : {loss.item()}")
                    return loss
                
                self.optimizer.step(closure=closure)
                
            input = next(iter(trainset))[0]
            pred, flops = self.pinn(input)

        flops_history = np.cumsum(flops_history)
        flops_history = flops_history.tolist()

        return pred, history, flops_history


    def test(self):

        domain = self.problem.domain
        ntest = 1000
        if domain.ndim == 1:
            points = torch.rand(ntest).reshape(-1, 1)
            points = points * (domain[1] - domain[0]) + domain[0]
        else:
            #2D case
            points = torch.rand(ntest, 2)
            points = points * (domain[:,1] - domain[:,0]) + domain[:,0] 
        
        self.pinn.eval()
        pred, flops = self.pinn(points)
        pred = self.problem.hard_constraint(pred, points)
        true = self.problem.exact_solution(points)

        # check that no unwanted broadcasting occured
        assert (pred - true).numel() == ntest

        relative_L2 = torch.sqrt(torch.sum((pred - true) ** 2) / torch.sum(true ** 2))

        return relative_L2.item()