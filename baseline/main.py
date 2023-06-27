import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.optimizer import ADAM

from fbpinn import FBPinn

class Cos1d(object):

#define problem together with exact solution to
#du/dx = cos(ω*x)
#u(x)=0
#solution u(x)=1/ω sin(ω*x)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples
        self.w = w

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(1, seed=0)
        points = sobol.draw(self.nsamples)

        points = points * (self.domain[1] - self.domain[0]) - self.domain[0] #caro: ich glaube es sollte + self.domain[0] sein?
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred):

        return torch.tanh(self.w * pred) * pred #caro: sollte es nicht tanh(w*input)*pred sein?

    def compute_loss(self, pred, input): #caro: then maybe rename to compute_pde_residual
        """
        Compute PDE loss using autograd
        """
        pred = self.hard_constraint(pred)
        dx = torch.autograd.grad(pred.sum(), input, retain_graph=True)
        f = torch.cos(self.w * input)
        
        assert (dx - f).size() == self.nsamples

        return dx - f

    #caro: add function for norm of residual which is to minimize

    def exact_solution(self, input):

        return torch.cos(self.w * input) #caro: exact solution is 1/ω sin(ω*x)
    



class Cos1d_multicscale(object):

#define problem together with exact solution to
#du/dx = ω1 cos(ω1*x) + ω2 cos(ω2*x)
#u(x)=0
#solution u(x)=sin(ω1*x)+sin(ω2*x)



    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples

        #in this case w=(w1,w2)
        self.w1 = w[0]
        self.w2 = w[1]

        self.training_dataset = self.assemble_dataset()

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(dimension=1, seed=0)
        points = sobol.draw(self.nsamples)

        #sample points in [a,b]
        points = points * (self.domain[1] - self.domain[0]) + self.domain[0] #maybe - ?
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred, input):
        """
        define hard constraint which automatically enforces
        boundary conditions
        """

        return torch.tanh(self.w1 * input)*torch.tanh(self.w2 * x) * pred

    def compute_pde_residual(self, pred, input):
        """
        Compute PDE loss using autograd
        """


        pred = self.hard_constraint(pred, input)
        dx = torch.autograd.grad(pred.sum(), input, retain_graph=True)
        f = self.w1 * torch.cos(self.w1 * input)+ self.w2 * torch.cos(self.w2 * input)
        
        assert (dx - f).size() == self.nsamples

        return dx - f
    
    def compute_loss(self, input, verbose=True):
        """
        Compute loss by applying the norm to the pde residual 
        """


        #unsupervised
        r_int  = self.compute_pde_residual(pred, input)
        loss_int = torch.mean(abs(r_int) ** 2)

        #get log loss 
        loss = torch.log10(loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss


    def exact_solution(self, input):

        return torch.sin(self.w1 * input)+torch.sin(self.w2 * input)


# define parameters
domain = torch.tensor((0, 1))
nwindows = 10
nsamples = 1000
nepochs = 1000
lr = 0.0001
hidden = 2
neurons = 12

problem = Cos1d(domain, nsamples, w = 15)


### Task Caro

# get training set
trainset = problem.assemble_dataset(domain, nsamples)

# define fbpinn model
model_fbpinn = FBPinn(nwindows, domain, hidden, neurons)


#define regular pinn model
#TODO: define instance of PiNN

# define optimizer
optimizer = ADAM(model_fbpinn.parameters(), lr)

# training loop
for i in range(nepochs):
    for input in trainset:
        optimizer.zero_grad()
        pred = model_fbpinn.forward()
        loss = problem.compute_loss(pred, input)
        loss.backward()


# do some plots (Figure 7) to visualize ben-moseley style 