import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.optimizer import ADAM

from fbpinn import FBPinn

class Cos1d(object):

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

        points = points * (self.domain[1] - self.domain[0]) - self.domain[0]
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred):

        return torch.tanh(self.w * pred) * pred

    def compute_loss(self, pred, input):
        """
        Compute PDE loss using autograd
        """
        pred = self.hard_constraint(pred)
        dx = torch.autograd.grad(pred.sum(), input, retain_graph=True)
        f = torch.cos(self.w * input)
        
        assert (dx - f).size() == self.nsamples

        return dx - f

    def exact_solution(self, input):

        return torch.cos(self.w * input)


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

#TODO: define problem together with exact solution to
#du/dx = ω1 cos(ω1x) + ω2 cos(ω2x)
#ω1=1, ω2=15

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