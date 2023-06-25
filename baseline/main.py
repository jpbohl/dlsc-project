import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import ADAM

from fbpinn import FBPinn


def assemble_dataset(domain, nsamples):
    """
    Sample points in given domain and create dataloader
    for training.
    """

    raise NotImplementedError

def compute_loss(pred, input):
    """
    Compute PDE loss using autograd
    """

    raise NotImplementedError

# define parameters
domain = torch.tensor((0, 1))
nwindows = 10
nsamples = 1000
nepochs = 1000
lr = 0.0001
hidden = 2
neurons = 12

#TODO: define problem together with exact solution to
#du/dx = ω1 cos(ω1x) + ω2 cos(ω2x)
#ω1=1, ω2=15

# get training set
trainset = assemble_dataset(domain, nsamples)

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
        #TODO: add hard constraint for boundary conditions
        loss = compute_loss(pred, input)
        loss.backward()


# do some plots to visualize ben-moseley style 