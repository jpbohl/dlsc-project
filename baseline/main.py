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

# get training set
trainset = assemble_dataset(domain, nsamples)

# define model
model = FBPinn(nwindows, domain, hidden, neurons)

# define optimizer
optimizer = ADAM(model.parameters(), lr)

# training loop
for i in range(nepochs):
    for input in trainset:
        optimizer.zero_grad()
        pred = model.forward()
        loss = compute_loss(pred, input)
        loss.backward()