import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.optimizer import ADAM

from fbpinn import FBPinn
from problems import Cos1d, Cos1dMulticscale


# define parameters
# Parameters in paper ω=1 : 
# domain = torch.tensor((-2*torch.pi, 2*torch.pi))
# nwindows = 5
# hidden = 2
# neurons = 16
# nepochs = 50000
# nsamples = 200
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

# define pinn model
model_pinn = Pinn(domain, hidden, neurons) # Isi: hier ggf nwindows auf 1 setzen und FBPinns benutzen?

# define optimizer
optimizer_fbpinn = ADAM(model_fbpinn.parameters(), lr)
optimizer_pinn = ADAM(model_pinn.parameters(), lr)

# training loop FBPiNN
for i in range(nepochs):
    for input in trainset:
        optimizer_fbpinn.zero_grad()
        pred = model_fbpinn.forward(input)
        loss = problem.compute_loss(pred, input)
        loss.backward()


# training loop PiNN
for i in range(nepochs):
    for input in trainset:
        optimizer_pinn.zero_grad()
        pred = model_pinn.forward(input)
        loss = problem.compute_loss(pred, input)
        loss.backward()

# do some plots (Figure 7) to visualize ben-moseley style 