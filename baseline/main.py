import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from fbpinn import FBPinn, Pinn
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
overlap=0.25
sigma=1

problem = Cos1d(domain, nsamples, w = 15)

### Task Caro

# get training set
trainset = problem.assemble_dataset()

# define fbpinn model
model_fbpinn = FBPinn(nwindows, domain, hidden, neurons, overlap, sigma)



# define pinn model
model_pinn = Pinn(domain, hidden, neurons) 

# Isi: hier ggf nwindows auf 1 setzen und FBPinns benutzen?
# Caro: gute Idee! dann ist aber auch noch die window function applied - ich schau wie ben das hat


# define optimizer

optimizer_pinn = optim.Adam(model_pinn.model.parameters(), 
                            lr=float(0.001))    

optimizer_fbpinn = optim.Adam(model_fbpinn.params,
                            lr=float(0.001))

# training loop FBPiNN
for i in range(nepochs):
    for input in trainset:
        optimizer_fbpinn.zero_grad()
        pred = model_fbpinn.forward(input[0])
        loss = problem.compute_loss(pred, input[0])
        loss.backward()


# training loop PiNN
for i in range(nepochs):
    for input in trainset:
        optimizer_pinn.zero_grad()
        pred = model_pinn.forward(input)
        loss = problem.compute_loss(pred, input)
        loss.backward()

# do some plots (Figure 7) to visualize ben-moseley style 

#plot of FBPiNN with subdomain definition - every subdomain different color

#plot of FBPiNN's solution vs exact solution

#plot of different PiNN config vs exact solution

#Test loss (L1 norm) vs Trainings step

#Test loss (L1 norm) vs FLOPS (floating point operations)


#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff


