import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from fbpinn import FBPinn, Pinn
from problems import Cos1d, Cos1dMulticscale

import matplotlib.pyplot as plt
import numpy as np

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
nepochs = 10 #1000
lr = 0.0001
hidden = 2
neurons = 16
overlap = 0.25
sigma = 1

problem = Cos1d(domain, nsamples, w = 15)

### Task Caro

# get training set
trainset = problem.assemble_dataset()

# define fbpinn model
fbpinn = FBPinn(nwindows, domain, hidden, neurons, overlap, sigma)

# define pinn model
pinn = Pinn(domain, hidden, neurons) 

# Isi: hier ggf nwindows auf 1 setzen und FBPinns benutzen?
# Caro: gute Idee! dann ist aber auch noch die window function applied - ich schau wie ben das hat


# define optimizer

optimizer_pinn = optim.Adam(pinn.model.parameters(), 
                            lr=float(0.1))    

optimizer_fbpinn = optim.Adam(fbpinn.models.parameters(),
                            lr=float(0.1))

# training loop FBPiNN
print("Training FBPINN")
history_fbpinn = list()
for i in range(nepochs):
    for input in trainset:
        optimizer_fbpinn.zero_grad()
        pred = fbpinn.forward(input[0])
        #loss = problem.debug_loss(pred, input[0])
        loss = problem.compute_loss(pred, input[0])
        #loss = problem.compute_loss(pred, input[0],verbose=True)
        loss.backward()
        optimizer_fbpinn.step()
        history_fbpinn.append(loss.item())

    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {loss.item()}")


# training loop PiNN
print("Training PINN")
history_pinn = list()
for i in range(nepochs):
    for input in trainset:
        optimizer_pinn.zero_grad()
        pred = pinn.forward(input[0])
        loss = problem.compute_loss(pred, input[0])
        loss.backward()
        optimizer_pinn.step()
        history_pinn.append(loss.item())
    
    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {loss.item()}")


# training loop PiNN
for i in range(nepochs):
    for input in trainset:
        optimizer_pinn.zero_grad()
        pred = pinn.forward(input[0])
        loss = problem.compute_loss(pred, input[0])
        loss.backward()
        optimizer_pinn.step()

# do some plots (Figure 7) to visualize ben-moseley style 

#plot of FBPiNN with subdomain definition - every subdomain different color

#plot of FBPiNN's solution vs exact solution

#plot of different PiNN config vs exact solution

#Test loss (L1 norm) vs Trainings step

fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

axs[0].grid(True, which="both", ls=":")
im1= axs[0].plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="Train Loss FBPiNN L2 norm ")
#plt.xscale("log")
axs[0].legend()

axs[1].grid(True, which="both", ls=":")
im2= axs[1].plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="Train Loss PiNN L2 norm ")
#plt.xscale("log")
axs[1].legend()

plt.show()

#Test loss (L1 norm) vs FLOPS (floating point operations)


#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff




