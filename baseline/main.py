import torch
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
nepochs = 1000 #1000
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
fbpinn = FBPinn(nwindows, domain, hidden, neurons, overlap, sigma, u_sd=1/15)

# define pinn model
pinn = Pinn(domain, hidden, neurons, u_sd=1/15) 

# Isi: hier ggf nwindows auf 1 setzen und FBPinns benutzen?
# Caro: gute Idee! dann ist aber auch noch die window function applied - ich schau wie ben das hat

# define optimizers
optimizer_pinn = optim.Adam(pinn.parameters(), 
                            lr=float(0.001))    

optimizer_fbpinn = optim.Adam(fbpinn.parameters(),
                            lr=float(0.001))

# training loop FBPiNN
print("Training FBPINN")
history_fbpinn = list()
for i in range(nepochs):
    for input, in trainset:
        optimizer_fbpinn.zero_grad()
        input.requires_grad_(True) # allow gradients wrt to input for pde loss
        pred = fbpinn.forward(input)
        pred = problem.hard_constraint(pred, input) # apply hard constraint for boundary
        loss = problem.compute_loss(pred, input)
        loss.backward()
        optimizer_fbpinn.step()
        history_fbpinn.append(loss.item())

    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {loss.item()}")


# training loop PiNN
print("Training PINN")
history_pinn = list()
for i in range(nepochs):
    for input, in trainset:
        optimizer_pinn.zero_grad()
        input.requires_grad_(True) # allow gradients wrt to input for pde loss
        pred = pinn.forward(input)
        pred = problem.hard_constraint(pred, input) # apply hard constraint for boundary
        loss = problem.compute_loss(pred, input)
        loss.backward()
        optimizer_pinn.step()
        history_pinn.append(loss.item())
    
    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {loss.item()}")


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




