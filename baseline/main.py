import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn
from problems import Cos1d, Cos1dMulticscale

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# define parameters
domain = torch.tensor((-2*torch.pi, 2*torch.pi))
nsamples = 3000
nwindows = 30
nepochs = 5000
lr = 0.001
hidden = 2
pinn_hidden = 5
neurons = 16
pinn_neurons = 128
overlap = 0.25
sigma = 0.02
w = 15

problem = Cos1d(domain, nsamples, w)
#problem = Cos1dMulticscale(domain, nsamples, w = w)

# get training set
trainset = problem.assemble_dataset()

# define fbpinn model
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma, u_sd=problem.u_sd)

# define pinn model
pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons, u_sd=problem.u_sd)

# Isi: hier ggf nwindows auf 1 setzen und FBPinns benutzen?
# Caro: gute Idee! dann ist aber auch noch die window function applied - ich schau wie ben das hat

# define optimizers
optimizer_pinn = optim.Adam(pinn.parameters(), 
                        lr=lr,
                        weight_decay=1e-3)

optimizer_fbpinn = optim.Adam(fbpinn.parameters(),
                            lr=lr,
                            weight_decay=1e-3)


# training loop FBPiNN
print("Training FBPINN")
history_fbpinn = list()
for i in range(nepochs):
    for input, in trainset:
        optimizer_fbpinn.zero_grad()
        input.requires_grad_(True)
        pred_fbpinn, fbpinn_output = fbpinn.forward(input)
        #loss = problem.debug_loss(pred_fbpinn, input)
        loss = problem.compute_loss(pred_fbpinn, input)
        loss.backward(retain_graph=True)
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
        loss = problem.compute_loss(pred, input)
        #loss = problem.debug_loss(pred, input)
        loss.backward()
        optimizer_pinn.step()
        history_pinn.append(loss.item())
    
    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {loss.item()}")

# do some plots (Figure 7) to visualize ben-moseley style 
#use gridspec for final layout

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)

fbpinn_subdom= fig.add_subplot(grid[0,:2])
fbpinn_vs_exact = fig.add_subplot(grid[0,2:])
training_error_l2=fig.add_subplot(grid[-1,-1])
pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

#plot of FBPiNN with subdomain definition - every subdomain different color


#plt.plot()

for i in range(nwindows):
    fbpinn_subdom.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

fbpinn_subdom.set_ylabel('u')
fbpinn_subdom.set_xlabel('x')
fbpinn_subdom.set_title('FBPiNN: individual network solution')


#plot of FBPiNN's solution vs exact solution

fbpinn_vs_exact.plot(input.detach().numpy(),pred_fbpinn.detach().numpy())
fbpinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy())
fbpinn_vs_exact.set_ylabel('u')
fbpinn_vs_exact.set_xlabel('x')
fbpinn_vs_exact.set_title('FBPiNN: global solution vs exact')

#plot of different PiNN config vs exact solution

pinn_vs_exact.plot(input.detach().numpy(),pred.detach().numpy())
pinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy())
pinn_vs_exact.set_ylabel('u')
pinn_vs_exact.set_xlabel('x')
pinn_vs_exact.set_title('PiNN: global solution vs exact')

#Test loss (L1 norm) vs Trainings step


#plt.plot()
training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="Train Loss FBPiNN L2 norm ")
training_error_l2.plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="Train Loss PiNN L2 norm ")
training_error_l2.legend()
training_error_l2.set_title('Comparing training errors')

plt.show()

#Test loss (L1 norm) vs FLOPS (floating point operations)



#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff

for i in range(nwindows):
  plt.hlines( 0 if i%2 else 0.1, fbpinn.partition_domain()[i][0], fbpinn.partition_domain()[i][1],  linewidth=5)
plt.hlines(-0.25, fbpinn.partition_domain()[0][0], fbpinn.partition_domain()[nwindows-1][1],  linewidth=2, color = 'tab:grey')
for j in range(nwindows-1):
    plt.hlines(-0.25,fbpinn.partition_domain()[j][1], fbpinn.partition_domain()[j+1][0],  linewidth=5, color = 'magenta')
plt.yticks([-1,0,1])

plt.show()



