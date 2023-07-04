import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn
from problems import Cos1d, Cos1dMulticscale, Sin1dSecondOrder, Cos1dMulticscale_Extention

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime

# define parameters
domain = torch.tensor((-2*torch.pi, 2*torch.pi))
nsamples = 300
nwindows = 30
nepochs = 200
nepochs_pinn = 500
lr = 1e-3
hidden = 2
pinn_hidden = 4
neurons = 16
pinn_neurons = 64
overlap = 0.25
sigma = 0.02
# w = 15
w = (1, 2, 4, 8, 16)


problem = Cos1dMulticscale_Extention(domain, nsamples, w)
# problem = Sin1dSecondOrder(domain, nsamples, w)
# problem = Cos1d(domain, nsamples, w)

# get training set
trainset = problem.assemble_dataset()

# define fbpinn model
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma)

# define pinn model
pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)

# define optimizers
optimizer_pinn = optim.Adam(pinn.parameters(), 
                        lr=lr)

optimizer_fbpinn = optim.Adam(fbpinn.parameters(),
                            lr=lr)

# optimizer_pinn = optim.LBFGS(pinn.parameters(),
#                               lr=float(0.5),
#                               max_iter=50,
#                               max_eval=50000,
#                               history_size=150,
#                               line_search_fn="strong_wolfe",
#                               tolerance_change=0.5 * np.finfo(float).eps)
# optimizer_fbpinn = optim.LBFGS(fbpinn.parameters(),
#                               lr=float(0.5),
#                               max_iter=50,
#                               max_eval=50000,
#                               history_size=150,
#                               line_search_fn="strong_wolfe",
#                               tolerance_change=0.5 * np.finfo(float).eps)


# training loop FBPiNN
print("Training FBPINN")
history_fbpinn = list()
for i in range(nepochs):
    for input, in trainset:
        def closure():
            optimizer_fbpinn.zero_grad()
            input.requires_grad_(True) # allow gradients wrt to input for pde loss
            pred_fbpinn, fbpinn_output, window_output = fbpinn.forward(input)
            #loss = problem.debug_loss(pred_fbpinn, input)
            loss = problem.compute_loss(pred_fbpinn, input)
            loss.backward(retain_graph=True)
            #optimizer_fbpinn.step()
            history_fbpinn.append(loss.item())
            return loss
        optimizer_fbpinn.step(closure=closure)
    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {history_fbpinn[-1]}")


# training loop PiNN
print("Training PINN")
history_pinn = list()
for i in range(nepochs_pinn):
    for input, in trainset:
        def closure():
            optimizer_pinn.zero_grad()
            input.requires_grad_(True) # allow gradients wrt to input for pde loss
            pred = pinn.forward(input)
            loss_pinn = problem.compute_loss(pred, input)
            #loss = problem.debug_loss(pred, input)
            loss_pinn.backward(retain_graph=True)
            history_pinn.append(loss_pinn.item())
            return loss_pinn   
        optimizer_pinn.step(closure=closure)
    
    if i % 10 == 0:
        print(f"Epoch {i} // Total Loss : {history_pinn[-1]}")

# # training loop FBPiNN
# print("Training FBPINN")
# history_fbpinn = list()
# for i in range(nepochs):
#     for input, in trainset:
#         optimizer_fbpinn.zero_grad()
#         input.requires_grad_(True) # allow gradients wrt to input for pde loss
#         pred_fbpinn, fbpinn_output, window_output = fbpinn(input)
#         loss = problem.compute_loss(pred_fbpinn, input)
#         #loss = problem.debug_loss(pred_fbpinn, input)
#         loss.backward()
#         optimizer_fbpinn.step()
#         history_fbpinn.append(loss.item())
        
#         # checks whether model output has changed
#         assert(not torch.equal(pred_fbpinn, fbpinn(input)[0]))

#     if i % 10 == 0:
#         print(f"Epoch {i} // Total Loss : {loss.item()}")


# # training loop PiNN
# print("Training PINN")
# history_pinn = list()
# for i in range(nepochs_pinn):
#     for input, in trainset:
#         optimizer_pinn.zero_grad()
#         input.requires_grad_(True) # allow gradients wrt to input for pde loss
#         pred = pinn(input)
#         loss = problem.compute_loss(pred, input)
#         #loss = problem.debug_loss(pred, input)
#         loss.backward()
#         optimizer_pinn.step()
#         history_pinn.append(loss.item())

#         # checks whether model output has changed
#         assert(not torch.equal(pred, pinn(input)))
    
#     if i % 10 == 0:
#         print(f"Epoch {i} // Total Loss : {loss.item()}")

# do some plots (Figure 7) to visualize ben-moseley style 
#use gridspec for final layout

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.2)

fbpinn_subdom = fig.add_subplot(grid[0,:2])
fbpinn_vs_exact = fig.add_subplot(grid[0,2:])
window_fct = fig.add_subplot(grid[1,0:2])
training_error_l2 = fig.add_subplot(grid[-1,-1])
pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

#plot of FBPiNN with subdomain definition - every subdomain different color


#plt.plot()
pred_fbpinn, fbpinn_output, window_output = fbpinn.forward(input)
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
pred = pinn.forward(input)
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



#Test loss (L1 norm) vs FLOPS (floating point operations)



#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff

for i in range(nwindows):
  window_fct.hlines( -0.5 if i%2 else -0.4, fbpinn.partition_domain()[i][0], fbpinn.partition_domain()[i][1],  linewidth=5)
   
window_fct.hlines(-1, fbpinn.partition_domain()[0][0], fbpinn.partition_domain()[nwindows-1][1],  linewidth=2, color = 'tab:grey')
for j in range(nwindows-1):
    window_fct.hlines(-1,fbpinn.partition_domain()[j][1], fbpinn.partition_domain()[j+1][0],  linewidth=5, color = 'magenta')

for i in range(nwindows):
    window_fct.plot(input.detach().numpy(),window_output[i,].detach().numpy())


window_fct.set_yticks([-1,-0.45,0,0.5,1],['overlap','subdomain',0,'window function',1])
window_fct.set_xlabel('x')
window_fct.set_title('FBPiNN window function and domains')

current_working_directory = os.getcwd()
target_dir =current_working_directory + '/results/'

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
plot_name= dt_string +'_' + str(round(history_fbpinn[-1],2))

plt.savefig( target_dir + 'plot_' + plot_name + '.png' )

plt.show()