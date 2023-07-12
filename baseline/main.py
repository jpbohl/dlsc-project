import torch

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Cos1d, Cos1dMulticscale, Sin1dSecondOrder, Cos1dMulticscale_Extention, Cos2d	

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime

import wandb

wandb.login()

# define parameters
#domain = torch.tensor(((-2*torch.pi, 2*torch.pi), (-2*torch.pi, 2*torch.pi)))
#domain = torch.tensor(((-2,2), (-2, 2)))
domain = torch.tensor((-2*torch.pi, 2*torch.pi))
nsamples = 3000
nwindows = 30
#nwindows = (6,2)
nepochs = 2000
nepochs_pinn = 2000
optimizer = "ADAM"
lr = 1e-3
hidden = 2
pinn_hidden = 4
neurons = 16
pinn_neurons = 64
overlap = 0.25
sigma = 0.05
w = 15
#w = [2 ** i for i in range(5)]
debug_loss = False

run = wandb.init(project="Sin1D 2ndOrder",
                 config={
                    "nsamples" : nsamples,
                    "nwindows" : nwindows,
                    "nepochs" : nepochs,
                    "nepochs_pinn" : nepochs_pinn,
                    "optimizer" : optimizer,
                    "lr" : lr,
                    "hidden" : hidden,
                    "pinn_hidden" : pinn_hidden,
                    "neurons" : neurons,
                    "pinn_neurons" : pinn_neurons,
                    "overlap" : overlap,
                    "sigma" : sigma,
                    "debug_loss" : debug_loss,
                    "w" : w})

#problem = Cos1dMulticscale_Extention(domain, nsamples, w)
#problem = Cos1dMulticscale(domain, nsamples, w)
problem = Sin1dSecondOrder(domain, nsamples, w)
#problem = Cos1d(domain, nsamples, w)
#problem = Cos2d(domain, nsamples, w)

# get training set
trainset = problem.assemble_dataset()
input = next(iter(trainset))[0] # get input points for plotting
print('input', input )

# define fbpinn model and trainer
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma)
fbpinn_trainer = FBPINNTrainer(run, fbpinn, lr, problem, optimizer, debug_loss)

# define pinn model and trainer
pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
pinn_trainer = PINNTrainer(run, pinn, lr, problem, optimizer, debug_loss)

#pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train_outward(nepochs, trainset)
pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)

pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs_pinn, trainset)

# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)


# do some plots (Figure 7) to visualize ben-moseley style 

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.2)

fbpinn_subdom = fig.add_subplot(grid[0,:2])
fbpinn_vs_exact = fig.add_subplot(grid[0,2:])
window_fct = fig.add_subplot(grid[1,0:2])
training_error_l2 = fig.add_subplot(grid[-1,-1])
training_error_flop= fig.add_subplot(grid[-1,-2])
pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

#plot of FBPiNN with subdomain definition - every subdomain different color

pred_fbpinn, fbpinn_output, window_output, flops = fbpinn.plotting_data(input)
for i in range(nwindows):
    fbpinn_subdom.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

fbpinn_subdom.set_ylabel('u')
fbpinn_subdom.set_xlabel('x')
fbpinn_subdom.set_title('FBPiNN: individual network solution')


#plot of FBPiNN's solution vs exact solution

fbpinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
fbpinn_vs_exact.plot(input.detach().numpy(),pred_fbpinn.detach().numpy(), label="Prediction")
fbpinn_vs_exact.set_ylabel('u')
fbpinn_vs_exact.set_xlabel('x')
fbpinn_vs_exact.legend()
fbpinn_vs_exact.set_title('FBPiNN: global solution vs exact')

# plot of different PiNN config vs exact solution

pred, flops = pinn.forward(input)
pinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
pinn_vs_exact.plot(input.detach().numpy(),pred.detach().numpy(), label="Prediction")
pinn_vs_exact.set_ylabel('u')
pinn_vs_exact.set_xlabel('x')
pinn_vs_exact.legend()
pinn_vs_exact.set_title('PiNN: global solution vs exact')

# Test loss (L1 norm) vs Trainings step

training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="Train Loss FBPiNN L2 norm ")
training_error_l2.plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="Train Loss PiNN L2 norm ")
training_error_flop.set_xlabel('Training Step')
training_error_flop.set_ylabel('log L2 error')
training_error_l2.legend()
training_error_l2.set_title('Comparing training errors')

#Test loss (L1 norm) vs FLOPS (floating point operations)

training_error_flop.plot(history_fbpinn_flops, history_fbpinn, label="Train Loss FBPiNN L2 norm ")
training_error_flop.plot(history_pinn_flops, history_pinn, label="Train Loss PiNN L2 norm ")
training_error_flop.set_xlabel('FLOPS')
training_error_flop.set_ylabel('log L2 error')
training_error_flop.legend()
training_error_flop.set_title('Comparing training errors vs FLOPs')

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


#save plots in folder results

current_working_directory = os.getcwd()
target_dir =current_working_directory + '/results/'

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
plot_name = dt_string +'_' + str(round(history_fbpinn[-1],2))
plot_name = run.name + "_"

plt.savefig( target_dir + 'plot_' + plot_name + '.png' )

plt.show()


target_dir =current_working_directory + '/models/'
# save models in folder models
torch.save(fbpinn.state_dict(), target_dir + dt_string + "_fbpinn.pt")
torch.save(pinn.state_dict(), target_dir + dt_string + "_pinn.pt")