import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Cos1dMulticscale

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime

# define parameters

domain = torch.tensor((-2*torch.pi, 2*torch.pi))
nsamples = 3000
nwindows = 30
nepochs = 10
nepochs_pinn = 10
lr = 1e-3
hidden = 2
neurons = 16
overlap = 0.25
sigma = 0.02
w=(1,15)


problem = Cos1dMulticscale(domain, nsamples, w)

# get training set
trainset = problem.assemble_dataset()
input = next(iter(trainset))[0] # get input points for plotting

#### FBPiNN trainer


# define fbpinn model and trainer
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma)
fbpinn_trainer = FBPINNTrainer(fbpinn, lr, problem)

pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)


####PiNN trainer

# define pinn model and trainer

pinn216= Pinn(problem, domain, 2, 16)
pinn_trainer_216 = PINNTrainer(pinn216, lr, problem)
pred_216, history_pinn_216, history_pinn_flops_216 = pinn_trainer_216.train(nepochs_pinn, trainset)

pinn464= Pinn(problem, domain, 4, 64)
pinn_trainer_464 = PINNTrainer(pinn464, lr, problem)
pred_464, history_pinn_464, history_pinn_flops_464 = pinn_trainer_464.train(nepochs_pinn, trainset)

pinn5128= Pinn(problem, domain, 5, 128)
pinn_trainer_5128 = PINNTrainer(pinn5128, lr, problem)
pred_5128, history_pinn_5128, history_pinn_flops_5128 = pinn_trainer_5128.train(nepochs_pinn, trainset)


# Realtive L2 Test Loss
relativeL2 = pinn_trainer_5128.test()
print("Relative L2 Loss: ", relativeL2)

# do some plots (Figure 7) to visualize ben-moseley style 

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.2)

fbpinn_subdom = fig.add_subplot(grid[0,:2])
fbpinn_vs_exact = fig.add_subplot(grid[0,2:])

pinn216_plot= fig.add_subplot(grid[1,:2])
pinn464_plot= fig.add_subplot(grid[1,2:])
pinn5128_plot= fig.add_subplot(grid[-1,:2])

training_error_l2 = fig.add_subplot(grid[-1,-1])
training_error_flop= fig.add_subplot(grid[-1,-2])
#pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

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

pred216, flops216 = pinn216.forward(input)
pinn216_plot.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
pinn216_plot.plot(input.detach().numpy(),pred216.detach().numpy(), label="Prediction")
pinn216_plot.set_ylabel('u')
pinn216_plot.set_xlabel('x')
pinn216_plot.legend()
pinn216_plot.set_title('PiNN: 2 hidden layers 16 neurons')

pred464, flops464 = pinn464.forward(input)
pinn464_plot.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
pinn464_plot.plot(input.detach().numpy(),pred464.detach().numpy(), label="Prediction")
pinn464_plot.set_ylabel('u')
pinn464_plot.set_xlabel('x')
pinn464_plot.legend()
pinn464_plot.set_title('PiNN: 4 hidden layers 64 neurons')

pred5128, flops5128 = pinn5128.forward(input)
pinn5128_plot.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
pinn5128_plot.plot(input.detach().numpy(),pred5128.detach().numpy(), label="Prediction")
pinn5128_plot.set_ylabel('u')
pinn5128_plot.set_xlabel('x')
pinn5128_plot.legend()
pinn5128_plot.set_title('PiNN: 5 hidden layers 128 neurons')

# Test loss (L1 norm) vs Trainings step

training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="FBPiNN")
training_error_l2.plot(np.arange(1, len(history_pinn_216) + 1), history_pinn_216, label="PiNN (2-16)")
training_error_l2.plot(np.arange(1, len(history_pinn_464) + 1), history_pinn_464, label="PiNN (4-64)")
training_error_l2.plot(np.arange(1, len(history_pinn_5128) + 1), history_pinn_5128, label="PiNN (5-128)")
training_error_flop.set_xlabel('Training Step')
training_error_flop.set_ylabel('Relative L2 error')
training_error_l2.legend()
training_error_l2.set_title('Comparing test errors')

#Test loss (L1 norm) vs FLOPS (floating point operations)

training_error_flop.plot(history_fbpinn_flops, history_fbpinn, label="FBPiNN")
training_error_flop.plot(history_pinn_flops_216, history_pinn_216, label="PiNN (2-16)")
training_error_flop.plot(history_pinn_flops_464, history_pinn_464, label="PiNN (4-64)")
training_error_flop.plot(history_pinn_flops_5128, history_pinn_5128, label="PiNN (5-128)")
training_error_flop.set_xlabel('FLOPS')
training_error_flop.set_ylabel('Relative L2 error')
training_error_flop.legend()
training_error_flop.set_title('Comparing Test errors vs FLOPs')



#save plots in folder results

current_working_directory = os.getcwd()
target_dir =current_working_directory + '/results/'

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
plot_name= dt_string +'_' + str(round(history_fbpinn[-1],2))

plt.savefig( target_dir + 'plot_' + plot_name + '.png' )

plt.show()


target_dir =current_working_directory + '/models/'
# save models in folder models
torch.save(fbpinn.state_dict(), target_dir + dt_string + "_fbpinn.pt")
torch.save(pinn216.state_dict(), target_dir + dt_string + "_pinn.pt")


#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(1, 2, hspace=0.4, wspace=0.2)

window_fct = fig.add_subplot(grid[0,0:2])

if len(fbpinn.manual_part)==0:
    partition = fbpinn.partition_domain()
else:
    partition = fbpinn.manual_partition()     

for i in range(nwindows):
  window_fct.hlines( -0.5 if i%2 else -0.4, partition[i][0], partition[i][1],  linewidth=5)
   
window_fct.hlines(-1, partition[0][0], partition[nwindows-1][1],  linewidth=2, color = 'tab:grey')
for j in range(nwindows-1):
    window_fct.hlines(-1,partition[j][1], partition[j+1][0],  linewidth=5, color = 'magenta')

for i in range(nwindows):
    window_fct.plot(input.detach().numpy(),window_output[i,].detach().numpy())


window_fct.set_yticks([-1,-0.45,0,0.5,1],['overlap','subdomain',0,'window function',1])
window_fct.set_xlabel('x')
window_fct.set_title('FBPiNN window function and domains')

plt.savefig( target_dir + 'plot_window_function' + plot_name + '.png' )

plt.show()