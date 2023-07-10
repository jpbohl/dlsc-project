import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Sin_osc

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime

domain = torch.tensor((6/(20*torch.pi), 6/torch.pi)) # for xsin(1/x)
nsamples = 3000
nwindows = 19
nepochs = 10
nepochs_pinn = 10
lr = 1e-4
hidden = 2
pinn_hidden = 5
neurons = 16
pinn_neurons = 128
overlap = 0.001
sigma = 0.002


#give manual domain for sin high and low freq partition

#for x*sin(1/x) idea to have 0s as intervals
# sin(1/x)==0 if 1/x= k*pi for k in natural numbers 
manual_part= [6/(i*torch.pi) for i in reversed(range(2,20))] #length 18 

problem = Sin_osc(domain, nsamples, 1)

# get training set
trainset = problem.assemble_dataset()
input = next(iter(trainset))[0] # get input points for plotting

#### FBPiNN trainer


# define fbpinn model and trainer
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma, manual_part)
fbpinn_trainer = FBPINNTrainer(fbpinn, lr, problem)

pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)

# define fbpinn model and trainer
fbpinn_fixedsub = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma)
fbpinn_trainer_fixedsub = FBPINNTrainer(fbpinn_fixedsub, lr, problem)

pred_fbpinn_fixedsub, history_fbpinn_fixedsub, history_fbpinn_flops_fixedsub = fbpinn_trainer_fixedsub.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer_fixedsub.test()
print("Relative L2 Loss: ", relativeL2)

####PiNN trainer

# define pinn model and trainer
pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
pinn_trainer = PINNTrainer(pinn, lr, problem)
pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs_pinn, trainset)

# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)



# do some plots (Figure 7) to visualize ben-moseley style 

fig = plt.figure(figsize=(15,8))
grid = plt.GridSpec(3, 12, hspace=0.4, wspace=0.4)

fbpinn_subdom = fig.add_subplot(grid[0,4:8])
window_fct = fig.add_subplot(grid[0,0:4])
fbpinn_vs_exact = fig.add_subplot(grid[0,8:])

fbpinn_fixedsub_subdom = fig.add_subplot(grid[1,4:8])
window_fixedsub_fct = fig.add_subplot(grid[1,0:4])
fbpinn_fixedsub_plot = fig.add_subplot(grid[1,8:])

pinn_vs_exact = fig.add_subplot(grid[-1,:6])

training_error_l2 = fig.add_subplot(grid[-1,6:9])
training_error_flop= fig.add_subplot(grid[-1,9:])


#plot of FBPiNN with subdomain definition - every subdomain different color

pred_fbpinn, fbpinn_output, window_output, flops = fbpinn.plotting_data(input)
for i in range(nwindows):
    fbpinn_subdom.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

fbpinn_subdom.set_ylabel('u')
fbpinn_subdom.set_xlabel('x')
fbpinn_subdom.set_title('FBPiNN: individual network solution')

#add-on: cool plot from fig 6 - with subdomain definition and overlap stuff

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

#plot of FBPiNN's solution vs exact solution

fbpinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
fbpinn_vs_exact.plot(input.detach().numpy(),pred_fbpinn.detach().numpy(), label="Prediction")
fbpinn_vs_exact.set_ylabel('u')
fbpinn_vs_exact.set_xlabel('x')
fbpinn_vs_exact.legend()
fbpinn_vs_exact.set_title('FBPiNN: global solution vs exact - adapted subdomains')

#plot of FBPiNN with subdomain definition - every subdomain different color

pred_fbpinn, fbpinn_output, window_output, flops = fbpinn_fixedsub.plotting_data(input)
for i in range(nwindows):
    fbpinn_fixedsub_subdom.plot(input.detach().numpy(),fbpinn_output[i,].detach().numpy())

fbpinn_fixedsub_subdom.set_ylabel('u')
fbpinn_fixedsub_subdom.set_xlabel('x')
fbpinn_fixedsub_subdom.set_title('FBPiNN: individual network solution - fixed subdomains')

#plot of window function

partition = fbpinn_fixedsub.partition_domain()


for i in range(nwindows):
  window_fixedsub_fct.hlines( -0.5 if i%2 else -0.4, partition[i][0], partition[i][1],  linewidth=5)
   
window_fixedsub_fct.hlines(-1, partition[0][0], partition[nwindows-1][1],  linewidth=2, color = 'tab:grey')
for j in range(nwindows-1):
    window_fixedsub_fct.hlines(-1,partition[j][1], partition[j+1][0],  linewidth=5, color = 'magenta')

for i in range(nwindows):
    window_fixedsub_fct.plot(input.detach().numpy(),window_output[i,].detach().numpy())


window_fixedsub_fct.set_yticks([-1,-0.45,0,0.5,1],['overlap','subdomain',0,'window function',1])
window_fixedsub_fct.set_xlabel('x')
window_fixedsub_fct.set_title('FBPiNN window function and domains')

#plot of FBPiNN's solution vs exact solution with equally spaced subdomains

fbpinn_fixedsub_plot.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
fbpinn_fixedsub_plot.plot(input.detach().numpy(),pred_fbpinn_fixedsub.detach().numpy(), label="Prediction")
fbpinn_fixedsub_plot.set_ylabel('u')
fbpinn_fixedsub_plot.set_xlabel('x')
fbpinn_fixedsub_plot.legend()
fbpinn_fixedsub_plot.set_title('FBPiNN: global solution vs exact - equal distance')

# plot of PiNN config vs exact solution

pred, flops = pinn.forward(input)
pinn_vs_exact.plot(input.detach().numpy(), problem.exact_solution(input).detach().numpy(), label="Exact Solution")
pinn_vs_exact.plot(input.detach().numpy(),pred.detach().numpy(), label="Prediction")
pinn_vs_exact.set_ylabel('u')
pinn_vs_exact.set_xlabel('x')
pinn_vs_exact.legend()
pinn_vs_exact.set_title('PiNN: global solution vs exact')


# Test loss (L1 norm) vs Trainings step

training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="FBPiNN adapted subdomains")
training_error_l2.plot(np.arange(1, len(history_fbpinn_fixedsub) + 1), history_fbpinn_fixedsub, label="FBPiNN equally distance")
training_error_l2.plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="PiNN ")
training_error_flop.set_xlabel('Training Step')
training_error_flop.set_ylabel('Relative L2 error')
training_error_l2.legend()
training_error_l2.set_title('Comparing test errors')

#Test loss (L1 norm) vs FLOPS (floating point operations)

training_error_flop.plot(history_fbpinn_flops, history_fbpinn, label="FBPiNN adapted subdomain")
training_error_flop.plot(history_fbpinn_flops_fixedsub, history_fbpinn_fixedsub, label="FBPiNN equally spaced")
training_error_flop.plot(history_pinn_flops, history_pinn, label="PiNN")
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
torch.save(fbpinn_fixedsub.state_dict(), target_dir + dt_string + "_fbpinn_fixedsub.pt")
torch.save(pinn.state_dict(), target_dir + dt_string + "_pinn.pt")