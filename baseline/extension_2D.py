import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Cos2d

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime


# define parameters


domain = torch.tensor(((-torch.pi, torch.pi), (-torch.pi, torch.pi)))
#domain = torch.tensor(((-2*torch.pi, 2*torch.pi), (-2*torch.pi, 2*torch.pi)))
w=7

nsamples = 1000*w**2 # 1000 samples per window/subdomain

nepochs = 1
nepochs_pinn = 1
lr = 1e-3
hidden = 2
pinn_hidden = 5
neurons = 16
pinn_neurons = 128
overlap = 0.75


sigma = (domain[0][1]-domain[0][0])/w*0.05
nwindows = (w,w)


problem = Cos2d(domain, nsamples, w)

if isinstance(nwindows, int):
    if domain.ndim != 1:
        raise ValueError('nwindows must be a tuple if domain.ndim > 1')
else:
    if domain.ndim != 2:
        raise ValueError('nwindows must be an integer if domain.ndim == 1')




# get training set
trainset, plotset = problem.assemble_dataset()
input = next(iter(plotset))[0] # get input points for plotting


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

pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
pinn_trainer = PINNTrainer(pinn, lr, problem)
pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs_pinn, trainset)



# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)


pred_fbpinn, fbpinn_output, window_output, flops_fbpinn = fbpinn.plotting_data(input)
pred_fbpinn = pred_fbpinn.reshape(-1, )

pred_pinn, flops_pinn = pinn.forward(input)
pred_pinn = pred_pinn.reshape(-1, )
fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=50)

# plot the global solution of FBPiNN 
im1 = axs[0,0].scatter(input[:,0].detach(), input[:,1].detach(), c=pred_fbpinn.detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
axs[0,0].set_xlabel('x1')
axs[0,0].set_ylabel('x2')
plt.colorbar(im1, ax=axs[0,0])
axs[0,0].grid(True, which='both', ls=":")
axs[0,0].set_title('FBPiNN: global solution')

# plot the global solution of PiNN
im2 = axs[0,1].scatter(input[:,0].detach(), input[:,1].detach(), c=pred_pinn.detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
axs[0,1].set_xlabel('x1')
axs[0,1].set_ylabel('x2')
plt.colorbar(im2, ax=axs[0,1])
axs[0,1].grid(True, which='both', ls=":")
axs[0,1].set_title('PiNN: global solution')

# plot the exact solution for comparison
im3 = axs[0,2].scatter(input[:,0].detach(), input[:,1].detach(), c=problem.exact_solution(input).detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
axs[0,2].set_xlabel('x1')
axs[0,2].set_ylabel('x2')
plt.colorbar(im3, ax=axs[0,2])
axs[0,2].grid(True, which='both', ls=":")
axs[0,2].set_title('Exact solution')

# plot the difference between FBPiNN's solution and exact solution

scale_max = max(abs((pred_fbpinn-problem.exact_solution(input)).detach().numpy().max()), abs((pred_pinn-problem.exact_solution(input)).detach().numpy().max()))

im4 = axs[1,0].scatter(input[:,0].detach(), input[:,1].detach(), c=(pred_fbpinn-problem.exact_solution(input)).detach(), cmap='viridis', vmin=-scale_max, vmax= scale_max)
axs[1,0].set_xlabel('x1')
axs[1,0].set_ylabel('x2')
plt.colorbar(im4, ax=axs[1,0])
axs[1,0].grid(True, which='both', ls=":")
axs[1,0].set_title('FBPiNN: difference')

# plot the difference between PiNN's solution and exact solution
im5 = axs[1,1].scatter(input[:,0].detach(), input[:,1].detach(), c=(pred_pinn-problem.exact_solution(input)).detach(), cmap='viridis', vmin=-scale_max, vmax= scale_max)
axs[1,1].set_xlabel('x1')
axs[1,1].set_ylabel('x2')
plt.colorbar(im5, ax=axs[1,1])
axs[1,1].grid(True, which='both', ls=":")
axs[1,1].set_title('PiNN: difference')

# plot the test errors against training steps
training_error_l2 = axs[1,2]
training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), (torch.tensor(history_fbpinn)), label="FBPiNN")
training_error_l2.plot(np.arange(1, len(history_pinn) + 1), (torch.tensor(history_pinn)), label="PiNN ")
training_error_l2.set_xlabel('Training Step')
training_error_l2.set_ylabel('Relative L2 error')
training_error_l2.legend()
training_error_l2.set_title('Comparing test errors')

plt.tight_layout()

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
torch.save(pinn.state_dict(), target_dir + dt_string + "_pinn.pt")