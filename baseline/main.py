import torch
import torch.optim as optim

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Cos1d, Cos1dMulticscale, Sin1dSecondOrder, Cos1dMulticscale_Extention, Cos2d, Sin_osc

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import os 
from datetime import datetime

# define parameters
#domain = torch.tensor((6/(20*torch.pi), 6/torch.pi)) # for xsin(1/x)
#domain = torch.tensor((-2*torch.pi, 2*torch.pi))
domain = torch.tensor(((-torch.pi, torch.pi), (-torch.pi, torch.pi))) # for 2D
#domain = torch.tensor(((0,4), (0,4)))
nsamples = 22500
#nwindows = 30
nepochs = 500
nepochs_pinn = 500
lr = 1e-3
hidden = 2
pinn_hidden = 5
neurons = 16
pinn_neurons = 128
overlap = 0.5
sigma = 0.0001
w=5
nwindows = (w,w)

#w = (1, 2, 4, 8, 16)

#problem = Cos1dMulticscale_Extention(domain, nsamples, w)
#problem = Sin1dSecondOrder(domain, nsamples, w)
#problem = Cos1d(domain, nsamples, w)
problem = Cos2d(domain, nsamples, w)

if isinstance(nwindows, int):
    if domain.ndim != 1:
        raise ValueError('nwindows must be a tuple if domain.ndim > 1')
else:
    if domain.ndim != 2:
        raise ValueError('nwindows must be an integer if domain.ndim == 1')


#problem = Cos1d(domain, nsamples, w)
#problem = Sin1dSecondOrder(domain, nsamples, w)



# get training set
if isinstance(nwindows, int):
    trainset = problem.assemble_dataset()
    input = next(iter(trainset))[0] # get input points for plotting
    #print('input', input )
else:
    trainset, plotset = problem.assemble_dataset()
    input = next(iter(plotset))[0] # get input points for plotting
    #print('input', input )

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
if isinstance(nwindows, int):
    pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
    pinn_trainer = PINNTrainer(pinn, lr, problem)
    pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs_pinn, trainset)

else:
    pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
    pinn_trainer = PINNTrainer(pinn, lr, problem)
    pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs_pinn, trainset)



# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("Relative L2 Loss: ", relativeL2)


# do some plots (Figure 7) to visualize ben-moseley style 
if isinstance(nwindows, int):
    fig = plt.figure(figsize=(15,8))
    grid = plt.GridSpec(3, 4, hspace=0.4, wspace=0.2)

    fbpinn_subdom = fig.add_subplot(grid[0,:2])
    fbpinn_vs_exact = fig.add_subplot(grid[0,2:])
    window_fct = fig.add_subplot(grid[1,0:2])

    pinn_vs_exact = fig.add_subplot(grid[-1,0:2])

    training_error_l2 = fig.add_subplot(grid[-1,-1])
    training_error_flop= fig.add_subplot(grid[-1,-2])


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

    training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), history_fbpinn, label="FBPiNN")
    training_error_l2.plot(np.arange(1, len(history_pinn) + 1), history_pinn, label="PiNN ")
    training_error_flop.set_xlabel('Training Step')
    training_error_flop.set_ylabel('Relative L2 error')
    training_error_l2.legend()
    training_error_l2.set_title('Comparing test errors')

    #Test loss (L1 norm) vs FLOPS (floating point operations)

    training_error_flop.plot(history_fbpinn_flops, history_fbpinn, label="FBPiNN")
    training_error_flop.plot(history_pinn_flops, history_pinn, label="PiNN")
    training_error_flop.set_xlabel('FLOPS')
    training_error_flop.set_ylabel('Relative L2 error')
    training_error_flop.legend()
    training_error_flop.set_title('Comparing Test errors vs FLOPs')

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
else: 
    
    #pred_fbpinn, fbpinn_output, window_output, flops = fbpinn.plotting_data(input)
    
    # fbpinn_plot.plot(input.detach().numpy(), pred_fbpinn.detach().numpy(),  label="FBPinn Solution")
    # fbpinn_vs_exact.set_ylabel('x2')
    # fbpinn_vs_exact.set_xlabel('x1')
    # fbpinn_vs_exact.legend()
    # fbpinn_vs_exact.set_title('FBPiNN: global solution')
    # fig = plt.figure(figsize=(15,8))
    # grid = plt.GridSpec(2, 3, hspace=0.4, wspace=0.2)
    
    # pinn_plot = fig.add_subplot(grid[0,0])
    # pinn_vs_exact = fig.add_subplot(grid[1,0])
    # fbpinn_plot = fig.add_subplot(grid[0,1])
    # fbpinn_vs_exact = fig.add_subplot(grid[1,1])
    # exact_plot = fig.add_subplot(grid[0,2])
    # training_error_l2 = fig.add_subplot(grid[1,2])
    
    # #plot of FBPiNN's solution vs exact solution
    # #pred_fbpinn, flops = fbpinn(input, active_models=None)
    pred_fbpinn, fbpinn_output, window_output, flops_fbpinn = fbpinn.plotting_data(input)
    pred_fbpinn = pred_fbpinn.reshape(-1, )
    
    pred_pinn, flops_pinn = pinn.forward(input)
    pred_pinn = pred_pinn.reshape(-1, )
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=50)
    
    im1 = axs[0,0].scatter(input[:,0].detach(), input[:,1].detach(), c=pred_fbpinn.detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
    axs[0,0].set_xlabel('x1')
    axs[0,0].set_ylabel('x2')
    plt.colorbar(im1, ax=axs[0,0])
    axs[0,0].grid(True, which='both', ls=":")
    axs[0,0].set_title('FBPiNN: global solution')
    
    im2 = axs[0,1].scatter(input[:,0].detach(), input[:,1].detach(), c=pred_pinn.detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
    axs[0,1].set_xlabel('x1')
    axs[0,1].set_ylabel('x2')
    plt.colorbar(im2, ax=axs[0,1])
    axs[0,1].grid(True, which='both', ls=":")
    axs[0,1].set_title('PiNN: global solution')
    
    im3 = axs[0,2].scatter(input[:,0].detach(), input[:,1].detach(), c=problem.exact_solution(input).detach(), cmap='viridis', vmin=-3/w, vmax= 3/w)
    axs[0,2].set_xlabel('x1')
    axs[0,2].set_ylabel('x2')
    plt.colorbar(im3, ax=axs[0,2])
    axs[0,2].grid(True, which='both', ls=":")
    axs[0,2].set_title('Exact solution')
    
    # plot the difference between FBPiNN's solution and exact solution
    
    #fig, axs = plt.subplots(1, 3, figsize=(12, 6), dpi=50)
    
    im4 = axs[1,0].scatter(input[:,0].detach(), input[:,1].detach(), c=(pred_fbpinn-problem.exact_solution(input)).detach(), cmap='viridis')
    axs[1,0].set_xlabel('x1')
    axs[1,0].set_ylabel('x2')
    plt.colorbar(im4, ax=axs[1,0])
    axs[1,0].grid(True, which='both', ls=":")
    axs[1,0].set_title('FBPiNN: difference')
    
    im5 = axs[1,1].scatter(input[:,0].detach(), input[:,1].detach(), c=(pred_pinn-problem.exact_solution(input)).detach(), cmap='viridis')
    axs[1,1].set_xlabel('x1')
    axs[1,1].set_ylabel('x2')
    plt.colorbar(im5, ax=axs[1,1])
    axs[1,1].grid(True, which='both', ls=":")
    axs[1,1].set_title('PiNN: difference')
    
    training_error_l2 = axs[1,2]
    training_error_l2.plot(np.arange(1, len(history_fbpinn) + 1), torch.log10(torch.tensor(history_fbpinn)), label="FBPiNN")
    training_error_l2.plot(np.arange(1, len(history_pinn) + 1), torch.log10(torch.tensor(history_pinn)), label="PiNN ")
    training_error_l2.set_xlabel('Training Step')
    training_error_l2.set_ylabel('Relative L2 error')
    training_error_l2.legend()
    training_error_l2.set_title('Comparing test errors')
    
    # window_plot = axs[1,2]
    # # for i in range(nwindows[0]*nwindows[1]):
    # #     wind_out = sum(window_output[i, ])
    # #print("r",torch.sum(window_output, dim =0))
    # axs[1,2].grid(True, which='both', ls=":")
    # #window_plot.scatter(input[:,0].detach(),input[:,1].detach(), c=(torch.sum(window_output, dim =0)).detach(), cmap = 'viridis')
    # window_plot.scatter(input[:,0].detach(),input[:,1].detach(), c=(window_output[13, ]).detach(), cmap = 'viridis')
    # plt.tight_layout()
    # #print(window_output[0, ], sum(window_output[0, ]), window_output[0,].shape)

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