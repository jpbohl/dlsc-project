import torch

from fbpinn import FBPinn, Pinn, FBPINNTrainer, PINNTrainer
from problems import Sin1dSecondOrder
from plot import plot

import time

# define parameters
domain = torch.tensor((-2*torch.pi, 2*torch.pi))
nsamples = 3000
nwindows = 30
nepochs = 5000
lr = 1e-3
hidden = 2
pinn_hidden = 5
neurons = 16
pinn_neurons = 128
overlap = 0.25
sigma = 0.02
w = 15

problem = Sin1dSecondOrder(domain, nsamples, w)

# get training set
trainset = problem.assemble_dataset()
input = next(iter(trainset))[0] # get input points for plotting

############################################################################################################
# FBPiNN training
############################################################################################################

# define fbpinn model and trainer
fbpinn = FBPinn(problem, nwindows, domain, hidden, neurons, overlap, sigma)
fbpinn_trainer = FBPINNTrainer(fbpinn, lr, problem)

pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer.test()
print("FBPiNN relative L2 Loss: ", relativeL2)

time.wait(5)

############################################################################################################
# PiNN training
############################################################################################################

# define pinn model and trainer
pinn = Pinn(problem, domain, pinn_hidden, pinn_neurons)
pinn_trainer = PINNTrainer(pinn, lr, problem)
pred, history_pinn, history_pinn_flops = pinn_trainer.train(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("PiNN Relative L2 Loss: ", relativeL2)

time.wait(5)

plot(fbpinn, pinn, problem, input, history_fbpinn, history_fbpinn_flops, history_pinn, history_pinn_flops)


############################################################################################################
# 
# Outward training: we now investigate outward training for FBPiNN
# 
############################################################################################################

# FBPiNN
print("Training FBPiNN with outward training scheme")
pred_fbpinn, history_fbpinn, history_fbpinn_flops = fbpinn_trainer.train_outward(nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = fbpinn_trainer.test()
print("FBPiNN Relative L2 Loss: ", relativeL2)
time.wait(5)

# PiNN (we increase training time for the PiNN to make the comparison fairer)
pred, history_pinn, history_pinn_flops = pinn_trainer.train(nwindows * nepochs, trainset)

# Realtive L2 Test Loss
relativeL2 = pinn_trainer.test()
print("PiNN Relative L2 Loss: ", relativeL2)
time.wait(5)

plot(fbpinn, pinn, problem, input, history_fbpinn, history_fbpinn_flops, history_pinn, history_pinn_flops)