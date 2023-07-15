# Deep Learning in Scientific Computing: Part B Project

<center> Carolin Heinzler, Isabella Thiel, Jan Philipp Bohl </center>
<br/><br/>
Repository to reproduce experiments from the Finite Bases Physics Informed Neural Network Paper [[1]](#1).

The code is contained in the baseline folder and is built up as follows

### main.py

Defines the parameters, combines the different parts, executes training and plots the results.

There also different scripts which contain parameters to reproduce different tasks discussed in the report and plot the results.
These are:

- `baseline_fig7.py` to reproduce Figure 7 from [1](1)
- `extension_sin2ndorder.py` to reproduce the results of the second order problem in 1D with standard and outward training schedules
    for the FBPiNN models
- `extension_2D.py` to reproduce the 2D problem from Section 5.3 of [1](1)
- `extension_LBFGS.py` for training with LBFGS optimizer instead of ADAM
- `extension_manual_subdomain.py` for a problem showing a custom domain decomposition adapted to a specific problem.

### problems.py

Contains the problem classes which implement the creation of datasets, unnormalization parameters, hard constraint, physics loss
and exact solution if available.

### nn.py

Contains the base fully connected neural network used to build PiNNs and FBPiNNs.

### fbpinn.py

Contains classes implementing FBPiNN and PiNN as torch.nn.Module children. 
The FBPiNN splits the domain into subdomains, computes window functions and then combines the different models in the forward pass. 
The PiNN class implements a standard PiNN. There are also trainer classes for PiNN and FBPiNN respectively, which defines
the optimizers and runs training and testing. In the case of FBPiNNs, it also implements outward training as explained in [[1]](#1).

### References
<a id="1">[1]</a> 
Ben Moseley, Andrew Markham, Tarje Nissen-Meyer (2022). 
Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations
Arxiv: https://arxiv.org/abs/2107.07871.
