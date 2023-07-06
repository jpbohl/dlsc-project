import torch
from torch.utils.data import DataLoader, TensorDataset


class Cos1d(object):

#define problem together with exact solution to
#du/dx = cos(ω*x)
#u(0)=0
#solution u(x)=1/ω sin(ω*x)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples
        self.w = w

        # Mean and variance to normalize hard constraint
        self.mean = (self.domain[1] + self.domain[0]) / 2
        self.std = (self.domain[1] - self.domain[0]) / 2
        
        # Mean and variance for unnormalisation
        self.u_sd = 1 / w 
        self.u_mean = 0

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(1, seed=0)
        points = sobol.draw(self.nsamples)

        points = points * (self.domain[1] - self.domain[0]) + self.domain[0]

        #in 1d we sort the points in ascending order 
        points, indices = torch.sort(points,dim=0)

        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred, input):

        input_norm = (input - self.mean) / self.std

        return torch.tanh(self.w * input_norm) * pred 

    def compute_pde_residual(self, pred, input): 
        """
        Compute PDE loss using autograd
        """

        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        f = torch.cos(self.w * input) 
        
        assert (dx - f).numel() == self.nsamples

        return dx - f
    
    def compute_loss(self, pred, input, verbose=False):
        """
        Compute loss by applying the norm to the pde residual 
        """
        
        residual  = self.compute_pde_residual(pred, input)
        loss = torch.mean(abs(residual) ** 2)

        #get log loss 
        loss = torch.log10(loss)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss

    def debug_loss(self, pred, input):

        residual = pred -  self.exact_solution(input)

        assert residual.numel() == self.nsamples

        return torch.mean(residual ** 2)

    def exact_solution(self, input):

        return torch.sin(self.w * input) / self.w


class Cos1dMulticscale(object):

#define problem together with exact solution to
#du/dx = ω1 cos(ω1*x) + ω2 cos(ω2*x)
#u(x)=0
#solution u(x)=sin(ω1*x)+sin(ω2*x)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples

        # in this case w = (w1,w2)
        self.w1 = w[0]
        self.w2 = w[1]

        # mean and variance to normalize hard constraint
        self.mean = (self.domain[1] + self.domain[0]) / 2
        self.std = (self.domain[1] - self.domain[0]) / 2

        # mean and variance to unnormalize NNs
        self.u_sd = 2
        self.u_mean = 0

        self.training_dataset = self.assemble_dataset()

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(dimension=1, seed=0)
        points = sobol.draw(self.nsamples)

        #sample points in [a,b]
        points = points * (self.domain[1] - self.domain[0]) + self.domain[0] #maybe - ?
        
        #in 1d we sort the points in ascending order 
        points, indices = torch.sort(points, dim=-2)
        
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred, input):
        """
        define hard constraint which automatically enforces
        boundary conditions
        """

        input_norm = (input - self.mean) / self.std

        return torch.tanh(self.w2 * input_norm) * pred 

    def compute_pde_residual(self, pred, input):
        """
        Compute PDE loss using autograd
        """
        
        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        f = self.w1 * torch.cos(self.w1 * input) + self.w2 * torch.cos(self.w2 * input)
        
        assert (dx - f).numel() == self.nsamples

        return dx - f
    
    def compute_loss(self, pred, input, verbose=False):
        """
        Compute loss by applying the norm to the pde residual 
        """
        
        #unsupervised
        r_int  = self.compute_pde_residual(pred, input)
        loss_int = torch.mean(abs(r_int) ** 2)

        #get log loss 
        loss = torch.log10(loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss
    
    def debug_loss(self, pred, input):

        residual = pred -  self.exact_solution(input)

        assert residual.numel() == self.nsamples

        return torch.mean(residual ** 2)


    def exact_solution(self, input):

        return torch.sin(self.w1 * input) + torch.sin(self.w2 * input)

class Sin1dSecondOrder(object):

# define problem together with exact solution to
# d^2 u/dx^2 = sin(wx)
# u(0)=0
# du(0) / dx = - 1 / w
# solution u(x)= - 1 / w^2 * sin(wx)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples
        
        self.w = w

        # mean and variance to normalize for hard constraint
        self.mean = (domain[0] + domain[1]) / 2
        self.std = (domain[1] - domain[0]) / 2
        
        # mean and variance for unnormalization of NNs
        self.u_sd = 1 / (w ** 2)
        self.u_mean = 0

        self.training_dataset = self.assemble_dataset()

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(dimension=1, seed=0)
        points = sobol.draw(self.nsamples)

        #sample points in [a,b]
        points = points * (self.domain[1] - self.domain[0]) + self.domain[0] #maybe - ?
        
        #in 1d we sort the points in ascending order 
        points, indices = torch.sort(points, dim=-2)
        
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred, input):
        """
        define hard constraint which automatically enforces
        boundary conditions
        """

        input_norm = (input - self.mean) / self.std
        tanh = torch.tanh(self.w * input_norm)

        return (- 1 / (self.w ** 2)) * tanh + (tanh ** 2) * pred


    def get_gradients(self, pred, input):

        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        ddx = torch.autograd.grad(dx.sum(), input, create_graph=True)[0]

        return pred, dx, ddx

    def pde_loss(self, pred, input):

        # get gradients of neural network
        u, du, ddu = self.get_gradients(pred, input)

        tanh = torch.tanh(self.w * input)
        sech2 = 1 - tanh ** 2 # squared hyperbolic secant
        
        # constant term derivaitve of constraining operator
        ddc = 2 * tanh * sech2 

        # derivative of the linear term
        l = 2 * (self.w ** 2) * u * sech2 * (sech2 - 2 * (tanh ** 2))
        dl = 4 * self.w * du * tanh * sech2
        ddl = (tanh ** 2) * ddu

        # source term
        f = torch.sin(self.w * input)

        residual = ddc + l + dl + ddl - f

        assert residual.numel() == self.nsamples

        return residual

    def compute_pde_residual(self, pred, input):
        """
        Compute PDE loss using autograd
        """
        
        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        ddx = torch.autograd.grad(dx.sum(), input, create_graph=True)[0]
        f = torch.sin(self.w * input)
        
        assert (ddx - f).numel() == self.nsamples

        return ddx - f
    
    def compute_loss(self, pred, input, verbose=False):
        """
        Compute loss by applying the norm to the pde residual 
        """
        
        #unsupervised
        r_int  = self.compute_pde_residual(pred, input)
        loss_int = torch.mean(abs(r_int) ** 2)

        #get log loss 
        loss = torch.log10(loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss
    
    def debug_loss(self, pred, input):

        residual = pred -  self.exact_solution(input)

        assert residual.numel() == self.nsamples

        return torch.mean(residual ** 2)


    def exact_solution(self, input):

        return (- 1 / (self.w ** 2)) * torch.sin(self.w * input)
    
    
class Cos1dMulticscale_Extention(object):
#Extension (a) problem:
#define problem together with exact solution to
#du/dx = ω1 cos(ω1*x) + ω2 cos(ω2*x)+ ω3 cos(ω3*x) + ω4 cos(ω4*x) + ω5 cos(ω5*x)
#u(x)=0
#solution u(x)=sin(ω1*x)+sin(ω2*x)+sin(ω3*x)+sin(ω4*x)+sin(ω5*x)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples

        # in this case w = (w1,w2,w3,w4,w5)
        self.w1 = w[0]
        self.w2 = w[1]
        self.w3 = w[2]
        self.w4 = w[3]
        self.w5 = w[4]

        # mean and variance to normalize hard constraint
        self.mean = (self.domain[1] + self.domain[0]) / 2
        self.std = (self.domain[1] - self.domain[0]) / 2

        # mean and variance to unnormalize NNs
        self.u_sd = 5
        self.u_mean = 0

        self.training_dataset = self.assemble_dataset()

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(dimension=1, seed=0)
        points = sobol.draw(self.nsamples)

        #sample points in [a,b]
        points = points * (self.domain[1] - self.domain[0]) + self.domain[0] 
        
        #in 1d we sort the points in ascending order 
        points, indices = torch.sort(points, dim=-2)
        
        dataset = TensorDataset(points)
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)

        return dataloader

    def hard_constraint(self, pred, input):
        """
        define hard constraint which automatically enforces
        boundary conditions
        """

        input_norm = (input - self.mean) / self.std

        return torch.tanh(self.w5 * input_norm) * pred 

    def compute_pde_residual(self, pred, input):
        """
        Compute PDE loss using autograd
        """
        
        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        f = self.w1 * torch.cos(self.w1 * input) + self.w2 * torch.cos(self.w2 * input) + self.w3 * torch.cos(self.w3 * input) + self.w4 * torch.cos(self.w4 * input) + self.w5 * torch.cos(self.w5 * input)
        
        assert (dx - f).numel() == self.nsamples

        return dx - f
    
    def compute_loss(self, pred, input, verbose=False):
        """
        Compute loss by applying the norm to the pde residual 
        """
        
        #unsupervised
        r_int  = self.compute_pde_residual(pred, input)
        loss_int = torch.mean(abs(r_int) ** 2)

        #get log loss 
        loss = torch.log10(loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss
    
    def debug_loss(self, pred, input):

        residual = pred -  self.exact_solution(input)

        assert residual.numel() == self.nsamples

        return torch.mean(residual ** 2)


    def exact_solution(self, input):

        return torch.sin(self.w1 * input) + torch.sin(self.w2 * input) + torch.sin(self.w3 * input) + torch.sin(self.w4 * input) + torch.sin(self.w5 * input)