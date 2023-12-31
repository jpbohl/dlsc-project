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

        return torch.tanh(self.w * input) * pred 


    def compute_pde_residual(self, pred, input): 
        """
        Compute PDE loss using autograd
        """

        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        f = torch.cos(self.w * input) 
        
        assert (dx - f).numel() == input.numel()

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

        return torch.tanh(self.w2 * input) * pred 

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

class Sin1DSecondOrder(object):

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

        tanh = torch.tanh(self.w * input)

        return (- 1 / (self.w ** 2)) * tanh + (tanh ** 2) * pred

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

        return torch.tanh(self.w5 * input) * pred 

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
    
    
class Cos2d(object):
#Extension (a) problem:
#define problem together with exact solution to
#du/dx1 + du/dx2 = cos(ω*x) + cos(ω*x)
#u(0,x2)=1/ω sin(ω*x2)
#domain = [-2*pi,2*pi]x[-2*pi,2*pi]
#solution u(x1,x2)=1/ω sin(ω*x1)+1/ω sin(ω*x2)

    def __init__(self, domain, nsamples, w):

        self.domain = domain
        self.nsamples = nsamples
        self.nsamples_2d = 100000

        # in this case w = w
        self.w = w

        # mean and variance to normalize hard constraint
        self.mean = (self.domain[:,1] + self.domain[:, 0]) / 2
        self.std = (self.domain[:, 1] - self.domain[:, 0]) / 2

        # mean and variance to unnormalize NNs
        self.u_sd = 1/w
        self.u_mean = 0

        self.training_dataset = self.assemble_dataset()

    def assemble_dataset(self):
        """
        Sample points in given domain and create dataloader
        for training.
        """

        sobol = torch.quasirandom.SobolEngine(dimension=2, seed=0, scramble=True)
        # point = sobol.draw(self.nsamples).reshape(-1, )
        # points =torch.cartesian_prod(point, point)
        points = sobol.draw(self.nsamples)
        points_plot = sobol.draw(self.nsamples_2d)
    
        #sample points in [a,b]x[c,d]
        points = points * (self.domain[:,1] - self.domain[:,0]) + self.domain[:,0] 
        points_plot = points_plot * (self.domain[:,1] - self.domain[:,0]) + self.domain[:,0]
        #print("points", points)

        points = points[points[:,0].argsort()]
        
        dataset = TensorDataset(points)
        dataset_plot = TensorDataset(points_plot)
        
        dataloader = DataLoader(dataset, batch_size=self.nsamples, shuffle=False)
        dataloader_plot = DataLoader(dataset_plot, batch_size=self.nsamples_2d, shuffle=False)

        return dataloader, dataloader_plot

    def hard_constraint(self, pred, input):
        """
        define hard constraint which automatically enforces
        boundary conditions
        """

        input_norm = (input - self.mean) / self.std
        #print("input_norm problem", input_norm.shape)
        output = 1/self.w * torch.sin(self.w *input[:, 1]) + torch.tanh(self.w * input[:, 0]) * pred 
        assert output.numel() == input.shape[0]
        #print("output problem", output.shape)
        return output

    def compute_pde_residual(self, pred, input):
        """
        Compute PDE loss using autograd
        """
        
        dx1 = torch.autograd.grad(pred.sum(), input, create_graph=True)[0][:,0]
        dx2 = torch.autograd.grad(pred.sum(), input, create_graph=True)[0][:,1]
        
        f = torch.cos(self.w * input[:, 0]) +torch.cos(self.w * input[:, 1])
        
        assert (dx1+dx2 - f).numel() == self.nsamples

        return dx1+dx2 - f
    
    def compute_loss(self, pred, input, verbose=False):
        """
        Compute loss by applying the norm to the pde residual 
        """
        
        #unsupervised
        r_int  = self.compute_pde_residual(pred, input)
        loss_int = torch.mean(abs(r_int) ** 2)
        #loss_int = self.debug_loss(pred, input)

        #get log loss 
        loss = torch.log10(loss_int)

        if verbose: print("Total loss: ", round(loss.item(), 4))

        return loss
    
    def debug_loss(self, pred, input):

        residual = pred -  self.exact_solution(input)

        assert residual.numel() == self.nsamples

        return torch.mean(abs(residual) ** 2)


    def exact_solution(self, input):

        return 1/self.w *torch.sin(self.w * input[:, 0]) + 1/self.w* torch.sin(self.w * input[:, 1]) 
    

class Sin_osc(object):

#define problem together with exact solution to
#du/dx = -(Cos[6/x]/x) + 1/6 Sin[6/x]
#u(1/Pi) = 1/Pi
#solution u(x)= x/6 Sin[6/x] 

#smallest width for (1/20pi, 1/pi) is 0.005

#domain  is torch.tensor((6/(20*torch.pi), 6/torch.pi))
#partition manual_part= [6/(i*torch.pi) for i in reversed(range(2,20))] #length 18 

    def __init__(self, domain, nsamples, w=1):

        self.domain = domain
        self.nsamples = nsamples
        self.w = w

        # Mean and variance to normalize hard constraint
        self.mean = (self.domain[1] + self.domain[0]) / 2
        self.std = (self.domain[1] - self.domain[0]) / 2
        
        # Mean and variance for unnormalisation
        self.u_sd = 0.2
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

        return torch.tanh(self.w * (input-6/(torch.pi*10))) * pred 

    def compute_pde_residual(self, pred, input): 
        """
        Compute PDE loss using autograd
        """

        # -(Cos[1/x]/x) + Sin[1/x]

        dx = torch.autograd.grad(pred.sum(), input, create_graph=True)[0]
        f = -(torch.cos(6/input)/input +1/6*torch.sin(6/input))
        
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

        return input/6*torch.sin(6/input)