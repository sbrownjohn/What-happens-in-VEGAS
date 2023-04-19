from torch.distributions.normal import Normal
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt
import itertools
import torch.utils.data as data

def eval_f(f,Z):
    z = Z.detach().numpy()
    return torch.FloatTensor(f(z))

def exp_loss(f,Z,log_det_jacobian,X,base,is_numpy):
    if is_numpy: 
        fs = eval_f(f,Z)
    else:
        fs = f(Z)
    log_pz = -log_det_jacobian
    D = log_pz.exp()*torch.square(log_pz - torch.absolute(fs).log())
    return D.mean()

def kl_loss(f,Z,log_det_jacobian,X,base):
    fs = eval_f(f,Z)
    log_pz = base.log_prob(X).sum(dim=1) - log_det_jacobian
    D = log_pz.exp()*(log_pz - torch.absolute(fs).log())
    return D.mean()

def loss_func(f,Z,log_det_jacobian,X,base,is_numpy):
    if is_numpy: 
        fs = eval_f(f,Z)
    else:
        fs = f(Z)
    pz = (-log_det_jacobian).exp()
    return torch.var(fs/pz).mean()

def preburn_loss(f,Z,log_det_jacobian,X,base,is_numpy):
    if is_numpy: 
        fs = eval_f(f,X)
    else:
        fs = f(X)
    pz = (-log_det_jacobian).exp()
    return (fs/pz).mean()

def masking(n_dim):
    if n_dim != 3:
        dims = range(n_dim)
        dims = [list("{0:b}".format(i)) for i in dims]
        m = len(max(dims,key=len))
        masks = []
        for j in range(m):
            mask = []
            for i in range(len(dims)):
                if len(dims[i]) == 0:
                    mask.append(0)
                else:
                    mask.append(int(dims[i].pop(-1)))
            masks.append(mask)
            masks.append(list(1-np.array(mask)))
        return masks
    
    return [[1,1,0],[0,1,1],[1,0,1]]

def create_base(n_dim:int,coupling_transform:str = 'piecewise_quadratic'):
    if coupling_transform in ['piecewise_quadratic','piecewise_linear']:
        A = torch.FloatTensor([0.0]*n_dim)
        B = torch.FloatTensor([1.0]*n_dim)
        base = Uniform(A, B)
        return base
    elif coupling_transform == 'affine':
        A = torch.FloatTensor([0.5]*n_dim)
        B = torch.FloatTensor([0.3]*n_dim)
        base = Normal(A, B)
        return base
    else:
        raise ValueError('coupling_transform must be one of "piecewise_quadratic","piecewise_linear","affine". Got {0}'.format(coupling_transform))

class Affine_Flow(nn.Module):
    def __init__(self,n_dim, mask, hidden_size=64, num_hidden_layers=2):
        super(Affine_Flow, self).__init__()
        self.mlp = FCNN(n_dim, hidden_size, num_hidden_layers, 2)
        self.mask = torch.tensor(mask)
        self.s_scale = nn.Parameter(torch.ones(1).float(), requires_grad=True)
        self.s_shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, reverse=False):
        x_masked = x * self.mask
        log_s, t = self.mlp(x_masked).chunk(2, dim=1)
        log_s = log_s.tanh() * self.s_scale + self.s_shift
        t = t  * (1-self.mask)
        log_s = log_s * (1-self.mask)
        if reverse:
            x = (x - t) * torch.exp(-log_s)
        else:
            x = x * torch.exp(log_s) + t
        return x, log_s.sum(dim=1)
 
class Cdf_Flow(nn.Module):
    def __init__(self,n_dim, mask, hidden_size=64, num_hidden_layers=2):
        super(Cdf_Flow, self).__init__()
        self.mlp = FCNN(n_dim, hidden_size, num_hidden_layers, 2)
        self.mask = torch.tensor(mask)

    def forward(self, x, reverse=False):
        x_masked = x * self.mask
        log_sigma, mu = self.mlp(x_masked).chunk(2, dim=1)
        distribution = Normal(mu, log_sigma.exp())
        dist_x = distribution.cdf(x) * (1-self.mask)
        log_J = (distribution.log_prob(x) * (1-self.mask)).sum(dim=1)
        z = dist_x + x*self.mask
        return z, log_J

class PW_Linear(nn.Module):
    def __init__(self,mask,n_bins,hidden_size=64,n_hidden_layers=2):
        super(PW_Linear,self).__init__()
        self.mask = torch.tensor(mask)
        self.n_dim = self.mask.size()[0]
        self.K = n_bins
        indices = torch.arange(0,self.n_dim)
        self.indices_copied = indices[self.mask == 0]
        self.n_dim_copied = len(self.indices_copied)
        self.indices_transformed = indices[self.mask == 1]
        self.n_dim_transformed = len(self.indices_transformed)
        output_size = self.n_dim_transformed*self.K
        self.net = FCNN(self.n_dim_copied,hidden_size,n_hidden_layers,output_size)
        
    def forward(self,x):
        assert x.size()[1] == self.n_dim, "x (dim = {0}) must have same dimension as mask (dim = {1})".format(x.size()[1],self.n_dim)
        x_c = x[:,self.indices_copied]
        x_t = x[:,self.indices_transformed]
        
        W = self.net(x_c)
        W_n = F.softmax(W.float(),dim=-1)
        b = torch.floor(x_t*self.K).type(torch.int64)
        alpha = (x_t*self.K) - b
        W_cum = W_n.cumsum(dim=-1)
        W_cum= F.pad(W_cum,(1,0),'constant',0)
        z_t = alpha*torch.gather(W_n,-1,b) + torch.gather(W_cum,-1,b)
        log_det_J = (torch.gather(W_n,-1,b)*self.K).log().sum(dim=-1)
        z = torch.zeros_like(x)
        z[:,self.indices_copied] = x_c
        z[:,self.indices_transformed] = z_t
        return z, log_det_J

class FCNN(nn.Module):
    '''
    input_size => number of dimensions copied
    hidden_size => number of features within hidden layer
    num_hidden_layers => number of hidden layers
    output_size => variable; depends of coupling transform
    '''
    def __init__(self, input_size:int, hidden_size:int, num_hidden_layers:int, output_size:int):
        super(FCNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class PW_Quad(nn.Module): 
    '''
    mask => 1D array of 1,0 and length of input vector. 1 indicates dimension will be transformed by coupling layer, 0 indicates copying
    k_bins => number of bins along each dimension for piecewise pdf
    '''
    def __init__(self, mask, K_bins, hidden_size=64, num_hidden_layers=2):
        super(PW_Quad, self).__init__()
        self.mask = torch.tensor(mask)
        self.n_dim = self.mask.size()[0]
        self.K = K_bins
        indices = torch.arange(0,self.n_dim)
        self.indices_copied = indices[self.mask == 0]
        self.n_dim_copied = len(self.indices_copied)
        self.indices_transformed = indices[self.mask == 1]
        self.n_dim_transformed = len(self.indices_transformed)
        output_size = self.n_dim_transformed*(2*self.K+1)
        self.mlp = FCNN(self.n_dim_copied,hidden_size,num_hidden_layers,output_size)
    
    def _normalise_V(self,V_unnormalised,W):
        V_size = self.K + 1
        V_denom = torch.zeros((V_unnormalised.size()[0],self.n_dim_transformed))
        for i in range(V_size-1):
            V_denom += 0.5*(torch.exp(V_unnormalised[:,:,i]) + torch.exp(V_unnormalised[:,:,i+1]))*W[:,:,i]
        V_denom = V_denom.view(-1,self.n_dim_transformed,1).expand(-1,self.n_dim_transformed,V_size)
        V = torch.exp(V_unnormalised)/V_denom
        return V
        
    def forward(self,x):
        x_copied = x[:,self.indices_copied]
        x_transformed = x[:,self.indices_transformed]
        
        params = self.mlp(x_copied)
        
        V_unnormalised = params[:,:(self.n_dim_transformed*(self.K+1))].view(-1,self.n_dim_transformed,self.K+1)
        W_unnormalised = params[:,(self.n_dim_transformed*(self.K+1)):].view(-1,self.n_dim_transformed,self.K)
        W = F.softmax(W_unnormalised,dim=-1)
        V = self._normalise_V(V_unnormalised,W)
        
        W_cum = W.cumsum(dim=-1)
        W_mask = W_cum > x_transformed.view(-1,self.n_dim_transformed,1).expand(-1,self.n_dim_transformed,self.K)
        b = torch.argmax(W_mask.long(),-1,keepdim=True)
        W_cum = F.pad(W_cum,(1,0),'constant',0)
        alpha = (x_transformed.view(-1,self.n_dim_transformed,1) - torch.gather(W_cum,-1,b))/torch.gather(W,-1,b)

        C_intercept = torch.ones_like(V)
        for i in range(0,self.K):
            C_intercept[:,:,i] = 0.5*(V[:,:,i] + V[:,:,i+1])*W[:,:,i]
        C_intercept_cum = C_intercept.cumsum(dim=-1)
        C_intercept_cum = F.pad(C_intercept_cum,(1,0),'constant',0)
        C = 0.5*torch.square(alpha)*(torch.gather(V,-1,b+1) - torch.gather(V,-1,b))*torch.gather(W,-1,b) + alpha*torch.gather(V,-1,b)*torch.gather(W,-1,b) + torch.gather(C_intercept_cum,-1,b)
        C = C.view(-1,self.n_dim_transformed)
        z = torch.ones_like(x)
        z[:,self.indices_copied] = x_copied
        z[:,self.indices_transformed] = C
        z[z>1] = 1.0
        
        q = torch.lerp(torch.gather(V,-1,b),torch.gather(V,-1,b+1),alpha)
        log_J = q.log().view(-1,self.n_dim_transformed).sum(dim=-1)
        
        return z, log_J

class NF(nn.Module):
    def __init__(self, transforms):
        super(NF, self).__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x):
        z, log_det_jacobian = x, torch.zeros_like(x).sum(dim=1)
        for transform in self.transforms:
            z, log_J = transform(z)
            log_det_jacobian += log_J
        return z, log_det_jacobian

    def invert(self, z):
        for transform in self.transforms[::-1]:
            z, _ = self.transform(z)
        return z

class EarlyStopper:
    def __init__(self,tolerance,bl_tolerance,delta_tolerance,min_delta):
        self.counter = 0
        self.early_stop = False
        self.tolerance = tolerance
        self.bl_tolerance = bl_tolerance
        self.bl_counter = 0
        self.delta_tolerance = delta_tolerance
        self.delta_counter = 0
        self.min_delta = min_delta
        
    def __call__(self,loss,prev_loss,best_loss):
        if loss.item() > prev_loss:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
                return
        else:
            self.counter = 0
            
        if loss.item() > best_loss:
            self.bl_counter += 1
            if self.bl_counter >= self.bl_tolerance:
                self.early_stop = True
                return
        else:
            self.bl_counter = 0
            
        if abs(loss.item() - prev_loss)/loss.item() < self.min_delta:
            self.delta_counter += 1
            if self.delta_counter >= self.delta_tolerance:
                self.early_stop = True
                return
        else:
            self.delta_counter = 0
            
          
class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]
            
class BasicTrainer:
    def __init__(self,model,f,base):
        self.model = model
        self.f = f 
        self.base = base
        self.best_model = model.state_dict()
        self.integrals = []
    
    def train(self,plot_loss=False,is_numpy=False,early_stopper_on=True,integrate=True,n_epochs=50,loss_function=loss_func,lr=1e-3,n_samples=10_000,tolerance=7,bl_tolerance = 7, delta_tolerance=7,min_delta=1e-3):
        self.model.train()
        optimizer = torch.optim.Adam(params=self.model.parameters(),lr=lr)
        losses = []
        early_stopper = EarlyStopper(tolerance,bl_tolerance,delta_tolerance,min_delta)
        best_loss = np.inf
        for i in range(n_epochs):
            ys = self.base.sample((n_samples,))
            ys = ys.float()
            xs, dx_by_dy = self.model(ys)
            loss = loss_function(self.f, xs, dx_by_dy,ys,self.base,is_numpy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if integrate:
                if is_numpy:
                    fs = eval_f(self.f,xs)
                else:
                    fs = self.f(xs)
                pz = (-dx_by_dy).exp()
                I = (fs/pz).mean()
                std = torch.std(fs/pz).mean()
                std = std/np.sqrt(n_samples)
                if not torch.isnan(I).any() and not torch.isnan(std).any():
                    self.integrals.append((I.item(),std.item()))
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.best_model = deepcopy(self.model.state_dict())
                
            if i > 9:
                if early_stopper_on:
                    early_stopper(loss,losses[-1],best_loss)
                    if early_stopper.early_stop:
                        losses.append(loss.item())
                        if plot_loss:
                            plt.figure()
                            print(losses)
                            plt.plot(range(i+1),losses)
                        return best_loss, i
            
            losses.append(loss.item())
        if plot_loss:
            plt.figure()
            print(losses)
            plt.plot(range(n_epochs),losses)
        
        return best_loss, i
    
    def batched_training(self,batch_size=1280,plot_loss=False,is_numpy=False,early_stopper_on=True,integrate=True,n_epochs=50,loss_function=loss_func,lr=1e-3,n_samples=10_000,tolerance=7,bl_tolerance = 7, delta_tolerance=7,min_delta=1e-3):
        n_train = n_samples
        train_data = self.base.sample((n_train,))

        train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=batch_size, shuffle=True)
        early_stopper = EarlyStopper(tolerance,bl_tolerance,delta_tolerance,min_delta)
        best_loss = np.inf
        self.model.train()
        optimizer = torch.optim.Adam(params=self.model.parameters(),lr=1e-3)
        losses=[]
        for i in range(n_epochs):
            for x in train_loader:
                x = x.float()
                z, dz_by_dx = self.model(x)
                loss = loss_function(self.f, z, dz_by_dx,x,self.base,is_numpy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if integrate:
                if is_numpy:
                    fs = eval_f(self.f,z)
                else:
                    fs = self.f(z)
                pz = (-dz_by_dx).exp()
                I = (fs/pz).mean()
                std = torch.std(fs/pz).mean()
                std = std/np.sqrt(batch_size)
                if not torch.isnan(I).any() and not torch.isnan(std).any():
                    self.integrals.append((I.item(),std.item()))
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.best_model = deepcopy(self.model.state_dict())
                
            if i > 5:
                if early_stopper_on:
                    early_stopper(loss,losses[-1],best_loss)
                    if early_stopper.early_stop:
                        losses.append(loss.item())
                        if plot_loss:
                            plt.figure()
                            print(losses)
                            plt.plot(range(i+1),losses)
                        
                        return best_loss, i
            
            losses.append(loss.item())
        if plot_loss:
            plt.figure()
            print(losses)
            plt.plot(range(n_epochs),losses)
        
            
        return best_loss, i
    
    def integrate(self,n_samples,is_numpy=False):
        with torch.no_grad():
            ys = self.base.sample((n_samples,))
            ys = ys.float()
            xs, dx_by_dy = self.model(ys)
            if is_numpy:
                fs = eval_f(self.f,xs)
            else:
                fs = self.f(xs)
            pz = (-dx_by_dy).exp()
            I = (fs/pz).mean()
            std = torch.std(fs/pz).mean()
            std = std/np.sqrt(n_samples)
            self.integrals.append((I.item(),std.item()))
            return
    
    def result(self):
        I = 0
        var_total = 0
        for val in self.integrals:
            if val[0] != torch.nan and val[1] != torch.nan:
                var_total += 1/(val[1])**2
        
        for val in self.integrals:
            if val[0] != np.nan and val[1] != np.nan:
                I += (val[0]/val[1]**2)/var_total
        
        return I, np.sqrt(1/var_total)
    
    def __call__(self,plot_loss=False,is_numpy=False,early_stopper_on=True,integrate=True,n_epochs=50,loss_function=loss_func,lr=1e-3,n_samples=10_000,tolerance=7,bl_tolerance = 7, delta_tolerance=7,min_delta=1e-3):
        best_loss, i = self.train(plot_loss=plot_loss,is_numpy=is_numpy,early_stopper_on=early_stopper_on,integrate=integrate,n_epochs=n_epochs,loss_function=loss_function,lr=lr,n_samples=n_samples,tolerance=tolerance,bl_tolerance=bl_tolerance, delta_tolerance=delta_tolerance,min_delta=min_delta)
        self.model.load_state_dict(self.best_model)
        remaining = n_epochs - 1 - i
        for i in range(remaining):
            self.integrate(n_samples,is_numpy)
        return self.result()
    
    def batched_integrate(self,batch_size=1280,plot_loss=False,is_numpy=False,early_stopper_on=True,integrate=True,n_epochs=50,loss_function=loss_func,lr=1e-3,n_samples=10_000,tolerance=7,bl_tolerance = 7, delta_tolerance=7,min_delta=1e-3):
        best_loss, i = self.batched_training(batch_size=batch_size,plot_loss=plot_loss,is_numpy=is_numpy,early_stopper_on=early_stopper_on,integrate=integrate,n_epochs=n_epochs,loss_function=loss_function,lr=lr,n_samples=n_samples,tolerance=tolerance,bl_tolerance=bl_tolerance, delta_tolerance=delta_tolerance,min_delta=min_delta)
        self.model.load_state_dict(self.best_model)
        remaining = n_epochs - 1 - i
        for i in range(remaining):
            self.integrate(n_samples)
        return self.result()

class TrainedDistribution():
    def __init__(self,model,base):
        self.model = model
        self.base = base
    
    def sample(self,n_samples):
        with torch.no_grad():
            ys = self.base.sample((n_samples,))
            ys = ys.float()
            xs, dx_by_dy = self.model(ys)
            return xs.detach().numpy()
    
    def sample_and_log_prob(self,n_samples):
        with torch.no_grad():
            ys = self.base.sample((n_samples,))
            ys = ys.float()
            xs, dx_by_dy = self.model(ys)
            log_prob = (-dx_by_dy)
            return xs.detach().numpy(), log_prob
    
    def log_prob(self,x):
        with torch.no_grad():
            ys, dx_by_dy = self.model.inverse(x)
            log_prob = (-dx_by_dy)
            return log_prob
    
    def _axis_combinations(self,n_dim):
        dims = range(n_dim)
        combinations = itertools.combinations(dims,2)
        return list(combinations)
    
    def plot_distribution(self,resolution='low',combinations=None,save=None,axes_off=False):
        assert resolution in ['low','high']
        if resolution == 'low':
            samples = self.sample(40_000)
            n_bins = 30
        if resolution == 'high':
            samples = self.sample(400_000)
            n_bins = 60
            
        n_dim = len(samples[0])
        if combinations == None:
            combinations = self._axis_combinations(n_dim)
        assert type(combinations) == list
        assert type(combinations[0]) == tuple
        n_plots = len(combinations)
        n_rows = (n_plots // 3) + 1
        if n_dim >2:
            plt.figure(figsize=(15,5*n_rows))
            for i in range(n_plots):
                plt.subplot(n_rows,3,i+1)
                plt.hist2d(samples[:,combinations[i][0]],samples[:,combinations[i][1]],n_bins,density=True,cmap='magma')
                plt.colorbar()
                plt.xlabel('x{0}'.format(combinations[i][0]))
                plt.ylabel('x{0}'.format(combinations[i][1]))
        else:
            plt.figure()
            plt.hist2d(samples[:,combinations[0][0]],samples[:,combinations[0][1]],n_bins,density=True,cmap='magma')
            plt.colorbar()
            plt.xlabel('x{0}'.format(combinations[0][0]))
            plt.ylabel('x{0}'.format(combinations[0][1]))
        
        if axes_off:
            plt.axis('off')   
        if save != None:
            plt.savefig(save)
        plt.show()
            
        return

def construct_flow(n_dim:int,coupling_transform:str,masks:any,hidden_size:int=64,num_hidden_layers:int=2,K_bins:int=16):
    base = create_base(n_dim,coupling_transform)
    if masks == None:
        masks = masking(n_dim)
    if coupling_transform == 'piecewise_quadratic':
        flows = []
        for mask in masks:
            flow = PW_Quad(mask,K_bins,hidden_size,num_hidden_layers)
            flows.append(flow)
        return NF(flows), base
    elif coupling_transform == 'piecewise_linear':
        flows = []
        for mask in masks:
            flow = PW_Linear(mask,K_bins,hidden_size,num_hidden_layers)
            flows.append(flow)
        return NF(flows), base
    elif coupling_transform == 'affine':
        flows = []
        for mask in masks:
            flow = Affine_Flow(n_dim,mask,hidden_size,num_hidden_layers)
            flows.append(flow)
        for mask in masks:
            flow = Cdf_Flow(n_dim,mask,hidden_size,num_hidden_layers)
            flows.append(flow)
        return NF(flows), base

class NF_Integrator:
    def __init__(self,limits:list,f:callable,masks=None,is_numpy=False,flow=None,coupling_transform:str='piecewise_quadratic',hidden_size:int=64,num_hidden_layers:int=2,K_bins:int=16):
        self.limits = limits
        self.f = f
        self.is_numpy = is_numpy
        self.n_dim = len(limits)
        if flow == None:
            self.flow, self.base = construct_flow(self.n_dim,coupling_transform,masks,hidden_size,num_hidden_layers,K_bins)
        else:
            raise NotImplementedError('This feature has not been implemented')
        
        self.trained_distrubition = TrainedDistribution(self.flow,self.base)
        
    def __call__(self,batch_size=1280,batched=False,adapt=True,n_samples=10000,n_epochs=100,plot_loss=False,early_stopper_on=True,loss_function=loss_func,lr=1e-3,tolerance=7,bl_tolerance = 7, delta_tolerance=7,min_delta=1e-3):
        trainer = BasicTrainer(self.flow,self.f,self.base)
        if adapt:
            if batched:
                result = trainer.batched_integrate(batch_size=batch_size,plot_loss=plot_loss,is_numpy=self.is_numpy,n_samples=n_samples,early_stopper_on=early_stopper_on,n_epochs=n_epochs,loss_function=loss_function,lr=lr,tolerance=tolerance,bl_tolerance=bl_tolerance,delta_tolerance=delta_tolerance,min_delta=min_delta)
                self.flow.load_state_dict(trainer.best_model)
                self.trained_distrubition = TrainedDistribution(self.flow,self.base)
                return result
                
            result = trainer(plot_loss=plot_loss,is_numpy=self.is_numpy,n_samples=n_samples,early_stopper_on=early_stopper_on,n_epochs=n_epochs,loss_function=loss_function,lr=lr,tolerance=tolerance,bl_tolerance=bl_tolerance,delta_tolerance=delta_tolerance,min_delta=min_delta)
            self.flow.load_state_dict(trainer.best_model)
            self.trained_distrubition = TrainedDistribution(self.flow,self.base)
            return result
        else:
            for i in range(n_epochs):
                trainer.integrate(n_samples)
            result = trainer.result()
            return result
        
        
if __name__ == '__main__':
    def integrand(X):
        r1 = torch.sqrt(torch.square(X[:,0]-0.5) + torch.square(X[:,1]-0.5))
        return torch.exp(-500*torch.square(r1-0.3))

    integrator = NF_Integrator([[0,1]]*2,integrand,coupling_transform='piecewise_linear')
    result = integrator(batched=True,n_epochs=100,n_samples=20_000,plot_loss=True)
    print(result)
    
    integrator = NF_Integrator([[0,1]]*2,integrand,coupling_transform='piecewise_quadratic')
    result = integrator(batched=True,n_epochs=100,n_samples=20_000,plot_loss=True)
    print(result)
    #integrator.trained_distrubition.plot_distribution('high')