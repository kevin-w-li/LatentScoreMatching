import torch
import numpy as np
from .VAEs import AutoEncoder
import torch.nn as nn
import tqdm
from scipy.special import logsumexp

class BilevelAutoEncoder(nn.Module):
    
    def __init__(self, Dz, Dh, Dx, init_std=0.2, update_step_size = 0.01, bilevel=True):      
        
        super().__init__()
        self.M = 1
        
        self.update_step_size = update_step_size
        
        if bilevel:
            self.e_W1 = (torch.randn(Dx, Dh) * init_std).requires_grad_(True)
            self.e_W2 = (torch.randn(Dh, Dh)    * init_std).requires_grad_(True)
            self.e_W3 = (torch.randn(Dh, Dz*2)  * init_std).requires_grad_(True)
            self.e_b1 = (torch.randn(Dh) * 0.0).requires_grad_(True)
            self.e_b2 = (torch.randn(Dh) * 0.0).requires_grad_(True)
            self.e_b3 = (torch.randn(Dz*2) * 0.0).requires_grad_(True)
        
        else:
        
            self.e_W1 = nn.Parameter(torch.randn(Dx, Dh) * init_std)
            self.e_W2 = nn.Parameter(torch.randn(Dh, Dh)    * init_std)
            self.e_W3 = nn.Parameter(torch.randn(Dh, Dz*2)  * init_std)
            self.e_b1 = nn.Parameter(torch.randn(Dh) * 0.0)
            self.e_b2 = nn.Parameter(torch.randn(Dh) * 0.0)
            self.e_b3 = nn.Parameter(torch.randn(Dz*2) * 0.0)

        
        self.d_W1 = nn.Parameter(torch.randn(Dz, Dh) * init_std)
        self.d_W2 = nn.Parameter(torch.randn(Dh, Dh) * init_std)
        self.d_W3 = nn.Parameter(torch.randn(Dh, Dx) * init_std)

        self.d_b1 = nn.Parameter(torch.randn(Dh) * init_std)
        self.d_b2 = nn.Parameter(torch.randn(Dh) * init_std)
        self.d_b3 = nn.Parameter(torch.randn(Dx) * init_std)

        
        self.phi_names = ['e_W1', 'e_W2', 'e_W3', 'e_b1', 'e_b2', 'e_b3']
        self.theta_names = ['d_W1', 'd_W2', 'd_W3', 'd_b1', 'd_b2', 'd_b3', 'logsigma']

        self.states = dict([(k, (torch.zeros_like(getattr(self, k)), torch.zeros_like(getattr(self, k)))) for k in self.phi_names])
        self.states['count'] = 0

        
        self.prior = torch.distributions.Normal(torch.zeros(Dz),1)
        
        self.nl = torch.relu
        self.logsigma = nn.Parameter(torch.ones([])*np.log(0.1))
        
    def decode(self, z):
                
        h = self.nl(z @ self.d_W1 + self.d_b1)
        h = self.nl(h @ self.d_W2 + self.d_b2)
        x = (h @ self.d_W3 + self.d_b3)
        p = torch.distributions.Normal(x, self.logsigma.exp())
        
        return p
    
    def encode(self, x):

        h = self.nl(x @ self.e_W1 + self.e_b1)
        h = self.nl(h @ self.e_W2 + self.e_b2)
        h = h @ self.e_W3 + self.e_b3

        m, std = torch.split(h, 2, dim=-1)
        std = torch.nn.functional.softplus(std)
        
        q = torch.distributions.Normal(m, std)
        
        return q
    
    @property
    def phi(self):
        return [getattr(self, k) for k in self.phi_names]
    
    @property
    def theta(self):
        return [getattr(self, k) for k in self.theta_names]

    def _update_encoder(self, dldphi, b1 = 0.9, b2=0.999, eps=10**-6, step_size=0.001):

        i = self.states['count']
        for n, g in zip(self.phi_names, dldphi):
            v = getattr(self, n)
            #m, s = self.states[n]
            #m = (1 - b1) * g      + b1 * m  # First  moment estimate.
            #s = (1 - b2) * (g**2) + b2 * s  # Second moment estimate.
            #mhat = m / (1 - b1**(i + 1))    # Bias correction.
            #shat = s / (1 - b2**(i + 1))
            #self.states[n] = mhat, shat
            #setattr(self, n, v + mhat / (shat.sqrt() + eps) * self.update_step_size)
            setattr(self, n, v + g * self.update_step_size)
        self.states['count'] += 1

        
    def kl_encoder(self, x, reparam=False, bilevel=False):
            
        logp = self.ELBO(x)
        dldphi = torch.autograd.grad(logp, self.phi, create_graph=bilevel)
        self._update_encoder(dldphi)
        
        return dldphi
        
                    
    def f_encoder(self, x, reparam=False, bilevel=False):
            
        pz_x = self.encode(x)
        
        if reparam:
            z = pz_x.rsample([])
        else:
            z = pz_x.sample([])
            z.requires_grad_()

        
        px_z = self.decode(z)
        
        logpx_z = px_z.log_prob(x).sum(-1)
      
        dzlogpz_x = - (z - pz_x.mean) /  (pz_x.stddev**2)
        dzlogpx_z = torch.autograd.grad((logpx_z).sum(), [z], create_graph=bilevel)[0]
        dzlogpz = -z
      
        loss = ((dzlogpz_x - dzlogpx_z - dzlogpz)**2).sum(-1).mean()

        dldphi = torch.autograd.grad(-loss, self.phi, create_graph=reparam)
        self._update_encoder(dldphi)
        
    def zero_encoder_grad(self):
        for v in self.phi:
            v.grad = None
            v.detach_()
            v.requires_grad_(True)
        for n in self.phi_names:
            for s in self.states[n]:
                s.grad = None
                s.detach_()
                s.requires_grad_(True)

            
            
    def joint_score_x(self, x):
                
        pz_x = self.encode(x)
        z = pz_x.rsample([])
        px_z = self.decode(z)
        
        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                (0.5 * dxlogpx_z.square() - (1. / px_z.stddev**2)).sum(-1)
               ).mean()

#         logp = px_z.log_prob(x).sum(-1).mean(0) - torch.distributions.kl.kl_divergence(pz_x, self.prior).sum(-1).mean(0)

        
        return loss
    
    def ELBO(self, x):
                
        pz_x = self.encode(x)
        z = pz_x.rsample([])
        px_z = self.decode(z)
        logp = px_z.log_prob(x).sum(-1).mean(0) - torch.distributions.kl.kl_divergence(pz_x, self.prior).sum(-1).mean(0)

        
        return logp
    
    

    def marginal_score(self, x, detach=False):

        pz_x = self.encode(x)
        z = pz_x.rsample([])
        px_z = self.decode(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                - 0.5 * dxlogpx_z.mean(0).square().sum(-1)
                + dxlogpx_z.square().sum(-1).mean(0)
                - (1 / px_z.stddev**2).mean(0).sum(-1)
               ).mean()

        return loss
    
    def bad_marginal_score(self, x):

        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.rsample([])
        px_z = self.decode(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        dxlogpz_x = torch.autograd.grad((logpz_x).sum(), [x], create_graph=True)[0]
        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                0.5 * dxlogpx_z.mean(0).square().sum(-1)
                + (dxlogpx_z * dxlogpz_x).sum(-1).mean(0)
                - (1 / px_z.stddev**2).mean(0).sum(-1)
               ).mean()

        return loss
    
    @torch.no_grad()
    def log_likelihood(self, x, nsample = 100000, batch_size=1000):
        x = torch.as_tensor(x)
        batch_size = min(nsample, batch_size)
        
        n = x.shape[0]
        m = nsample // batch_size
        ws = []
        for i in tqdm.trange(m):
            z = self.prior.sample([batch_size])
            px_z = self.decode(z[None])
            w = px_z.log_prob(x[:,None,:]).sum(-1)
            ws += w.cpu(),
        w = torch.cat(ws, 1)
        loglik = w.logsumexp(1) - np.log(nsample)
        return loglik
    
    @torch.no_grad()
    def FD(self, x, nsample=10000, batch_size=1000):
        x = torch.as_tensor(x)
        batch_size = min(nsample, batch_size)
        
        m = nsample // batch_size
        fs = []
        ws = []
        cs = []

        for i in tqdm.trange(m):
            z = self.prior.sample([batch_size])
            px_z = self.decode(z[None])
            w = px_z.log_prob(x[:,None,:]).sum(-1)
            ws += w,
            s   = - (x[:,None,:] - px_z.mean) / px_z.stddev**2 
            fs += s,
            cs += - (1. / px_z.stddev**2).sum(-1),

        f = torch.cat(fs, 1)
        w = torch.cat(ws, 1)
        c = torch.cat(cs, 1)
        n = f.square().sum(-1)
        
        w -= w.max(dim=1, keepdim=True)[0]
        w = np.exp(w)

        fd = (w * (n + c)).sum(1) / w.sum(1) - 0.5 * ( (f * w[...,None]).sum(1) / w.sum(1, keepdim=True) ).square().sum(-1)

        return fd

        
        
    @torch.no_grad()
    def sample(self, n, add_noise = False):
        z  = self.prior.sample([n])
        x = self.decode(z).mean
        return x

           
           
            
        
        
