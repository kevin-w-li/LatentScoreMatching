import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .Generators import dc_gen, fc_gen, large_dc_gen
from .Features import dc_feat, fc_feat
from .KernelWakeSleep import KRR, KRR_rf
from .ResidualNetworks import ResNet18Enc, ResNet18Dec, ResNet

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class AutoEncoder(nn.Module):
    
    def __init__(self, Dz, Dh, Dx, fit_sigma=True, layer_type="conv", ngf=64, nc=1, ndf = 64, ebm_Dh = 100,
            sm_noise_std=0.1, score_v=1, M = 5, denoise_ae=False, im_size = 32, device=torch.device("cuda:0"), init_log_sigma=0.0,
            attention = False, sigmoid=True, nl="relu", prior = 'gaussian', bn=False):
        super().__init__()

        if nl == "elu":
            nl_layer = nn.ELU
        else:
            nl_layer = nn.ReLU
        
        self.Dz = Dz
        self.Dx = Dx
        self.Dh = Dh
        self.M = M
        self.device=device
        self.denoise_ae = denoise_ae
        self.prior_mu = nn.Parameter(torch.zeros(Dz, device=device), requires_grad=False)
        self.prior_logsigma = nn.Parameter(torch.zeros(Dz, device=device), requires_grad=False)
        if prior == 'gaussian':
            self.prior = td.Independent(td.Normal(self.prior_mu, self.prior_logsigma.exp()), 1)
        if prior == 'ebm':
            self.prior = EBM(Dz, Dh=ebm_Dh, device=device, bn=False)
        
        self.attention = attention
        self.attender = nn.Sequential(ResNet(Dz, 300, 300), nn.Linear(300, Dz), nn.Sigmoid())
        
        if layer_type == "fc":
            self.decoder = fc_gen(Dz, Dh, Dh, Dx, tanh=False, sigmoid=sigmoid, nl_layer=nl_layer, bn=bn)
            self.encoder = fc_feat(Dx, Dh, Dh, Dz*2, final_bn=False, nl_layer=nl_layer, bn=bn)
        elif layer_type == "conv":
            self.decoder = dc_gen(Dz,ngf, tanh=False, sigmoid=sigmoid, nc=nc, nl_layer=nl_layer)
            self.encoder = dc_feat(nfinal = Dz*2, ndf=ndf, nc = nc, final_bn=False, nl_layer=nl_layer)
        elif layer_type == "resnet":
            self.encoder = ResNet18Enc(z_dim=Dz, nc = nc, im_size = im_size)
            self.decoder = ResNet18Dec(z_dim=Dz, nc = nc, im_size = im_size)

        
        self.sm_noise_std = sm_noise_std
        self.logsigma = nn.Parameter(torch.tensor([init_log_sigma]), requires_grad=fit_sigma)
        self.score_v = score_v
            
        self.enc_losses = []
        self.dec_losses = []
        self.ELBO_losses = []
        self.recon_losses = []
        self.MMDs = []
        self.q_hists = []
        self.joint_fds = []
    
        
        
    def evaluate(self, test_data_loader, noise_std=None):

        if noise_std is None:
            noise_std = self.sm_noise_std
        
        losses = []
        samples = []
        ELBOs = []
        encs = []
        decs = []
        qstds = []
        JFDs = []
        
        for idx, (images, _) in enumerate(test_data_loader):
            bs = images.shape[0]
            clean_x = torch.as_tensor(images).view(bs, -1).to(self.device)
            x = clean_x + torch.randn_like(clean_x) * noise_std
            q = self.encode(x)
            qstds += q.stddev.detach().cpu(),

            s = q.sample([])
            samples += s.cpu(),

            recon_x = self.decoder(q.mean).view(bs,-1)
            losses += (clean_x - recon_x).square().mean().item(),
            if not isinstance(self.prior, EBM):
                ELBOs += self.ELBO(clean_x).item(),
            else:
                ELBOs += np.nan,
            encs  += self.score_z(clean_x).item(),
            decs  += self.score_x(clean_x).item(),
            JFDs  += self.joint_score(clean_x).item(),
            
        self.ELBO_losses += np.mean(ELBOs),
        self.recon_losses += np.mean(losses),
        self.dec_losses += np.mean(decs),
        self.enc_losses += np.mean(encs),
        self.joint_fds += np.mean(JFDs),
        
        qstds = torch.cat(qstds).numpy().T
        self.q_hists += np.asarray([np.histogram(s, bins=np.linspace(0,1.5,31))[0] for s in qstds]), 
        
        q_samples = torch.cat(samples)
        nsample = q_samples.shape[0]
        p_samples = self.prior.sample([nsample]).cpu()
        
        kernel = lambda X, Y: (X@Y.T + 1)**3
        
        KXX = kernel(q_samples, q_samples)
        KXY = kernel(q_samples, p_samples)
        KYY = kernel(p_samples, p_samples)
        
        MMD = (KXX.sum() - KXX.trace()) / nsample / (nsample-1) + \
              (KYY.sum() - KYY.trace()) / nsample / (nsample-1) - \
              2 * KXY.mean()
        
        self.MMDs += MMD.item(),

    def sample(self, N, noisy=True, **ebm_params):
        if isinstance(self.prior, EBM):
            z = self.prior.sample([N], **ebm_params)
        else:
            z = self.prior.sample([N])


        if noisy:
            x = self.decode(z).sample([])
        else:
            x = self.decode(z).mean.detach()
        return x

    
    def decode(self, z, sigma=None):
        z_shape = z.shape[:-1]
        z = z.reshape(-1, self.Dz)
        if sigma is None:
            sigma = self.logsigma.exp()

        if self.attention:
            z = z * self.attender(z)

        mu = self.decoder(z)
        return td.Independent(td.Normal(mu.reshape(*z_shape,-1), sigma), 1)
        
    def encode(self, x, sigma=None):
        h = self.encoder(torch.cat([x], -1))
        mu, std = torch.split(h, [self.Dz, self.Dz], dim=-1)
        return td.Independent(td.Normal(mu, std.exp()), 1)
    
    def ELBO(self, clean_x,  reparam=True):
        
        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        pz_x = self.encode(x)
        z = pz_x.rsample([M])
        px_z= self.decode(z)
        if self.denoise_ae:
            loss = - (px_z.log_prob(clean_x).mean(0)).mean() + td.kl_divergence(pz_x, self.prior).mean()
        else:
            loss = - (px_z.log_prob(x).mean(0)).mean() + td.kl_divergence(pz_x, self.prior).mean()
        return loss
    
    def score_z(self, clean_x, reparam=False):
        
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        
        M = self.M
        pz_x = self.encode(x)
        if reparam:
            z = pz_x.rsample([M])
        else:
            z = pz_x.sample([M])
        z.requires_grad_()
        px_z = self.decode(z)

        logpx_z = px_z.log_prob(x)
        dzlogpz_x = - (z - pz_x.mean) /  (pz_x.stddev**2)
        dzlogpx_z = torch.autograd.grad((logpx_z).sum(), [z], create_graph=True)[0]

        if isinstance(self.prior, EBM):
            dzlogpz   = self.prior.score(z)
        else:
            dzlogpz   = -z

        loss = ((dzlogpz_x - dzlogpx_z - dzlogpz)**2).sum(-1).mean()
        return loss
    
    def score_x(self, clean_x):
        
                
        if self.score_v == 1:
            loss = self._score_x_v1(clean_x)
            
        if self.score_v == 2:
            loss = self._score_x_v2(clean_x)
    
        if self.score_v == 3:
            loss = self._score_x_v3(clean_x)
                        
        return loss
        
    
    def _score_x_v1(self, clean_x):
        
        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        
        x = torch.repeat_interleave(x, M, dim=0)
        clean_x = torch.repeat_interleave(clean_x, M, dim=0)

        x.requires_grad=True

        # compute logps
        pz_x = self.encode(x)
        z = pz_x.sample([])
        px_z = self.decode(z)
        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        # compute scores
        dxlogpx_z = torch.autograd.grad((logpx_z-logpz_x).sum(), [x], retain_graph=True, create_graph=True)[0]
        loss = ( ( (dxlogpx_z) - (clean_x-x)/self.sm_noise_std**2 )**2).sum(-1).mean()
        return loss
        
        
    def _score_x_v2(self, clean_x):
        
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std

        x.requires_grad=True

        M = self.M
        # compute logps
        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)
        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        # compute scores
        dxlogpx_z = torch.autograd.grad((logpx_z-logpz_x).sum() / M, [x], retain_graph=True, create_graph=True)[0]
        loss = ( ( (dxlogpx_z) - (clean_x-x)/self.sm_noise_std**2 )**2).sum(-1).mean()

        return loss
        
    def _score_x_v3(self, clean_x):
        
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad=True

        # compute logps
        M = self.M
        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)
        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        # compute scores
        dxlogpx_z = torch.autograd.grad((logpx_z).sum() / M, [x], retain_graph=True, create_graph=True)[0]
        loss = ( ( (dxlogpx_z) - (clean_x-x)/self.sm_noise_std**2 )**2).sum(-1).mean()
        
        return loss


    def joint_score(self, clean_x, reparam=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x = torch.repeat_interleave(x, M, dim=0)
        x.requires_grad_()

        pz_x = self.encode(x)
        if reparam:
            z = pz_x.rsample([])
        else:
            z = pz_x.sample([])
            z.requires_grad_()
        px_z = self.decode(z)

        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        if isinstance(self.prior, EBM):
            dzlogpz   = self.prior.score(z)
        else:
            dzlogpz   = -z
        dzlogpz_x = - (z - pz_x.mean) / (pz_x.stddev**2)
        dxlogpz_x = torch.autograd.grad((logpz_x).sum(), [x], create_graph=True)[0]
        dzlogpx_z = torch.autograd.grad((logpx_z).sum(), [z], create_graph=True)[0]
        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                0.5 * (dzlogpz_x - dzlogpx_z - dzlogpz).square().sum(-1) + 
                (0.5 * dxlogpx_z.square() - (1. / px_z.stddev**2) + 0.5 * dxlogpz_x.square()).sum(-1)
               ).mean()

        return loss

    def joint_score_without_curve(self, clean_x, reparam=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x = torch.repeat_interleave(x, M, dim=0)
        x.requires_grad_()

        pz_x = self.encode(x)
        if reparam:
            z = pz_x.rsample([])
        else:
            z = pz_x.sample([])
            z.requires_grad_()
        px_z = self.decode(z)

        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        if isinstance(self.prior, EBM):
            dzlogpz   = self.prior.score(z)
        else:
            dzlogpz   = -z
        dzlogpz_x = - (z - pz_x.mean) / (pz_x.stddev**2)
        dxlogpz_x = torch.autograd.grad((logpz_x).sum(), [x], create_graph=True)[0]
        dzlogpx_z = torch.autograd.grad((logpx_z).sum(), [z], create_graph=True)[0]
        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                0.5 * (dzlogpz_x - dzlogpx_z - dzlogpz).square().sum(-1) + 
                (0.5 * dxlogpx_z.square() - (1. / px_z.stddev**2) ).sum(-1)
               ).mean()

        return loss

    def joint_score_x(self, clean_x):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x = torch.repeat_interleave(x, M, dim=0)
        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.sample([])
        px_z = self.decode(z)

        logpz_x = pz_x.log_prob(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                (0.5 * dxlogpx_z.square() - (1. / px_z.stddev**2)).sum(-1)
               ).mean()

        return loss

    def unbiased_marginal_score(self, clean_x, detach=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                - 0.5 * dxlogpx_z.mean(0).square().sum(-1)
                #+ dxlogpx_z.square().sum(-1).mean(0)
                + dxlogpx_z.var(0, unbiased=True).sum(-1) + dxlogpx_z.square().mean(0).sum(-1)
                - (1 / px_z.stddev**2).mean(0).sum(-1)
               ).mean()

        return loss

    def marginal_score(self, clean_x, detach=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                - 0.5 * dxlogpx_z.mean(0).square().sum(-1)
                + dxlogpx_z.square().sum(-1).mean(0)
                - (1 / px_z.stddev**2).mean(0).sum(-1)
               ).mean()

        return loss

    def bad_marginal_score(self, clean_x, detach=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.sample([M])
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

    def unbiased_bad_marginal_score(self, clean_x, detach=False):

        M = self.M
        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad_()

        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)

        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        dxlogpz_x = torch.autograd.grad((logpz_x).sum(), [x], create_graph=True)[0]
        dxlogpx_z = - (x - px_z.mean) / (px_z.stddev**2)

        loss = (
                0.5 * (dxlogpx_z.var(0, unbiased=True).sum(-1) + dxlogpx_z.square().mean(0).sum(-1))
                #0.5 * dxlogpx_z.mean(0).square().sum(-1)
                + (dxlogpx_z * dxlogpz_x).sum(-1).mean(0)
                - (1 / px_z.stddev**2).mean(0).sum(-1)
               ).mean()

        return loss
    def _score_x_v4(self, clean_x):

        x = clean_x + torch.randn_like(clean_x) * self.sm_noise_std
        x.requires_grad=True
        # compute logps
        M = self.M
        pz_x = self.encode(x)
        z = pz_x.sample([M])
        px_z = self.decode(z)
        logpz_x = pz_x.log_prob(z)
        logpx_z = px_z.log_prob(x)

        # compute scores
        dxlogpx_z = torch.autograd.grad((logpx_z).sum() / M, [x], retain_graph=True, create_graph=True)[0]
        loss = ( ( (dxlogpx_z) - (clean_x-x)/self.sm_noise_std**2 )**2).sum(-1).mean()
        

