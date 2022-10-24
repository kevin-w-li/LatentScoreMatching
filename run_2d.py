import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import torch
import numpy as np
import torch.distributions as td
import torch.nn.functional as F

import argparse

def make_grid(low, high, N, device):
    z1, z2 = torch.meshgrid(torch.linspace(low,high,N), torch.linspace(low,high,N))
    zz = torch.stack([z1,z2], dim=-1).reshape(-1, 2).to(device)
    return z1, z2, zz


def likelihood(z, lik_name):
    if lik_name  == "sym":
        return td.Normal(z.prod(-1), 0.5)
    elif lik_name  == "bias":
        return td.Normal( (z[...,0] * 2 + 1) * z[...,1], 1.0)
    elif lik_name == "relu":
        return td.Normal( (torch.relu(z[...,0])) * z[...,1], 1.0)
    elif lik_name == "soft":
        def softplus(x, alpha):
            y = x.clone()
            p = alpha*x
            idx = torch.logical_and(p < 20, p > -20)
            y[idx] = torch.log(1+(p[idx]).exp())/alpha
            y[p>=20]=x[p>=20]
            y[p<=-20]=0
            return y
        return td.Normal( (softplus(z[...,0], 10)) * z[...,1], 1.0)


def normalized_density(p):
    p = p - p.max()
    p = p.exp()/p.exp().sum()
    return p

def fit_fisher(m, lik_fun, opt_name="adam", niter = 100000, nsample=100, reparam=False):
        
    prior = td.Normal(torch.zeros(2), torch.ones(2))
    m = torch.tensor(m, requires_grad=True)
    s = torch.zeros(2, requires_grad=True)

    if opt_name == "sgd":
        opt = torch.optim.SGD([m,s], lr=1e-4)
    elif opt_name == "adam":
        opt = torch.optim.Adam([m,s], lr=1e-4)

    ms = []
    ss = []
    losses = []
    for i in range(niter):
        q = td.Normal(m, s.exp())
        if reparam:
            z = q.rsample([nsample])
            z_d = z.detach()
            z_d.requires_grad_()
        else:
            z = q.sample([nsample])
            z.requires_grad_()

        dlogqz_x = - ( z - q.mean) / q.scale.square()
        dlogpz   = - (z - prior.mean) / prior.scale.square()
        if reparam == 2:
            logpx_z = lik_fun(z_d)
            dlogpx_z = torch.autograd.grad(logpx_z.sum(), [z_d], create_graph=True)[0]
        else:
            logpx_z = lik_fun(z)
            dlogpx_z = torch.autograd.grad(logpx_z.sum(), [z], create_graph=True)[0]

        loss = (dlogqz_x - dlogpx_z - dlogpz).square().sum(-1).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        ms += q.mean.detach().numpy().copy(),
        ss += q.scale.detach().numpy().copy(),
        losses += loss.item(),
    return ms, ss, losses, q

def fit_kl(m, lik_fun, opt_name="adam", niter = 100000, nsample=100, reparam=True, closed_kl=True):
        
    prior = td.Normal(torch.zeros(2), torch.ones(2))
    m = torch.tensor(m, requires_grad=True)
    s = torch.zeros(2, requires_grad=True)

    if opt_name == "sgd":
        opt = torch.optim.SGD([m,s], lr=1e-4)
    elif opt_name == "adam":
        opt = torch.optim.Adam([m,s], lr=1e-4)

    ms = []
    ss = []
    losses = []
    for i in range(niter):
        q = td.Normal(m, s.exp())
        if reparam:
            z = q.rsample([nsample])
        else:
            z = q.sample([nsample])
            z.requires_grad_()

        if closed_kl:
            loss = - lik_fun(z).mean() + td.kl.kl_divergence(q, prior).sum()
        else:
            loss = - lik_fun(z).mean()- prior.log_prob(z).sum(-1).mean(0) - q.entropy().sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        ms += q.mean.detach().numpy().copy(),
        ss += q.scale.detach().numpy().copy(),
        losses += loss.item(),
    return ms, ss, losses, q


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--x1", type=float, default=0.0)
    parser.add_argument("--x2", type=float, default=0.0)
    parser.add_argument("--reparam", type=int, default=0)
    parser.add_argument("--loss", type=str, default="fisher", choices=["fisher", "kl"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--closed_kl", type=int, default=0)
    parser.add_argument("--lik_name", type=str, default="sym", choices = ["sym", "relu", "soft", "bias"])
    parser.add_argument("--opt_name", type=str, default="adam", choices = ["sgd", "adam"])

    args = parser.parse_args()

    m0 = [args.x1, args.x2]

    device=torch.device("cpu")
    Ngrid = 100
    low = -3
    high = 3
    z1, z2, zz = make_grid(-low, high, Ngrid, device)

    x = torch.tensor([2.0])
    nsample = 1
    niter = 1000000

    filename = f"{args.lik_name}_{args.opt_name}_x{m0[0]:.1f}_{m0[1]:.1f}_r{int(args.reparam)}_l{args.loss[0]}_c{args.closed_kl}_s{args.seed}.npz"
    '''
    if os.path.exists(f"results/two_d/{filename}"):
        exit()
    '''
    lik_fun = lambda z: likelihood(z, args.lik_name).log_prob(x)

    if args.loss == "fisher":
        ms, ss, losses, q = fit_fisher(m0, lik_fun, args.opt_name, reparam=args.reparam, nsample=nsample, niter=niter)
    elif args.loss == "kl":
        ms, ss, losses, q = fit_kl(m0, lik_fun, args.opt_name, reparam=args.reparam, closed_kl = args.closed_kl, niter=niter, nsample=nsample)

    np.savez("results/two_d/" + filename, mu0=np.array(m0), losses = losses, ms = ms, ss = ss)
