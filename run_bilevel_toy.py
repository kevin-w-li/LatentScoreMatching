import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm as tqdm
from models.bilevels import BilevelAutoEncoder
from torch.utils.data import TensorDataset
import argparse

device=torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="sparse", choices=['sparse','mixture'])
parser.add_argument("--sm_noise_std", type=float, default=0.0)
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--reparam", type=int, default=0)
parser.add_argument("--score_v", type=int, default=1, choices=[1])
parser.add_argument("--objective", type=str, default="kl-xf", choices=["kl", "jf-mf", "kl-mf", "kl-xf", "jf-xf", 'jf-bmf', 'kl-bmf', 'kl-kl'])
parser.add_argument("--layer_type", type=str, default="fc")
parser.add_argument("--M", type=int, default=1)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--nl", type=str, default="relu", choices=["elu", "relu"])
parser.add_argument("--erl", type=int, default=-2)
parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'])
args = parser.parse_args()


n_epoch = args.n_epoch
bs = args.bs
dataset = args.dataset
sm_noise_std = args.sm_noise_std
reparam = args.reparam
objective = args.objective
score_v = args.score_v
seed = args.seed
layer_type = args.layer_type
K = args.K
lr = 1e-4
M = args.M
nl = args.nl
prior = 'gaussian' if seed < 10 else 'ebm'
bilevel = (objective!= 'kl')
mode = args.mode

filename = f"{dataset}_smn{sm_noise_std}_sv{score_v}_dae{reparam}_obj{objective}_lt{layer_type}_M{M}_K{K}_nl{nl[0]}_s{seed:02d}"
summary_fn = f"results/bilevel/summary/{filename}"
ckpt_fn = f"results/bilevel/ckpts/{filename}"
if os.path.exists(summary_fn) and mode == 'train':
    print("done")
    exit()

print(filename)


class SparseData(object):
        
    def __init__(self):
        self.A = torch.tensor([[1.0,0.3], [-0.3,1.0]]) / 10
        self.p = 2
    
    def sample(self, n):
        z = torch.randn(n, 2)
        z = torch.sign(z) * torch.abs(z) ** self.p
        x = z @ self.A
        x = x + torch.randn_like(x) * 0.01
        return x
    
    
class Mixture(object):
        
    def sample(self, n):
        x1 = torch.randn(n//2, 2) * 0.2-0.5
        x2 = torch.randn(n//2, 2) * 0.2+0.5        
        x = torch.cat([x1, x2])
        return x
    
torch.random.manual_seed(seed)

if args.dataset == "sparse":
    data_dist = SparseData()
    Dh = 100
elif args.dataset == "mixture":
    data_dist = Mixture()
    Dh = 30

test_data = data_dist.sample(5000)

Dx = 2
Dz = 2

# score matching model
if bilevel:
    i_obj, l_obj = objective.split('-')
    if i_obj == 'jf':
        update_step_size = 1e-3
    elif i_obj == 'kl':
        update_step_size = 1e-2
    else:
        raise NameError("i_obj incorrect")
else:
    update_step_size = 0.0

model = BilevelAutoEncoder(Dz, Dh, Dx, 0.1, update_step_size=update_step_size, bilevel = bilevel).to(device=device)

dec_opt  = torch.optim.Adam(model.parameters(), lr = lr)

invalid = False

# with torch.autograd.detect_anomaly():
if mode == 'test':
    n_epoch = 0
    if os.path.exists(ckpt_fn):
        ckpt = torch.load(ckpt_fn)[0]
        model.load_state_dict(ckpt)
    else:
        exit()

for ei in tqdm.tqdm(range(n_epoch), ncols=100):

    dataset = TensorDataset(data_dist.sample(100000), torch.ones(100000))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)

    model.train()

    #for idx, (images, _) in enumerate(tqdm.tqdm(data_loader, leave=False, ncols=100)):
    for idx, (images, _) in enumerate(data_loader):
        if images.shape[0] != bs:
            continue

        clean_x = images.view(bs,-1).to(device)

        model.zero_encoder_grad()

        if bilevel:

            for _ in range(M):
                if i_obj == "kl":
                    loss_z = model.kl_encoder(clean_x)
                elif i_obj == "jf":
                    loss_z = model.f_encoder(clean_x, reparam=bool(reparam))

            for _ in range(K):
                if i_obj == "kl":
                    loss_z = model.kl_encoder(clean_x, bilevel=True)
                elif i_obj == "jf":
                    loss_z = model.f_encoder(clean_x, reparam=bool(reparam), bilevel=True)


            if l_obj == "xf":
                loss_x = model.joint_score_x(clean_x)
            elif l_obj == "mf":
                loss_x = model.marginal_score(clean_x)
            elif l_obj == "bmf":
                loss_x = model.bad_marginal_score(clean_x)
            elif l_obj == "kl":
                loss_x = -model.ELBO(clean_x)

        else:
                loss_x = -model.ELBO(clean_x)

        dec_opt.zero_grad()
        loss_x.backward()
        if not torch.isfinite(loss_x):
            invalid = True
            break
        dec_opt.step()

    if ei % 10 == 9:
        torch.save([model.state_dict()], ckpt_fn)

model.eval()
if mode == 'test' and not invalid:
    loglik = model.log_likelihood(test_data).mean()
    fd = model.FD(test_data).mean()
else:
    loglik = torch.as_tensor(np.nan)
    fd = torch.as_tensor(np.nan)
print(loglik.item(), fd.item())

if mode == 'train':
    torch.save([model.state_dict()], ckpt_fn)
with open(f"results/bilevel/summary/{filename}", 'w') as f:
    line = str(loglik.item()) + ',' + str(fd.item()) + '\n'
    f.write(line)
