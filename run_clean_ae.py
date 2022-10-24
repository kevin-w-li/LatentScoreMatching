import numpy as np
import torch, os
from torch.utils.data import TensorDataset, DataLoader
import tqdm as tqdm
from models.VAEs import AutoEncoder
import torchvision
from torchvision import datasets
from torchvision import transforms
import argparse

device=torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=100, help='batch size')
parser.add_argument("--seed", type=int, default=0, help='random seed')
parser.add_argument("--dataset", type=str, default="mnist", choices=['mnist', 'fashion', 'celeb'])
parser.add_argument("--sm_noise_std", type=float, default=1e-1, help='noise added to the data')
parser.add_argument("--n_epoch", type=int, default=1000, 'number of training epochs')
parser.add_argument("--objective", type=str, default="f", choices=["jf", "kl", "jf-mf", "kl-mf", "kl-xf", "jf-xf", 'jf-bmf', 'kl-bmf'])
parser.add_argument("--layer_type", type=str, default="conv", choices=["conv", "resnet"])
parser.add_argument("--M", type=int, default=1)
parser.add_argument("--K", type=int, default=1)
parser.add_argument("--nl", type=str, default="relu", choices=["elu", "relu"])
args = parser.parse_args()


n_epoch = args.n_epoch
bs = args.bs
dataset = args.dataset
sm_noise_std = args.sm_noise_std
objective = args.objective
seed = args.seed
layer_type = args.layer_type
K = args.K
lr = 1e-4
M = args.M
nl = args.nl
prior = 'gaussian' if seed < 10 else 'ebm'

filename = f"{dataset}_smn{sm_noise_std}_sv1_dae0_obj{objective}_lt{layer_type}_M{M}_K{K}_nl{nl[0]}_s{seed:02d}"

continue_run=False
if os.path.exists(f"results/clean_ae/ckpts/{filename}"):
    if os.path.exists(f"results/clean_ae/summary/{filename}"):
        summary = np.loadtxt(f"results/clean_ae/summary/{filename}", delimiter=',')
        summary = np.atleast_2d(summary)
        if summary.shape[0] >= (n_epoch//10):
            exit()
        elif summary.shape[0] < 2:
            os.remove(f"results/clean_ae/summary/{filename}")
        elif summary.shape[0] < n_epoch // 10:
            print("continue")
            continue_run = True
            model_dict, enc_opt_dict, dec_opt_dict, q_hists = torch.load(f"results/clean_ae/ckpts/{filename}")



        
print(filename)

torch.random.manual_seed(seed)

if args.dataset == "mnist":
    nc = 1
    pad = 2
    im_size = 32
    dataset = datasets.MNIST(root='../data', train=True, transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Pad(pad), 
                                                                    transforms.ToTensor(),]), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Pad(pad), 
                                                                    transforms.ToTensor(),]), download=True)
if args.dataset == "fashion":
    nc = 1
    pad = 2
    im_size = 32
    dataset = datasets.FashionMNIST(root='../data', train=True, transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Pad(pad), 
                                                                    transforms.ToTensor(),]), download=True)

    test_dataset = datasets.FashionMNIST(root='../data', train=False, transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Pad(pad), 
                                                                    transforms.ToTensor(),]), download=True)
elif args.dataset == "celeb":
        
    nc = 3
    pad = 0
    im_size = 64
    data = torch.as_tensor(np.load("../common_data/celeba64.npz")["images"]) / 255.0
    dataset = TensorDataset(data[:100000], torch.zeros(100000))
    test_dataset = TensorDataset(data[-10000:], torch.zeros(10000))


Dx = im_size*im_size*nc
Dz = 100
Dh = 512

data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=1, drop_last=True)


# score matching model
model = AutoEncoder(Dz, Dh, Dx, nc = nc, M = M, sm_noise_std = sm_noise_std,
        device=device, layer_type = layer_type, im_size = im_size).to(device=device)


enc_opt  = torch.optim.Adam(model.encoder.parameters(), lr=lr)
dec_opt  = torch.optim.Adam(list(model.decoder.parameters())+[model.logsigma], lr = lr)

start = 0

for ei in tqdm.tqdm(range(start, n_epoch), ncols=100):

    model.train()

    for idx, (images, _) in enumerate(data_loader):
        if images.shape[0] != bs:
            continue

        learning_step = ((idx+1) % K == 0)

        clean_x = images.view(bs,-1).to(device)


        if objective in ["kl", "jf"]:

            if objective == "kl":
                loss = model.ELBO(clean_x)
            else:
                loss = model.joint_score(clean_x, reparam=True)

            enc_opt.zero_grad()
            loss.backward(retain_graph=False)
            enc_opt.step()

            if learning_step:
                if objective == "kl":
                    loss = model.ELBO(clean_x)
                else:
                    loss = model.joint_score(clean_x)
                dec_opt.zero_grad()
                loss.backward(retain_graph=False)
                dec_opt.step()

        else:

            i_obj, l_obj = objective.split('-')

            if i_obj == "kl":
                loss_z = model.ELBO(clean_x)
            elif i_obj == "jf":
                loss_z = model.score_z(clean_x)
            enc_opt.zero_grad()
            loss_z.backward()
            enc_opt.step()

            if learning_step:
                if l_obj == "jf":
                    loss_x = model.joint_score(clean_x)
                if l_obj == "jfwc":
                    loss_x = model.joint_score_without_curve(clean_x)
                if l_obj == "xf":
                    loss_x = model.joint_score_x(clean_x)
                elif l_obj == "mf":
                    loss_x = model.marginal_score(clean_x)
                elif l_obj == "umf":
                    loss_x = model.unbiased_marginal_score(clean_x)
                elif l_obj == "kl":
                    loss_x = model.ELBO(clean_x)
                elif l_obj == "ubmf":
                    loss_x = model.unbiased_bad_marginal_score(clean_x)
                elif l_obj == "bmf":
                    loss_x = model.bad_marginal_score(clean_x)

                dec_opt.zero_grad()
                loss_x.backward()
                dec_opt.step()


    if (ei+1) % 10 == 0:
        model.eval()
        model.evaluate(test_data_loader)

        torch.save([model.state_dict(), enc_opt.state_dict(), dec_opt.state_dict(),
            model.q_hists
            ], f"results/clean_ae/ckpts/{filename}")
        with open(f"results/clean_ae/summary/{filename}", 'a') as f:
            line = f"{model.recon_losses[-1]},{model.MMDs[-1]},{model.ELBO_losses[-1]},{model.enc_losses[-1]},{model.dec_losses[-1]},{model.joint_fds[-1]}\n"
            f.write(line)
