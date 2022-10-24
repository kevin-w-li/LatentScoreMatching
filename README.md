# Decoder-based latent variable models through thelens of Fisher divergence

Here is a subset of the code for the paper.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
To use the notebook, you should also install the jupyter notebook kernels.

## Training

We provide training code for the following experiments:
1. Inference over 2d latent space with Fisher divergence and KL divergence: `run_2d.py`, `two_d.ipynb`
2. Training VAE model with objectives proposed in the paper: `run_ae.py`, `denosing_vae.ipynb`
3. Bilevel optimization for VAE models: `run_bilevel_toy.py`

All the scripts above should run without any arguments. See the argument list for specific settings.

