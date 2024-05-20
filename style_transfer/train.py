import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    Encoder,
    Generator,
    Discriminator,
    ResidualBlock,
    weights_init_normal,
    LambdaLR,
)
from data_proc import DataProc

from tqdm import tqdm

import argparse
import os
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument(
    "--n_epochs", type=int, default=100, help="number of epochs of training"
)
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--n_spkrs", type=int, default=2, help="size of the batches")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--decay_epoch", type=int, default=50, help="epoch from which to start lr decay"
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--sample_interval",
    type=int,
    default=100,
    help="interval between saving generator samples",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=1,
    help="interval between saving model checkpoints",
)
parser.add_argument(
    "--n_downsample", type=int, default=2, help="number downsampling layers in encoder"
)
parser.add_argument(
    "--dim", type=int, default=32, help="number of filters in first encoder layer"
)

opt = parser.parse_args()
print(opt)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create sample and checkpoint directories
os.makedirs(f"saved_models/{opt.model_name}", exist_ok=True)

# the input dimensions of the melspectrograms
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2**opt.n_downsample

# Initialize generator and discriminator
encoder = Encoder(dim=opt.dim, in_channels=opt.channels, n_downsample=opt.n_downsample)
shared_G = ResidualBlock(features=shared_dim)
G1 = Generator(
    dim=opt.dim,
    out_channels=opt.channels,
    n_upsample=opt.n_downsample,
    shared_block=shared_G,
)
G2 = Generator(
    dim=opt.dim,
    out_channels=opt.channels,
    n_upsample=opt.n_downsample,
    shared_block=shared_G,
)
D1 = Discriminator(input_shape)
D2 = Discriminator(input_shape)


# load checkpointing
if opt.epoch != 0:
    # Load pretrained models
    checkpoint = torch.load(
        f"saved_models/{opt.model_name}/checkpoint_{opt.epoch}.pth",
        map_location=torch.device(device),
    )
    encoder.load_state_dict(checkpoint["E"])
    G1.load_state_dict(checkpoint["G1"])
    G2.load_state_dict(checkpoint["G2"])
    D1.load_state_dict(checkpoint["D1"])
    D2.load_state_dict(checkpoint["D2"])
    losses = checkpoint["losses"]
else:
    # Initialize weights
    encoder.apply(weights_init_normal)
    G1.apply(weights_init_normal)
    G2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)
    losses = {"G": [], "D": []}


encoder.to(device)
G1.to(device)
G2.to(device)
D1.to(device)
D2.to(device)


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), G1.parameters(), G2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

# Prepare dataloader
dataloader = torch.utils.data.DataLoader(
    DataProc(opt, split="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()

criterion_pixel = nn.L1Loss()


def criterion_reconst(reconstruction, original):
    return original.sub(reconstruction).flatten(start_dim=1).pow(2).sum(dim=1).mean()


def compute_kl(mu):
    return mu.flatten(start_dim=1).pow(2).sum(dim=1).mean()


# Loss weights
lambda_0 = 100  # GAN
lambda_1 = 0.1  # KL (encoded spect)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 0.1  # KL (encoded translated spect)
lambda_4 = 100  # Cycle pixel-wise
lambda_5 = 10  # latent space L1

encoder.train()
G1.train()
G2.train()
D1.train()
D2.train()

for epoch in range(opt.epoch, opt.n_epochs):
    epoch_losses = {"G": [], "D": []}
    progress = tqdm(enumerate(dataloader), desc="", total=len(dataloader))
    for i, batch in progress:
        # Set model input
        X1 = batch["A"].to(device)
        X2 = batch["B"].to(device)

        # Adversarial ground truths
        valid = torch.empty((X1.size(0), 1), requires_grad=False).fill_(1).to(device)
        fake = torch.empty((X1.size(0), 1), requires_grad=False).fill_(0).to(device)

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        mu1, Z1 = encoder(X1)
        mu2, Z2 = encoder(X2)

        # Latent space feat
        feat_1 = mu1.view(mu1.size()[0], -1).mean(dim=0)
        feat_2 = mu2.view(mu2.size()[0], -1).mean(dim=0)

        # Reconstruct speech
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)

        # Translate speech
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)

        # Cycle translation
        mu1_, Z1_ = encoder(fake_X1)
        mu2_, Z2_ = encoder(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)

        # Losses
        loss_GAN_1 = lambda_0 * criterion_GAN(D1(fake_X1), valid)
        loss_GAN_2 = lambda_0 * criterion_GAN(D2(fake_X2), valid)
        loss_KL_1 = lambda_1 * compute_kl(mu1)
        loss_KL_2 = lambda_1 * compute_kl(mu2)
        loss_ID_1 = lambda_2 * criterion_reconst(recon_X1, X1)
        loss_ID_2 = lambda_2 * criterion_reconst(recon_X2, X2)
        loss_KL_1_ = lambda_3 * compute_kl(mu1_)
        loss_KL_2_ = lambda_3 * compute_kl(mu2_)
        loss_cyc_1 = lambda_4 * criterion_reconst(cycle_X1, X1)
        loss_cyc_2 = lambda_4 * criterion_reconst(cycle_X2, X2)
        loss_feat = lambda_5 * criterion_pixel(feat_1, feat_2)

        # Total loss
        loss_G = (
            loss_KL_1
            + loss_KL_2
            + loss_ID_1
            + loss_ID_2
            + loss_GAN_1
            + loss_GAN_2
            + loss_KL_1_
            + loss_KL_2_
            + loss_cyc_1
            + loss_cyc_2
            + loss_feat
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(
            D1(fake_X1.detach()), fake
        )

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(
            D2(fake_X2.detach()), fake
        )

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        epoch_losses["G"].append(loss_G.detach().cpu().item())
        epoch_losses["D"].append((loss_D1 + loss_D2).detach().cpu().item())

        # update progress bar

        progress.set_description(
            f"[Epoch {epoch}/{opt.n_epochs}] [D loss: {np.mean(epoch_losses['D'])}] [G loss: {np.mean(epoch_losses['G'])}] "
        )

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    losses["G"].append(np.mean(epoch_losses["G"]))
    losses["D"].append(np.mean(epoch_losses["D"]))

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        checkpoint = {
            "E": encoder.state_dict(),
            "G1": G1.state_dict(),
            "G2": G2.state_dict(),
            "D1": D1.state_dict(),
            "D2": D2.state_dict(),
            "losses": losses,
        }
        torch.save(checkpoint, f"saved_models/{opt.model_name}/checkpoint_{epoch}.pth")
