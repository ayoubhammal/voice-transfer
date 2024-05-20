import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from utils import reconstruct_waveform, save_wav
from params import num_samples
from data_proc import InferenceDataProc
import os

import torch
import torch.nn as nn

from models import (
    Encoder,
    Generator,
    Discriminator,
    ResidualBlock,
    weights_init_normal,
    LambdaLR,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--epoch", type=int, help="epoch of the checkpoint")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--n_downsample", type=int, default=2, help="number downsampling layers in encoder"
)
parser.add_argument(
    "--dim", type=int, default=32, help="number of filters in first encoder layer"
)

opt = parser.parse_args()
print(opt)


device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(f"inference_output/{opt.model_name}/A2B", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/A", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/B2A", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/B", exist_ok=True)

os.makedirs(f"inference_output/{opt.model_name}/A2B_mel", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/A_mel", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/B2A_mel", exist_ok=True)
os.makedirs(f"inference_output/{opt.model_name}/B_mel", exist_ok=True)

# Prepare datasets

dataset_A = InferenceDataProc(opt, key="A")
dataset_B = InferenceDataProc(opt, key="B")

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


encoder.to(device)
G1.to(device)
G2.to(device)
D1.to(device)
D2.to(device)

encoder.eval()
G1.eval()
G2.eval()
D1.eval()
D2.eval()

with torch.no_grad():
    # inference on dataset A
    for x in tqdm(dataset_A, desc="transfer of A -> B "):
        wav_file_name = x["wav_file_name"]
        mel_spec = x["mel_spec"]

        batch = []
        for i in range(0, mel_spec.shape[1], num_samples):
            chunk = mel_spec[:, i : i + num_samples]
            if chunk.shape[1] < num_samples:
                chunk = np.pad(
                    chunk, pad_width=((0, 0), (0, num_samples - chunk.shape[1]))
                )
            batch.append(chunk)

        # n_chunks x 1 x 128 x 128
        batch_input = (
            torch.tensor(np.array(batch), requires_grad=False)
            .unsqueeze(dim=1)
            .to(device)
        )

        # transfer
        mu, z = encoder(batch_input)

        batch_output = G2(mu)
        A2B_mel_spec = (
            torch.cat(list(batch_output.squeeze(dim=1)), dim=1)
            .detach()
            .cpu()
            .numpy()[:, : mel_spec.shape[1]]
        )
        # vocoder
        A2B_waveform = reconstruct_waveform(A2B_mel_spec)
        save_wav(
            A2B_waveform,
            os.path.join(f"inference_output/{opt.model_name}/A2B", wav_file_name),
        )
        np.save(
            os.path.join(
                f"inference_output/{opt.model_name}/A2B_mel", f"{wav_file_name}.npy"
            ),
            A2B_mel_spec,
        )

        A_waveform = reconstruct_waveform(mel_spec)
        save_wav(
            A_waveform,
            os.path.join(f"inference_output/{opt.model_name}/A", wav_file_name),
        )
        np.save(
            os.path.join(
                f"inference_output/{opt.model_name}/A_mel", f"{wav_file_name}.npy"
            ),
            mel_spec,
        )

    # inference on dataset B
    for x in tqdm(dataset_B, desc="transfer of B -> A "):
        wav_file_name = x["wav_file_name"]
        mel_spec = x["mel_spec"]

        batch = []
        for i in range(0, mel_spec.shape[1], num_samples):
            chunk = mel_spec[:, i : i + num_samples]
            if chunk.shape[1] < num_samples:
                chunk = np.pad(
                    chunk, pad_width=((0, 0), (0, num_samples - chunk.shape[1]))
                )
            batch.append(chunk)

        # n_chunks x 1 x 128 x 128
        batch_input = (
            torch.tensor(np.array(batch), requires_grad=False)
            .unsqueeze(dim=1)
            .to(device)
        )

        # transfer
        mu, z = encoder(batch_input)
        batch_output = G1(mu)
        B2A_mel_spec = (
            torch.cat(list(batch_output.squeeze(dim=1)), dim=1)
            .detach()
            .cpu()
            .numpy()[:, : mel_spec.shape[1]]
        )

        # vocoder
        B2A_waveform = reconstruct_waveform(B2A_mel_spec)
        save_wav(
            B2A_waveform,
            os.path.join(f"inference_output/{opt.model_name}/B2A", wav_file_name),
        )
        np.save(
            os.path.join(
                f"inference_output/{opt.model_name}/B2A_mel", f"{wav_file_name}.npy"
            ),
            B2A_mel_spec,
        )

        B_waveform = reconstruct_waveform(mel_spec)
        save_wav(
            B_waveform,
            os.path.join(f"inference_output/{opt.model_name}/B", wav_file_name),
        )
        np.save(
            os.path.join(
                f"inference_output/{opt.model_name}/B_mel", f"{wav_file_name}.npy"
            ),
            mel_spec,
        )
