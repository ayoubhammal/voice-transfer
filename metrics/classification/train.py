import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import argparse
import os

from tqdm import tqdm, trange

from model import (
    Wav2Vec2ForSpeechClassification,
    ClassificationDataset,
    create_collate_fn,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--n_epochs", type=int, default=2, help="number of epochs of training"
)

opt = parser.parse_args()
print(opt)

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(f"saved_models/{opt.model_name}", exist_ok=True)

# initialize the feature extractor of the wav2vec2-base model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# initialize the dataloader
dataloader = torch.utils.data.DataLoader(
    ClassificationDataset(opt.dataset, feature_extractor, sample_rate=16_000),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    collate_fn=create_collate_fn(feature_extractor),
)

# initialize the wav2vec2-base model from the huggingface checkpoint
wav2vec2model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# create the classification model
model = Wav2Vec2ForSpeechClassification(wav2vec2model)
model.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# loss function
loss_fn = nn.BCEWithLogitsLoss()

losses = []
for epoch in range(opt.n_epochs):
    # training epoch
    model.train()
    epoch_losses = []
    progress = tqdm(dataloader, desc="")
    for batch in progress:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)

        optimizer.zero_grad()
        # forward pass
        logits, embeddings = model(x)

        # loss back propagation
        loss = loss_fn(logits, y)
        loss.backward()

        # optimizer step
        optimizer.step()

        # logging
        epoch_losses.append(loss.detach().cpu().item())

        progress.set_description(
            f"[train epoch {epoch}/{opt.n_epochs}] [loss {np.mean(epoch_losses)}] "
        )

    losses.append(np.mean(epoch_losses))

    # evaluation epoch
    with torch.no_grad():
        model.eval()
        n_correct = 0
        total = 0
        progress = tqdm(dataloader, desc="")
        for batch in progress:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)

            logits, embeddings = model(x)

            n_correct += torch.sum(y == (logits >= 0)).detach().cpu().item()
            total += y.size(0)

            progress.set_description(
                f"[eval epoch {epoch}/{opt.n_epochs}] [accuracy {n_correct / total}] "
            )

# checkpointing the trained classification model
checkpoint = {
    "state_dict": model.state_dict(),
    "losses": losses,
}
torch.save(checkpoint, f"saved_models/{opt.model_name}/checkpoint_wav2vec2.pth")
