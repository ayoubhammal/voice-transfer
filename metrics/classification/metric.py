import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import argparse
import os
import pickle

from tqdm import tqdm, trange

from model import (
    Wav2Vec2ForSpeechClassification,
    ClassificationDataset,
    EvalClassificationDataset,
    create_collate_fn,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--eval_dataset", type=str, help="path to generated dataset")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)

opt = parser.parse_args()
print(opt)

device = "cuda" if torch.cuda.is_available() else "cpu"


os.makedirs(f"metric_output/{opt.model_name}", exist_ok=True)

# initialize the feature extractor of the wav2vec2-base model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# the original data dataloader
dataloader = torch.utils.data.DataLoader(
    ClassificationDataset(opt.dataset, feature_extractor, sample_rate=16_000),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    collate_fn=create_collate_fn(feature_extractor),
)

# the generated data dataloader
eval_dataloader = torch.utils.data.DataLoader(
    EvalClassificationDataset(opt.eval_dataset, feature_extractor, sample_rate=16_000),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    collate_fn=create_collate_fn(feature_extractor),
)

# initialize the wav2vec2-base model from the huggingface checkpoint
wav2vec2model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# create the classification model
model = Wav2Vec2ForSpeechClassification(wav2vec2model)
model.to(device)

# load the model's checkoint
checkpoint = torch.load(
    f"saved_models/{opt.model_name}/checkpoint_wav2vec2.pth",
    map_location=torch.device(device),
)
model.load_state_dict(checkpoint["state_dict"])

# save the embeddings on all the audio files for visualization later
embeddings = {"A": [], "B": [], "B2A": [], "A2B": []}

# evaluation on A and B
with torch.no_grad():
    model.eval()
    n_correct = 0
    total = 0
    progress = tqdm(dataloader, desc="")
    for batch in progress:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)

        logits, hidden = model(x)

        # save embeddings
        for i in range(y.size(0)):
            if y[i] == 0:
                embeddings["A"].append(hidden[i].detach().cpu().numpy())
            else:
                embeddings["B"].append(hidden[i].detach().cpu().numpy())

        # track accuracy
        n_correct += torch.sum(y == (logits >= 0)).detach().cpu().item()
        total += y.size(0)

        progress.set_description(f"[original] [accuracy {n_correct / total}] ")

# evaluation on the A2B end B2A
with torch.no_grad():
    model.eval()
    n_correct = 0
    total = 0
    progress = tqdm(eval_dataloader, desc="")
    for batch in progress:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)

        logits, hidden = model(x)

        # save embeddings
        for i in range(y.size(0)):
            if y[i] == 0:
                embeddings["B2A"].append(hidden[i].detach().cpu().numpy())
            else:
                embeddings["A2B"].append(hidden[i].detach().cpu().numpy())

        # track accuracy
        n_correct += torch.sum(y == (logits >= 0)).detach().cpu().item()
        total += y.size(0)

        progress.set_description(f"[eval] [accuracy {n_correct / total}] ")

accuracy = n_correct / total


# log accuracy
print(f"accuracy on the generated dataset: {accuracy}")
with open(
    os.path.join(f"metric_output/{opt.model_name}", "classification_accuracy.txt"), "w"
) as accuracy_file:
    accuracy_file.write(f"accuracy on the generated dataset: {accuracy}")

# saving embeddings
pickle.dump(
    embeddings,
    open(os.path.join(f"metric_output/{opt.model_name}", "embeddings.pkl"), "wb"),
)
