import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from scipy.io import wavfile

sample_rate = 16_000


class Wav2Vec2ForSpeechClassification(nn.Module):
    def __init__(self, wav2vec2model):
        super(Wav2Vec2ForSpeechClassification, self).__init__()
        self.num_labels = 1

        # wav2vec2 model
        self.wav2vec2 = wav2vec2model
        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),
        )

        # freezing the feature extractor part of the wav2vec2 model
        self.freeze_feature_extractor()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
        self,
        hidden_states,
    ):
        # use mean aggregation over the hidden states
        return torch.mean(hidden_states, dim=1)

    def forward(
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        # aggregation over all the hidden states
        hidden_states = self.merged_strategy(hidden_states)
        logits = self.classifier(hidden_states).squeeze(dim=1)

        return logits, hidden_states


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Load the audio into two classes A=0 and B=1 for classification
    """

    def __init__(self, dataset_path, feature_extractor, sample_rate):
        classes = ["A", "B"]
        self.file_paths = {cl: [] for cl in classes}
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

        for cl in classes:
            cl_path = os.path.join(dataset_path, cl)
            for wav_file_name in os.listdir(cl_path):
                file_path = os.path.join(cl_path, wav_file_name)
                self.file_paths[cl].append(file_path)

    def __len__(self):
        return sum([len(paths) for paths in self.file_paths.values()])

    def __getitem__(self, index):
        if index < len(self.file_paths["A"]):
            label = 0
            path = self.file_paths["A"][index]
        else:
            label = 1
            path = self.file_paths["B"][index - len(self.file_paths["A"])]

        _, data = wavfile.read(path)

        feature = self.feature_extractor(data, sampling_rate=self.sample_rate)

        result = {}
        result["input_values"] = feature.input_values[0]
        result["labels"] = label

        return result


class EvalClassificationDataset(torch.utils.data.Dataset):
    """
    Load the audio into two classes B2A=0 and A2B=1 for classification evaluation.
    """

    def __init__(self, dataset_path, feature_extractor, sample_rate):
        classes = ["B2A", "A2B"]
        self.file_paths = {cl: [] for cl in classes}
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

        for cl in classes:
            cl_path = os.path.join(dataset_path, cl)
            for wav_file_name in os.listdir(cl_path):
                file_path = os.path.join(cl_path, wav_file_name)
                self.file_paths[cl].append(file_path)

    def __len__(self):
        return sum([len(paths) for paths in self.file_paths.values()])

    def __getitem__(self, index):
        if index < len(self.file_paths["B2A"]):
            label = 0
            path = self.file_paths["B2A"][index]
        else:
            label = 1
            path = self.file_paths["A2B"][index - len(self.file_paths["B2A"])]

        _, data = wavfile.read(path)

        feature = self.feature_extractor(data, sampling_rate=self.sample_rate)

        result = {}
        result["input_values"] = feature.input_values[0]
        result["labels"] = label

        return result


def create_collate_fn(feature_extractor):
    """creates a collate function using the feature extractor padding method."""

    def collate_fn(features):
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [feature["labels"] for feature in features]

        batch = feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=torch.float)

        return batch

    return collate_fn
