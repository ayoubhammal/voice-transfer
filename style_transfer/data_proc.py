import torch
import numpy as np
import pickle
import random
import librosa
import os

from params import num_samples


class DataProc(torch.utils.data.Dataset):
    def __init__(self, args, split: str):
        self.args = args
        # load the selected split pickle file
        if split == "train" or split == "test":
            self.data_dict = pickle.load(
                open(
                    os.path.join(args.dataset, f"{args.model_name}_{split}.pickle"),
                    "rb",
                )
            )

    def __len__(self):
        """
        returns the maximum number of 128 frame chunks available.
        """
        total_len = 0
        for i in self.data_dict.keys():
            tmp = np.sum([j["mel_spec"].shape[1] for j in self.data_dict[i]])
            total_len = max(total_len, tmp / 128)
        return int(total_len)

    def __getitem__(self, item):
        rslt = {}
        for i in self.data_dict.keys():
            # chose random item based on prop distribution (lenght of each sample)
            tmp_lens = [j["mel_spec"].shape[1] for j in self.data_dict[i]]
            item = np.random.choice(len(tmp_lens), p=tmp_lens / np.sum(tmp_lens))
            rslt[i] = self.random_sample(i, item)

        return rslt

    def random_sample(self, i, item):
        """sample a random 128 frame chunk from the item."""
        n_samples = num_samples
        data = self.data_dict[i][item]["mel_spec"]

        assert data.shape[1] >= n_samples

        rand_i = random.randint(0, data.shape[1] - n_samples)
        data = data[:, rand_i : rand_i + n_samples]
        return np.array([data])


class InferenceDataProc(torch.utils.data.Dataset):
    def __init__(self, args, key):
        self.args = args
        split = "test"
        self.key = key
        self.data_dict = pickle.load(
            open(os.path.join(args.dataset, f"{args.model_name}_{split}.pickle"), "rb")
        )

    def __len__(self):
        return len(self.data_dict[self.key])

    def __getitem__(self, item):
        return self.data_dict[self.key][item]
