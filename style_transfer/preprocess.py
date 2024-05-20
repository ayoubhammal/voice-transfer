import argparse
import pickle
from tqdm import tqdm
from utils import ls, preprocess_wav, melspectrogram
from params import num_samples
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--n_spkrs", type=int, default=2, help="size of the batches")
parser.add_argument("--test_ratio", type=float, default=0.2, help="test ratio")
parser.add_argument("--A", type=str, default="A", help="first strata")
parser.add_argument("--B", type=str, default="B", help="second strata")


opt = parser.parse_args()
print(opt)

feats_train = {}
feats_test = {}

for id_strata, strata in [("A", opt.A), ("B", opt.B)]:
    # load all wav paths of each category A and B
    wavs = ls(f"{os.path.join(opt.dataset, strata)} | grep .wav")

    # split into train and test sets
    test_size = int(len(wavs) * opt.test_ratio)
    train_size = len(wavs) - test_size

    train_wavs = wavs[:train_size]
    test_wavs = wavs[train_size:]

    # preprocess each wav audio of each set
    for feats, wavs in [(feats_train, train_wavs), (feats_test, test_wavs)]:
        feats[id_strata] = []
        for i, wav in tqdm(enumerate(wavs), total=len(wavs), desc=f"{strata}"):
            # preprocessing: resampling, normalization and silence removal
            sample = preprocess_wav(f"{opt.dataset}/{strata}/{wav}")
            spect = melspectrogram(sample)
            if spect.shape[1] >= num_samples:
                feats[id_strata].append(
                    {
                        "wav_file_name": wav,
                        "mel_spec": spect,
                    }
                )

# save into pickle files
pickle.dump(feats_train, open(f"{opt.dataset}/{opt.model_name}_train.pickle", "wb"))
pickle.dump(feats_test, open(f"{opt.dataset}/{opt.model_name}_test.pickle", "wb"))
