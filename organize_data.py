import os
import pandas as pd

dataset_path = "./"
organized_path = dataset_path + "organized/"
os.mkdir(organized_path)

df = pd.read_csv(
    dataset_path + "wav2spk.txt", sep=" ", header=None, names=["wav_file", "speaker_id"]
)

for id_speaker in df["speaker_id"].unique():
    os.mkdir(f"{organized_path}spkr_{id_speaker}")
    source_files = df[df["speaker_id"] == id_speaker]["wav_file"]
    for source_file in source_files:
        cp_command = f"cp {dataset_path}wavs/{source_file} {organized_path}spkr_{id_speaker}/{source_file}"
        os.popen(cp_command)
