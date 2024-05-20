import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer

from scipy.io import wavfile

import argparse
import os
import pickle

from tqdm import tqdm, trange


def speech_to_text(processor, model, forced_decoder_ids, audio, sampling_rate, device):
    # preprocess the audio
    input_features = processor(
        audio, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(device)

    # predict the output ids
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )

    # decode the ids into text
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, normalize=True
    )

    return transcription


def evaluate_wer(
    reference: list,
    generated: list,
    sampling_rate: int,
    language: str,
    batch_size: int,
    device: str,
):
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(
        device
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    reference_transcription = []
    generated_transcription = []

    word_error_rates = []
    # for each batch
    for i in trange(0, len(reference), batch_size):
        batch_reference = reference[i : i + batch_size]
        batch_generated = generated[i : i + batch_size]

        # transcribe the audio
        reference_transcription += speech_to_text(
            processor, model, forced_decoder_ids, batch_reference, sampling_rate, device
        )
        generated_transcription += speech_to_text(
            processor, model, forced_decoder_ids, batch_generated, sampling_rate, device
        )

        # calculate the word error rate
        for ref, gen in zip(reference_transcription, generated_transcription):
            word_error_rates.append(jiwer.wer(ref, gen))

    return word_error_rates


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="name of the model")
parser.add_argument("--dataset", type=str, help="path to dataset")
parser.add_argument("--eval_dataset", type=str, help="path to generated dataset")
parser.add_argument("--batch_size", type=int, default=4, help="batch size")

opt = parser.parse_args()
print(opt)

os.makedirs(f"metric_output/{opt.model_name}", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

referenceA = []
generatedA2B = []

# laod the audio of A and A2B
for filename in os.listdir(os.path.join(opt.eval_dataset, "A2B")):
    generated_path = os.path.join(opt.eval_dataset, "A2B", filename)
    reference_path = os.path.join(opt.dataset, "A", filename)

    _, reference_data = wavfile.read(reference_path)
    _, generated_data = wavfile.read(generated_path)

    referenceA.append(reference_data)
    generatedA2B.append(generated_data)

# calculate the word error rate
wer_A = evaluate_wer(
    referenceA, generatedA2B, 16_000, "english", opt.batch_size, device
)

referenceB = []
generatedB2A = []

# laod the audio of B and B2A
for filename in os.listdir(os.path.join(opt.eval_dataset, "B2A")):
    generated_path = os.path.join(opt.eval_dataset, "B2A", filename)
    reference_path = os.path.join(opt.dataset, "B", filename)

    _, reference_data = wavfile.read(reference_path)
    _, generated_data = wavfile.read(generated_path)

    referenceB.append(reference_data)
    generatedB2A.append(generated_data)

# calculate the word error rate
wer_B = evaluate_wer(
    referenceB, generatedB2A, 16_000, "english", opt.batch_size, device
)


# log the word error rate
print(
    f"wer A: {np.mean(wer_A)}±{np.std(wer_A)}, wer B: {np.mean(wer_B)}±{np.std(wer_B)}"
)
with open(os.path.join(f"metric_output/{opt.model_name}", "wer.txt"), "w") as wer_file:
    wer_file.write(
        f"wer A: {np.mean(wer_A)}±{np.std(wer_A)}, wer B: {np.mean(wer_B)}±{np.std(wer_B)}"
    )

pickle.dump(
    {"A": wer_A, "B": wer_B},
    open(os.path.join(f"metric_output/{opt.model_name}", "wers.pkl"), "wb"),
)
