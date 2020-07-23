from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa
from multiprocessing import cpu_count
import argparse
import re


def build_from_path(in_dir, out_dir, csv_path, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(csv_path), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            last_idx = line.find('|')
            line = line[:last_idx]
            parts = line.split('_')
            wav_path = os.path.join(in_dir, parts[0], parts[1], 'wav', '%s.wav' % parts[2])
            save_dir = os.path.join(parts[0], parts[1])
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, save_dir, index, wav_path)))
            index += 1
    return [future.result() for future in futures]

def _process_utterance(out_dir, save_dir, index, wav_path):
    # Load the audio to a numpy array:
    sr = 24000
    fft_size = 2048
    hop_length = 300
    win_length = 1200
    window = 'hann'
    num_mels = 80
    fmin = 80
    fmax = 7600
    eps = 1e-10

    wav, _ = librosa.load(wav_path, sr=sr)
    wav_trim, _ = librosa.effects.trim(wav, top_db=25)
    out = wav_trim
    x_stft = librosa.stft(out, n_fft=fft_size, hop_length=hop_length,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    mel_basis = librosa.filters.mel(sr, fft_size, num_mels, fmin, fmax)

    mel_spectrogram = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    pad = (out.shape[0] // hop_length + 1) * hop_length - out.shape[0]
    pad_l = pad // 2
    pad_r = pad // 2 + pad % 2

    # zero pad for quantized signal
    out = np.pad(out, (pad_l, pad_r), mode="constant", constant_values=0)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * hop_length

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * hop_length]
    assert len(out) % hop_length == 0

    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_path= os.path.join(save_dir, 'audio-%05d.npy' % index)
    mel_path = os.path.join(save_dir, 'mel-%05d.npy' % index)

    if not os.path.isdir(os.path.join(out_dir, save_dir)):
        os.makedirs(os.path.join(out_dir, save_dir))

    np.save(os.path.join(out_dir, audio_path),
            out.astype(np.float32), allow_pickle=False)
    np.save(os.path.join(out_dir, mel_path),
            mel_spectrogram.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return audio_path, mel_path, timesteps


def preprocess(in_dir, out_dir, csv_path, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, csv_path, num_workers)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = 24000
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))


if __name__ == "__main__":
    print('<--- Preprocess start --->')
    parser = argparse.ArgumentParser(description='Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', '-i', type=str, default='datasets', help='In Directory')
    parser.add_argument('--csv_path', '-ci', type=str, default='datasets/metadata_jka.csv', help='Metadata path')
    parser.add_argument('--out_dir', '-o', type=str, default='datasets/preprocessed', help='Out Directory')
    args = parser.parse_args()

    num_workers = cpu_count()
    preprocess(args.in_dir, args.out_dir, args.csv_path, num_workers)
    print('<--- Preprocess done --->')