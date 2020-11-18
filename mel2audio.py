import torch
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from args import parse_args
from hps import Hyperparameters
from model import SmartVocoder
import numpy as numpy
import soundfile as sf
import numpy as np
import os
import time

torch.backends.cudnn.benchmark = False

data_path = "mels_LJ"
sample_path = "outputs_LJ"
load_path = "pretrained/checkpoint.pth"
temp = 0.6
hop_size = 256
sr=22050

def build_model(hps):
    model = SmartVocoder(hps)
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def synthesize(model):
    for filename in os.listdir(data_path):
        mel = np.load(os.path.join(data_path, filename))
        mel = torch.tensor(mel).to(device).unsqueeze(0).permute(0, 2, 1)
        B, C, T = mel.shape
        print('Input mel shape:', mel.shape)
        z = torch.randn(1, 1, T*hop_size).to(device)
        z = z * temp
        timestemp = time.time()
        with torch.no_grad():
            y_gen = model.reverse(z, mel).squeeze()
        wav = y_gen.to(torch.device("cpu")).data.numpy()
        wav_name = '{}/{}.wav'.format(sample_path,  filename.split('.')[0])
        dur_wav = len(wav) / sr
        dur_synth = time.time() - timestemp
        print('(높을 수록 좋음) RTF: {:0.2f}, Speech 길이: {:0.2f}(s)'.format(dur_wav / dur_synth, dur_wav))
        sf.write(wav_name, wav, sr)
        print('{} Saved!'.format(wav_name))


if __name__ == "__main__":
    global global_step
    global start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    hps = Hyperparameters(args)
    model = build_model(hps)
        
    print("Load checkpoint from: {}".format(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    if not os.path.isdir(sample_path):
        os.makedirs(sample_path)
    print('sample_path', sample_path)
    with torch.no_grad():
        synthesize(model)
