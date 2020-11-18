import torch
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from args import parse_args
from data import KORDataset, collate_fn_synth
from hps import Hyperparameters
from model import SmartVocoder
from utils import mkdir
import librosa
import os
import time

torch.backends.cudnn.benchmark = False


def load_dataset(args):
    collate_fn2 = lambda batch: collate_fn_synth(batch, args.hop_length)
    test_dataset = KORDataset(args.data_path, False, 0.1)
    synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn2,
                            num_workers=args.num_workers, pin_memory=True)
    print('sr', args.sr)
    return synth_loader


def build_model(hps):
    model = SmartVocoder(hps)
    print('number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    return model


def synthesize(model, temp, num_synth):
    global global_step
    print('temp', temp)
    for batch_idx, (x, c) in enumerate(synth_loader):
        if batch_idx < num_synth:
            x, c = x.to(device), c.to(device)
            q_0 = Normal(x.new_zeros(x.size()), x.new_ones(x.size()))
            z = q_0.sample()
            z = z * temp
            torch.cuda.synchronize()
            timestemp = time.time()
            with torch.no_grad():
                y_gen = model.reverse(z, c).squeeze()

            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}.wav'.format(sample_path, batch_idx)
            torch.cuda.synchronize()
            print('{} seconds'.format(time.time() - timestemp))
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))


def load_checkpoint(step, model):
    checkpoint_path = os.path.join(save_path, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)

    # generalized load procedure for both single-gpu and DataParallel models
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    g_epoch = checkpoint["global_epoch"]
    g_step = checkpoint["global_step"]

    return model, g_epoch, g_step


if __name__ == "__main__":
    global global_step
    global start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    sample_path, save_path = mkdir(args, synthesize=True)
    synth_loader = load_dataset(args)
    hps = Hyperparameters(args)
    model = build_model(hps)
    model, global_epoch, global_step = load_checkpoint(args.load_step, model)
    # model = WaveNODE.remove_weightnorm(model)
    model.to(device)
    model.eval()

    print('sample_path', sample_path)

    with torch.no_grad():
        synthesize(model, args.temp, args.num_synth)