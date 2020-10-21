import torch
from torch.utils.data import Dataset
import numpy as np
import os
from datetime import datetime

use_cuda = torch.cuda.is_available()

max_time_steps = 16000
upsample_conditional_features = True
hop_length = 256


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x

def collate_fn(batch):
    """
    Create batch

    Args : batch(tuple) : List of tuples / (x, c)  x : list of (T,) c : list of (T, D)

    Returns : Tuple of batch / Network inputs x (B, C, T), Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2

    np.random.seed(datetime.now().microsecond)
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            if upsample_conditional_features:
                assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length

                max_steps = max_time_steps - max_time_steps % hop_length   # To ensure Divisibility

                if len(x) > max_steps:
                    max_time_frames = max_steps // hop_length
                    s = np.random.randint(0, len(c) - max_time_frames)
                    ts = s * hop_length
                    x = x[ts:ts + hop_length * max_time_frames]
                    c = c[s:s + max_time_frames]
                    assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length
            else:
                pass
            new_batch.append((x, c))
        batch = new_batch
    else:
        pass

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # x_batch : [B, T, 1]
    x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len) for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T')
        c_batch = torch.tensor(c_batch).transpose(1, 2).contiguous()
        del max_len
    else:
        c_batch = None

    # Convert to channel first i.e., (B, C, T) / C = 1
    x_batch = torch.tensor(x_batch).transpose(1, 2).contiguous()

    # Add extra axis i.e., (B, T, 1)
    y_batch = torch.tensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.tensor(input_lengths)
    return x_batch, y_batch, c_batch, input_lengths