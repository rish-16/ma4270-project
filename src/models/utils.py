"""
Taken from:
https://github.com/hyunwoongko/transformer
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_and_partition_data(
    data_path: Path, seq_length: int = 100
) -> tuple[np.ndarray, int]:
    """Loads the given data and paritions it into sequences of equal length.

    Args:
        data_path: path to the dataset
        sequence_length: length of the generated sequences

    Returns:
        tuple[np.ndarray, int]: tuple of generated sequences and number of
            features in dataset
    """
    data = np.load(data_path)
    num_features = len(data.keys())

    # Check that each feature provides the same number of data points
    data_lens = [len(data[key]) for key in data.keys()]
    assert len(set(data_lens)) == 1

    num_sequences = data_lens[0] // seq_length
    sequences = np.empty((num_sequences, seq_length, num_features))

    for i in range(0, num_sequences):
        # [sequence_length, num_features]
        sample = np.asarray(
            [data[key][i * seq_length : (i + 1) * seq_length] for key in data.keys()]
        ).swapaxes(0, 1)
        sequences[i] = sample

    return sequences, num_features


def make_datasets(sequences: np.ndarray) -> tuple[TensorDataset, TensorDataset]:
    """Create train and test dataset.

    Args:
        sequences: sequences to use [num_sequences, sequence_length, num_features]

    Returns:
        tuple[TensorDataset, TensorDataset]: train and test dataset
    """
    # Split sequences into train and test split
    train, test = train_test_split(sequences, test_size=0.2)
    return TensorDataset(torch.Tensor(train)), TensorDataset(torch.Tensor(test))


def visualize(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
    pred_infer: torch.Tensor,
    idx=0,
) -> None:
    """Visualizes a given sample including predictions.

    Args:
        src: source sequence [bs, src_seq_len, num_features]
        tgt: target sequence [bs, tgt_seq_len, num_features]
        pred: prediction of the model [bs, tgt_seq_len, num_features]
        pred_infer: prediction obtained by running inference
            [bs, tgt_seq_len, num_features]
        idx: batch index to visualize
    """
    x = np.arange(src.shape[1] + tgt.shape[1])
    src_len = src.shape[1]

    fig = plt.figure()
    plt.plot(x[:src_len], src[idx].cpu().detach(), label="Source", color="b", marker="*")
    plt.plot(x[src_len:], tgt[idx].cpu().detach(), label="Target", color="g", marker="o")
    # plt.plot(x[src_len:], pred[idx].cpu().detach(), label="pred", color="r", marker="s")
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(), label="Masked Pred", color="y", marker="1")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\sin(x)$")
    fig.savefig("tf_sin.pdf")

def split_sequence(
    sequence: np.ndarray, ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits a sequence into 2 (3) parts, as is required by our transformer
    model.

    Assume our sequence length is L, we then split this into src of length N
    and tgt_y of length M, with N + M = L.
    src, the first part of the input sequence, is the input to the encoder, and we
    expect the decoder to predict tgt_y, the second part of the input sequence.
    In addition we generate tgt, which is tgt_y but "shifted left" by one - i.e. it
    starts with the last token of src, and ends with the second-last token in tgt_y.
    This sequence will be the input to the decoder.


    Args:
        sequence: batched input sequences to split [bs, seq_len, num_features]
        ratio: split ratio, N = ratio * L

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: src, tgt, tgt_y
    """
    src_end = int(sequence.shape[1] * ratio)
    # [bs, src_seq_len, num_features]
    src = sequence[:, :src_end]
    # [bs, tgt_seq_len, num_features]
    tgt = sequence[:, src_end - 1 : -1]
    # [bs, tgt_seq_len, num_features]
    tgt_y = sequence[:, src_end:]

    return src, tgt, tgt_y

def infer(model, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
    output = torch.zeros((src.shape[0], tgt_len + 1, src.shape[2])).to(src.device)
    output[:, 0] = src[:, -1]
    for i in range(tgt_len):
        out = model(src, output)[:, i]
        print (output.shape, out.shape)
        output[:, i + 1] = out

    return output[:, 1:]