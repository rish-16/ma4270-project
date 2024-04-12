"""
Taken from:
https://github.com/hyunwoongko/transformer
"""

from pathlib import Path
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
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


def get_device():
    """Returns the device available.
    Returns:
        string: string representing the device available 
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
    sequences = np.empty(
        (len(data['y']) - seq_length + 1, seq_length, num_features))

    for i in range(0, len(data['y']) - seq_length + 1):
        # [sequence_length, num_features]
        sample = np.asarray(
            [data[key][i: i+seq_length] for key in data.keys()]
        ).swapaxes(0, 1)
        sequences[i] = sample

    return sequences, num_features


def visualize(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred_infer: torch.Tensor,
    viz_file_path: Path,
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
    plt.plot(x[:src_len], src[idx].cpu().detach(),
             label="Source", color="b", marker="*")
    plt.plot(x[src_len:], tgt[idx].cpu().detach(),
             label="Target", color="g", marker="o")
    # plt.plot(x[src_len:], pred[idx].cpu().detach(), label="pred", color="r", marker="s")
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(),
             label="Pred", color="y", marker="1")
    plt.legend()
    plt.grid()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    fig.savefig(viz_file_path)


def infer(model, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
    output = torch.zeros((1, src.size(1) + tgt_len, src.size(2))
                         ).to(src.device)  # Batch size of 1
    output[:, :src.size(1), :] = src

    for i in range(tgt_len):
        inp = output[:, i:i+src.shape[1], :]
        out = model(inp)
        output[:, i+src.shape[1]] = out

    return output[:, src.shape[1]:]

def viz_weights(model):
    attn_weights = model.encoder_layer.state_dict()['self_attn.in_proj_weight']
    print (attn_weights.shape)