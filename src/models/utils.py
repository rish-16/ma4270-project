"""
Taken from:
https://github.com/hyunwoongko/transformer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns

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


def load_and_partition_data(data_path, seq_length=100):
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
    src,
    tgt,
    pred_infer,
    viz_file_path,
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
    plt.plot(x[:src_len], src[idx].cpu().detach(),label="Source", color="b", marker="*")
    plt.plot(x[src_len:], tgt[idx].cpu().detach(), label="Target", color="g", marker="o")
    # plt.plot(x[src_len:], pred[idx].cpu().detach(), label="pred", color="r", marker="s")
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(), label="Prediction", color="r", marker="1")
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$f(x)$", fontsize=14)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    fig.savefig(viz_file_path)

@torch.no_grad()
def infer(model, src, tgt_len):
    output = torch.zeros((1, src.size(1) + tgt_len, src.size(2))).to(src.device)  # Batch size of 1
    output[:, :src.size(1), :] = src

    for i in range(tgt_len):
        inp = output[:, i:i+src.shape[1], :]
        out = model(inp)
        output[:, i+src.shape[1]] = out

    return output[:, src.shape[1]:]

def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_heads = 1
    num_layers = 1
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    
    for row in range(num_layers):
        for column in range(num_heads):
            # ax[row][column].imshow(attn_maps[1][column], origin='lower', vmin=0)
            ax[row][column] = sns.heatmap(attn_maps[1][1], linewidth=0.5)
            ax[row][column].set_xticks(list(range(seq_len))[::9])
            ax[row][column].set_xticklabels(input_data.tolist()[::9])
            ax[row][column].set_yticks(list(range(seq_len))[::9])
            ax[row][column].set_yticklabels(input_data.tolist()[::9])
            ax[row][column].set_title(f"Layer {row+1+1}, Head {column+1}")
            plt.ylabel("Queries")
            plt.xlabel("Keys")
    fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    fig.savefig("attn_plots_layer2.pdf")
    plt.show()

def viz_weights(model, src, n_heads):
    src = model.encoder_embedding(src)
    src = model.positional_encoding(src)
    attn_maps = model.transformer_encoder.get_attention_maps(src) # get final layer attention maps for all heads
    # print ("atten", attn_maps.shape)

    plot_attention_maps(None, attn_maps)

    return attn_maps