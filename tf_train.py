from pathlib import Path
import numpy as np
import torch

import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from typing import Callable

from src.models.transformer import TransformerWithPE
from src.models.utils import (
    get_device,
    load_and_partition_data,
    visualize,
    infer,
    viz_weights
)

from src.models.gen_data import (
    generate_sine_data,
    generate_sine_incr_data,
    generate_square_data
)

BS = 512
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = 100
NUM_VIS_EXAMPLES = 1
NUM_LAYERS = 2
LR = 0.001

SEQ_LEN = 100
TRAIN_NUM_STEPS = 1000
TEST_NUM_STEPS = 200


def train(train_data_path: Path, test_data_path: Path, viz_file_path: Path,
          generate_data: Callable[[int, int, int, float], None] = None):
    if generate_data is not None:
        generate_data(train_data_path, test_data_path,
                      TRAIN_NUM_STEPS, TEST_NUM_STEPS, SEQ_LEN)

    train_sequences, train_num_features = load_and_partition_data(
        train_data_path, SEQ_LEN)
    test_tgt_y = torch.Tensor(np.load(test_data_path)["y"])  # Batch size of 1
    test_tgt_y = test_tgt_y.reshape((1, test_tgt_y.size(0), 1))

    train_set = TensorDataset(torch.Tensor(train_sequences))
    train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
    device = torch.device(get_device())

    model = TransformerWithPE(train_num_features, train_num_features,
                              FEATURE_DIM, NUM_HEADS, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            src = batch[0][:, :-1]
            tgt_y = batch[0][:, -1]

            src = src.to(device)
            tgt_y = tgt_y.to(device)

            # [bs, tgt_seq_len, num_features]
            pred = model(src)
            loss = criterion(pred, tgt_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: "
            f"{(epoch_loss / len(train_loader)):.4f}"
        )

    model.eval()
    with torch.no_grad():
        tgt_len = test_tgt_y.size(1) - SEQ_LEN + 1
        src = test_tgt_y[:, :SEQ_LEN-1, :]
        src = src.to(device)

        pred_infer = infer(model, src, tgt_len=tgt_len)
        # infer_loss = criterion(pred_infer, tgt_y)
        # eval_loss += loss.item()
        src_for_viz = torch.Tensor(np.load(train_data_path)["y"])[-3*SEQ_LEN:]
        src_for_viz = src_for_viz.reshape((1, src_for_viz.size(0), 1))

        visualize(
            src_for_viz, test_tgt_y[:, SEQ_LEN-1:, :], pred_infer, viz_file_path)

    return model


def main() -> None:
    model = train(
        os.path.join(os.getcwd(), "data", "sine_train.npz"),
        os.path.join(os.getcwd(), "data", "sine_test.npz"),
        os.path.join(os.getcwd(), "tf_sine.pdf"),
        generate_sine_data
    )
    viz_weights(model)

    # model = train(
    #     os.path.join(os.getcwd(), "data", "sine_incr_train.npz"),
    #     os.path.join(os.getcwd(), "data", "sine_incr_test.npz"),
    #     os.path.join(os.getcwd(), "tf_sine_incr.pdf"),
    #     generate_sine_incr_data
    # )
    # viz_weights(model)

    # model = train(
    #     os.path.join(os.getcwd(), "data", "square_train.npz"),
    #     os.path.join(os.getcwd(), "data", "square_test.npz"),
    #     os.path.join(os.getcwd(), "tf_square.pdf"),
    #     generate_square_data
    # )
    # viz_weights(model)

if __name__ == "__main__":
    main()
