from pathlib import Path
import numpy as np
import torch, math
    
import torch
from torch.utils.data import DataLoader

from src.models.transformer import TransformerWithPE
from src.models.utils import (
    load_and_partition_data,
    make_datasets,
    split_sequence,
    visualize,
    infer
)

BS = 512
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = 100
NUM_VIS_EXAMPLES = 1
NUM_LAYERS = 2
LR = 0.001

def generate_data(data_path: Path, num_steps: int, interval: float = 0.1) -> None:
    x = np.linspace(0, num_steps * interval, num_steps)
    # y = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    y = np.sin(x)

    np.savez(data_path, y=y)

generate_data("data.npz", 1000000)

def main() -> None:
    # Load data and generate train and test datasets / dataloaders
    sequences, num_features = load_and_partition_data("data.npz")
    train_set, test_set = make_datasets(sequences)
    train_loader, test_loader = DataLoader(train_set, batch_size=BS, shuffle=True), DataLoader(test_set, batch_size=BS, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer and loss criterion
    model = TransformerWithPE(num_features, num_features, FEATURE_DIM, NUM_HEADS, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # Train loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            src, tgt, tgt_y = split_sequence(batch[0])
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_y = tgt_y.to(device)
            # [bs, tgt_seq_len, num_features]
            pred = model(src, tgt)
            loss = criterion(pred, tgt_y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: "
            f"{(epoch_loss / len(train_loader)):.4f}"
        )

    # Evaluate model
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            src, tgt, tgt_y = split_sequence(batch[0])
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            # [bs, tgt_seq_len, num_features]
            pred_infer = infer(model, src, tgt.shape[1])
            infer_loss = criterion(pred_infer, tgt_y)
            eval_loss += loss.item()

            if idx < NUM_VIS_EXAMPLES:
                visualize(src, tgt, pred, pred_infer)

    avg_eval_loss = eval_loss / len(test_loader)
    avg_infer_loss = infer_loss / len(test_loader)

    print(f"Eval / Infer Loss on test set: {avg_eval_loss:.4f} / {avg_infer_loss:.4f}")

if __name__ == "__main__":
    main()
