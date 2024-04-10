import os
import numpy as np
from pathlib import Path
from scipy import signal

def generate_square_data(train_data_path: Path, test_data_path: Path, train_num_steps: int, test_num_steps: int, seq_len: int, interval: float = 0.1) -> None:
    context_len = seq_len - 1
    train_start, train_end = 0, train_num_steps * interval
    test_start, test_end = train_end - context_len * interval, train_end  + test_num_steps * interval

    train_x = np.linspace(train_start, train_end, train_num_steps, endpoint=False)
    train_y = signal.square(train_x)

    test_x = np.linspace(test_start, test_end, test_num_steps + context_len, endpoint=False)
    test_y = signal.square(test_x)

    np.savez(train_data_path, y=train_y)
    np.savez(test_data_path, y=test_y)


def generate_sine_incr_data(train_data_path: Path, test_data_path: Path, train_num_steps: int, test_num_steps: int, seq_len: int, interval: float = 0.1) -> None:
    ALPHA = 0.01
    context_len = seq_len - 1
    train_start, train_end = 0, train_num_steps * interval
    test_start, test_end = train_end - context_len * interval, train_end  + test_num_steps * interval

    train_x = np.linspace(train_start, train_end, train_num_steps, endpoint=False)
    train_y = np.sin(train_x) * np.exp(ALPHA * train_x)

    test_x = np.linspace(test_start, test_end, test_num_steps + context_len, endpoint=False)
    test_y = np.sin(test_x) * np.exp(ALPHA * test_x)

    np.savez(train_data_path, y=train_y)
    np.savez(test_data_path, y=test_y)

def generate_sine_data(train_data_path: Path, test_data_path: Path, train_num_steps: int, test_num_steps: int, seq_len: int, interval: float = 0.1) -> None:
    context_len = seq_len - 1
    train_start, train_end = 0, train_num_steps * interval
    test_start, test_end = train_end - context_len * interval, train_end + test_num_steps * interval

    train_x = np.linspace(train_start, train_end, train_num_steps, endpoint=False)
    train_y = np.sin(train_x)

    test_x = np.linspace(test_start, test_end, test_num_steps + context_len, endpoint=False)
    test_y = np.sin(test_x)

    np.savez(train_data_path, y=train_y)
    np.savez(test_data_path, y=test_y)