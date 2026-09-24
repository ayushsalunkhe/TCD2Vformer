import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

def extract_standard_time_features(df: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(df['date'])
    hour = dates.dt.hour.values / 23.0 - 0.5
    weekday = dates.dt.weekday.values / 6.0 - 0.5
    day = (dates.dt.day.values - 1) / 30.0 - 0.5
    month = (dates.dt.month.values - 1) / 11.0 - 0.5
    return np.stack([hour, weekday, day, month], axis=1)

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, stamp: np.ndarray, seq_len: int = 96, pred_len: int = 48):
        self.data = data
        self.stamp = stamp
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.stamp[s_begin:s_end]
        seq_y_mark = self.stamp[r_begin:r_end]

        return (torch.tensor(seq_x, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
                torch.tensor(seq_x_mark, dtype=torch.float32),
                torch.tensor(seq_y_mark, dtype=torch.float32))

def get_data_loaders(dataset_name: str, seq_len: int = 96, pred_len: int = 48, batch_size: int = 64, data_root: str = './datasets'):
    if dataset_name.lower() == 'etth1':
        csv_path = os.path.join(data_root, 'ETT-small', 'ETTh1.csv')
    elif dataset_name.lower() == 'etth2':
        csv_path = os.path.join(data_root, 'ETT-small', 'ETTh2.csv')
    elif dataset_name.lower() == 'ettm1':
        csv_path = os.path.join(data_root, 'ETT-small', 'ETTm1.csv')
    elif dataset_name.lower() == 'ettm2':
        csv_path = os.path.join(data_root, 'ETT-small', 'ETTm2.csv')
    elif dataset_name.lower() in ['exchange', 'exchange_rate']:
        csv_path = os.path.join(data_root, 'exchange_rate', 'exchange_rate.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col != 'date']
    data_raw = df[feature_cols].values.astype(np.float32)
    stamp_raw = extract_standard_time_features(df).astype(np.float32)

    n_total = len(data_raw)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val

    train_data = data_raw[:n_train]
    val_data = data_raw[n_train:n_train + n_val]
    test_data = data_raw[n_train + n_val:]

    train_stamp = stamp_raw[:n_train]
    val_stamp = stamp_raw[n_train:n_train + n_val]
    test_stamp = stamp_raw[n_train + n_val:]

    mean = np.mean(train_data, axis=0, keepdims=True)
    std = np.std(train_data, axis=0, keepdims=True)
    std[std == 0] = 1.0

    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std

    train_set = TimeSeriesWindowDataset(train_norm, train_stamp, seq_len=seq_len, pred_len=pred_len)
    val_set = TimeSeriesWindowDataset(val_norm, val_stamp, seq_len=seq_len, pred_len=pred_len)
    test_set = TimeSeriesWindowDataset(test_norm, test_stamp, seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    metadata = {
        'dataset': dataset_name,
        'num_variables': data_raw.shape[1],
        'stamp_dim': stamp_raw.shape[1],
        'total_samples': n_total,
        'train_samples': len(train_set),
        'val_samples': len(val_set),
        'test_samples': len(test_set),
        'mean': mean,
        'std': std
    }

    return train_loader, val_loader, test_loader, metadata
