import os
import pickle
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset


class IPIIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = self.index_all_data()

    def index_all_data(self):
        extension = ('.pkl')
        filename_list = sorted([f for f in os.listdir(self.data_dir) if (os.path.isfile(os.path.join(self.data_dir, f)))])
        filename_list = [f for f in filename_list if f.endswith(extension)]
        return filename_list

    def tanh(self, x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        image_path = os.path.join(self.data_dir, self.data_list[item])

        # load
        with open(image_path, "rb") as f:
            slice_dict = pickle.load(f)
        signal = slice_dict['signal']
        label = slice_dict['label']

        # transform signal to magnitude images
        signal = np.abs(signal)

        # add additional noise to signal
        std = np.random.normal(0.05, 0.05)
        std = np.clip(std, 0.01, 0.3)
        std = 0.1
        noise_signal = np.random.normal(0.0, std, signal.shape).astype(np.float32)
        signal = signal
        signal = np.clip(signal, 0, 2).astype(np.float32)

        # scale label
        label = 2 * self.tanh(label) - 1

        return signal, label

