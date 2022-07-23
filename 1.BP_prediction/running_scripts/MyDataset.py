import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

DATA_ROOT = r'G:\PythonPro\BP_prediction\data'
sequence_file_name = 'input_sequence.csv'
feature_file_name = 'input_feature.csv'
BP_file_name = 'target_BP.csv'


class MainDataset(Dataset):
    # mode='train' or 'valid' or 'test'
    def __init__(self, mode, user_id, window_size, targetBP=['SBP', 'DBP'], nrows=None):
        sequence_file = os.path.join(DATA_ROOT, user_id, mode + '_' + sequence_file_name)
        feature_file = os.path.join(DATA_ROOT, user_id, mode + '_' + feature_file_name)
        BP_file = os.path.join(DATA_ROOT, user_id, mode + '_' + BP_file_name)
        self.window_size = window_size
        self.sequence_data = pd.read_csv(sequence_file, header=None, dtype=np.float32, nrows=nrows).values[:, -window_size:]
        self.feature_data = pd.read_csv(feature_file, header=None, dtype=np.float32, nrows=nrows).values
        self.bp_data = pd.read_csv(BP_file, header=None, dtype=np.float32, nrows=nrows).values
        if len(targetBP) != 2:
            if targetBP[0] == 'SBP':
                self.bp_data = np.expand_dims(self.bp_data[:, 0], axis=1)
            else:
                self.bp_data = np.expand_dims(self.bp_data[:, 1], axis=1)

        self.sample_num = self.bp_data.shape[0]
        self.sequence_data = torch.tensor(self.sequence_data)
        self.feature_data = torch.tensor(self.feature_data)
        self.bp_data = torch.tensor(self.bp_data)

    def __getitem__(self, index):
        return self.sequence_data[index], self.feature_data[index], self.bp_data[index]

    def __len__(self):
        return self.sample_num
