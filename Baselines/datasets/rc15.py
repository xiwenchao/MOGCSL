import numpy as np
import pandas as pd
import torch
import pickle

class RC15Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path):
        # data = pd.read_csv(dataset_path).to_numpy()
        with open(dataset_path, 'rb') as f:
            data = np.array(pickle.load(f))
        data_statis = pd.read_pickle('./data/RC15/data_statis.df')
        item_num = data_statis['item_num'][0]
        state_size = data_statis['state_size'][0]
        self.outupt_dim = item_num
        self.state_size = state_size
        self.numerical_data = data[:, : -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.int32)
        self.numerical_num = self.numerical_data.shape[1]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.numerical_data[index], self.labels[index]

