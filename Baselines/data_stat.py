import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np

from datasets.aliexpress import AliExpressDataset
from datasets.kaggle import KaggleDataset
from models.sharedbottom import SharedBottomModel
from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe import MMoEModel
from models.aitm import AITMModel
from models.metaheac import MetaHeacModel





def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset(path)
    elif 'Kaggle' in name:
        return KaggleDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)
    

def main(dataset_name,
         dataset_path):
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.pickle')
    train_data_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

    loader = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)

    for i, (numerical_fields, labels) in enumerate(loader):
        print(labels.shape)  # [128, 2]
        valid = np.where(labels[:, 1] != -1)[0]
        valid_labels = labels[valid, 1]
        valid_numerical_fields = numerical_fields[valid]
        print(valid_numerical_fields.shape)

        break

if __name__ == '__main__':
    main('Kaggle', './data/')