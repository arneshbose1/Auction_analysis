import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataloader, dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import copy
import json
import time
import numpy as np
import random
import warnings
import wandb

warnings.filterwarnings("ignore")

from model import Role_Classifier
from dataset import PlayerDataset

train_config = dict(
    num_epochs = 100,
    batch_size = 32,
    lr = 0.001,
    encoder_activation = 'ReLu',
    encoder_depth = 2,
    encoder_dropout = 0.2,
    encoder_dimension = 8,
    decoder_activation = 'sigmoid',
    decoder_depth = 2,
    decoder_dropout = 0.1
    )


cate = 'bowler'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_load_path = 'trained_models/' + cate + '.pth'


def main():
    with open('data_config.json') as fp:
        data_config = json.load(fp)

    batch_size = train_config['batch_size']
    num_epochs = train_config['num_epochs']
    lr = train_config['lr']

    dataset = PlayerDataset(data_config, role=cate, split='eval')

    model = Role_Classifier(train_config, device)
    model = model.to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()


    player_pkey = 39
    sample={}
    sample['tendency'] = torch.from_numpy(dataset.get_tendency(player_pkey, role=cate).astype(np.float32))
    sample['career'] = torch.from_numpy(dataset.get_career(player_pkey, role=cate).astype(np.float32))
    sample['tendency'] = torch.reshape(sample['tendency'], (1,5))
    sample['career'] = torch.reshape(sample['career'],(1,5))

    for key, value in sample.items():
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].to(device)


    outputs = model(sample)
    print(player_pkey)
    print(sample)
    print(outputs)




if __name__ == '__main__':
    main()