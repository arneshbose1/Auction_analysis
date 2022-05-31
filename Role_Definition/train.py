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
    
run = wandb.init(config = train_config, project = 'DDP', entity = 'arneshbose1')

cate = 'bowler'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = 'trained_models/'+cate+'.pth'
    
    
def main():
    with open('data_config.json') as fp:
        data_config = json.load(fp)
    
    batch_size = train_config['batch_size']
    num_epochs = train_config['num_epochs']
    lr = train_config['lr']

    print("Train Dataset")
    train_dataset = PlayerDataset(data_config, role=cate, split='train')
    print("Val Dataset")
    val_dataset = PlayerDataset(data_config, role=cate, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    print("Dataset:", dataset_sizes)

    model = Role_Classifier(train_config, device)
    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1000
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0005)
    
    print(model)
    
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        train_loss = 0
        val_loss = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0

            # Iterate over data.
            for examples in dataloaders[phase]:
                for key, value in examples.items():
                    if torch.is_tensor(examples[key]):
                        examples[key] = examples[key].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(examples)

                    loss = criterion(outputs, examples['role'])
                    running_loss += loss.item() * batch_size

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            wandb.log({ '{}_loss'.format(phase): epoch_loss})
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    wandb.log({'val_loss': best_val_loss})
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)
    print('Model saved at '+model_save_path)
    print(best_val_loss)

if __name__ == '__main__':
    main()