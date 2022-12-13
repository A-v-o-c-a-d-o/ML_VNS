import torch
import pandas as pd
import numpy as np
from torch import nn 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Datasets.datasets import *


def train(model, img_model, loss_func, img_loss_func, optimizer, img_optimizer, data_loader, lr_scheduler, epoch):
    print(f'Epoch {epoch}: ')
    model.train()
    img_model.train()
    losses = []
    correct = 0

    for input in data_loader:
        optimizer.zero_grad()
        img_optimizer.zero_grad()
        outputs = model(input)
        img_outputs = img_model(input) #img

        loss = loss_func(outputs, input['targets'])
        img_loss = img_loss_func(img_outputs, input['targets']) #img
        pred = torch.round((2*torch.max(outputs, dim=1)[1] + torch.max(img_outputs, dim=1)[1])/3)
        # _, pred = torch.max(img_outputs, dim=1)

        correct += torch.sum(pred == input['targets'])
        losses.append(loss.item() + img_loss.item())
        # losses.append(img_loss.item())
        img_loss.backward() #img
        img_optimizer.step() #img
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(data_loader.dataset)} Loss: {np.mean(losses)}')

def test(model, img_model, data_loader, loss_func, img_loss_func):
    model.eval()
    img_model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        for input in data_loader:
            outputs = model(input)
            img_outputs = img_model(input) #img

            pred = torch.round((2*torch.max(outputs, dim=1)[1] + torch.max(img_outputs, dim=1)[1])/3)
            # _, pred = torch.max(img_outputs, dim=1)

            loss = loss_func(outputs, input['targets'])
            img_loss = img_loss_func(img_outputs, input['targets']) #img
            correct += torch.sum(pred == input['targets'])
            losses.append(loss.item() + img_loss.item())
            # losses.append(img_loss.item())
    
        print(f'Test Accuracy: {correct.double()/len(data_loader.dataset)} Loss: {np.mean(losses)}')

def get_dataloader():
    df = pd.read_csv('Data/full_train.csv')
    df = df.dropna()
    # shuffle the DataFrame rows
    df = df.sample(frac = 1)
    train_df, test_df = train_test_split(df, train_size=0.7)
    train_dataloader = DataLoader(MyDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(MyDataset(test_df), batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, test_dataloader

def predict(model: torch.nn.Module, img_model: torch.nn.Module, N):
    df = pd.read_csv('Data/test.csv')
    dataloader = DataLoader(MyDataset(df), batch_size=BATCH_SIZE, shuffle=False)
    n = df.shape[0]
    predicted = torch.Tensor([]).to(device=device)
    with torch.no_grad():
        for input in dataloader:
            pred = torch.round((2*torch.max(model(input), dim=1)[1] + torch.max(img_model(input), dim=1)[1])/3)
            # _, pred = torch.max(img_model(input), 1)
            predicted = torch.concat((predicted, pred), dim=0)
            if (predicted.shape[0]%(64*5) == 0 or predicted.shape[0] == n): 
                print(f'Running: {100 * predicted.shape[0] / n}%')
    assert predicted.shape[0] == n
    df['Rating'] = predicted.tolist()
    submission = pd.concat([df['RevId'], df['Rating']], axis=1)
    submission.to_csv(f'Results/submission{N}.csv', index=False)