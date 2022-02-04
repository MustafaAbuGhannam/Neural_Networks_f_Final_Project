import imp
from operator import mod
from statistics import mode
from typing import List
import torch
import torch.nn as nn
from contrastiveLoss import ContrastiveLoss



def trainContrastive(model : nn.Module, loss_function : ContrastiveLoss, optimizer : torch.optim.Adam, train_loader : List, loss_history : List):
    model.train()
    batches_loss = []

    for _ , data in enumerate(train_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()

        output1, output2 = model(image1, image2)
        optimizer.zero_grad()
        loss = loss_function(output1, output2, label)
        batches_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
    
    loss_history.append(sum(batches_loss) / len(batches_loss))
    print('epoch loss: {}'.format(sum(batches_loss) / len(batches_loss)))




    
def train (model : nn.Module, loss_function : nn.MSELoss, optimizer : torch.optim.Adam, train_loader : List, loss_history : List):
    model.train() 
    batches_loss = []
    for _ , data in enumerate(train_loader):
        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()

        output = model(image1, image2)
        optimizer.zero_grad()
        new_label = torch.empty((label.shape[0], 2)).float().cuda()

        for i in range(label.shape[0]):
            if label[i].item() == 1.:
                new_label[i][0] = 0.
                new_label[i][1] = 1.
            if label[i].item() == 0.:
                new_label[i][0] = 1.
                new_label[i][1] = 0.

        loss = loss_function(output, new_label)
        batches_loss.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
    loss_history.append( sum(batches_loss) / len(batches_loss))
    print('epoch loss: {}'.format(sum(batches_loss) / len(batches_loss)))