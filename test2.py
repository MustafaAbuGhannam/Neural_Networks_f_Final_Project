from unittest import result
import matplotlib.pyplot as plt
import torch
from Net import Net
from VGGNet import VggNet
from ResNet18 import ResNet
from contrastiveLoss import ContrastiveLoss
from train import trainContrastive, train
from test import testContrastive, test
from data_set import LinesDataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    model = ResNet()
    loss_function = torch.nn.MSELoss()
    loss_history = []
    train_acc = []
    test_acc = []
    y_pred = []
    y_true = []
    result1 = []
    result2 = []
    test_line_data_set = LinesDataSet(csv_file="Test_labels_for_english.csv", root_dir='english_data_set', transform=transforms.Compose([transforms.ToTensor()]))
    test_line_data_loader = DataLoader(test_line_data_set, shuffle=True, batch_size=1)
    if torch.cuda.is_available():
        model = model.cuda()
        loss_function = loss_function.cuda()
    model.load_state_dict(torch.load('C:/Users/FinalProject/Desktop/important_for_the_project/models/model_without_reg_for_english_data_set/model_0.pt', map_location='cuda:0'))
    test(model, test_line_data_loader, acc_history= [], all_acc_train=train_acc, all_acc_test= test_acc, train_flag=False, y_pred = y_pred, y_true = y_true, result = result1, result1=result2)
    plt.subplot(211)
    plt.imshow(np.transpose(result1[0][0][0].cpu().numpy(),(1,2,0)),cmap= "gray")
    plt.subplot(212)
    plt.imshow(np.transpose(result1[0][1][0].cpu().numpy(),(1,2,0)),cmap= "gray")
    plt.show()
    plt.subplot(211)

    plt.imshow(np.transpose(result2[0][0][0].cpu().numpy(),(1,2,0)),cmap= "gray")
    plt.subplot(212)

    plt.imshow(np.transpose(result2[0][1][0].cpu().numpy(),(1,2,0)),cmap= "gray")
    plt.show()
    