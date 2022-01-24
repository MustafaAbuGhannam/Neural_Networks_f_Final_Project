from typing import List
import torch
import torch.nn as nn
from contrastiveLoss import ContrastiveLoss
import torch.nn.functional as F


def test(model : nn.Module, test_loader : List, acc_history : List, all_acc_train : List, all_acc_test : List, train_flag : bool, y_pred : List, y_true : List):
    model.eval()
    for _ , data in enumerate(test_loader):

        image1, image2, label = data
        image1 = image1.float().cuda()
        image2 = image2.float().cuda()
        image1 = image1[:,None,:,:]
        image2 = image2[:,None,:,:]
        label = label.cuda()

        with torch.no_grad:
            output1, output2 = model(image1, image2)
            dist_ = F.pairwise_distance(output1, output2,keepdim=True)
            dist_ = torch.pow(dist_, 2)
            y_true.append(int(label.cpu().item()))
            
            if label.item() == 1. and  dist_.item() >= 0.5: 
                acc_history.append(1)
                y_pred.append(1)
            elif label.item() == 0.  and dist_.item() < 0.5:
                acc_history.append(1)
                y_pred.append(0)
            else:
                acc_history.append(0)
                if label.item() == 1.:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
    
    print('test acc: {}'.format(sum(acc_history) / len(acc_history)))
    
    if train_flag:
        all_acc_train.append(sum(acc_history) / len(acc_history))
    else:
        all_acc_test.append(sum(acc_history) / len(acc_history))
