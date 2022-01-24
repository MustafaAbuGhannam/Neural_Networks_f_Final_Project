import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, output1, output2, label):
        distance_ = F.pairwise_distance(output1, output2,keepdim=True)
        # cos  = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        # distance_ = 1 - cos(output1, output2) 
        loss_contrastive = torch.mean((label)* torch.pow(distance_, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - distance_, min=0.0), 2))

        return loss_contrastive
