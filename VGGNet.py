from torch import nn
import torch
import torchvision


class VggNet(nn.Module):
    def __init__(self):
        super(VggNet, self).__init__()
        self.cnn = torchvision.models.vgg16_bn(pretrained=False)
        self.cnn.features[0] = nn.Conv2d(1, 64, 3, 1, 1)
        num_features = 1000
        #self.cnn.add_module('classifier2', nn.Sequential(nn.ReLU(),nn.Linear(num_features, 200), nn.ReLU()))
        self.fc1 = nn.Sequential(nn.Linear(4000, 2), nn.Sigmoid())

    def forward_once(self, x):
        output = self.cnn(x)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        h_ = output1 * output2
        dist_ = torch.pow((output1 - output2), 2)
        V_ = torch.cat((output1, output2, dist_, h_), dim=1)
        output = self.fc1(V_)
        return output

