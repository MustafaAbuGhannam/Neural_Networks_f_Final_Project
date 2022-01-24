import torch
import torch.nn as nn
import torchvision



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = torchvision.models.resnet18(pretrained = False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, 3, bias=False)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, 64)

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return (output1, output2)

