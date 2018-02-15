import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Classification #
##################

class ResBlock(nn.Module):
    def __init__(self, NumChannels): 
        super().__init__()
        self.conv0 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
        self.selu0 = nn.SELU()
        self.selu1 = nn.SELU()
    def forward(self, x):
        y = self.conv0(x)
        y = self.selu0(y)
        y = self.conv1(y)
        return self.selu1(torch.add(y, x))

class ResNet(nn.Module):
    def build_layer(self, NumLayers, NumChannels):
        layers = []
        for _ in range(NumLayers):
            layers.append(ResBlock(NumChannels))
        return nn.Sequential(*layers)
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 192, 3, stride=1)
        self.fc1 = nn.Linear(192, 192)
        self.fc2 = nn.Linear(192, 2)
        self.selu0 = nn.SELU()
        self.selu1 = nn.SELU()
        self.selu2 = nn.SELU()
        self.selu3 = nn.SELU()
        self.selu4 = nn.SELU()
        self.selu5 = nn.SELU()
        self.selu6 = nn.SELU()
        self.block0 = self.build_layer(4, 32)
        self.block1 = self.build_layer(4, 64)
        self.block2 = self.build_layer(4, 96)
    def forward(self, x):
        x = x.view(-1, 1, 25, 25, 25)
        x = self.selu0(x)
        x = self.conv0(x)
        x = self.selu1(x)
        x = self.block0(x)
        x = self.conv1(x)
        x = self.selu2(x)
        x = self.block1(x)
        x = self.conv2(x)
        x = self.selu3(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.selu4(x)
        x = self.conv4(x)
        x = self.selu5(x)
        x = x.view(-1, 192)
        x = self.fc1(x)
        x = self.selu6(x)
        x = self.fc2(x)
        return x

class Classifier():
    def __init__(self, learningRate, decayRate):
        self.net = ResNet()
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
    def train(self, ECALs, HCALs, truth):
        self.net.train()
        self.optimizer.zero_grad()
        outputs = self.net(ECALs)
        loss = self.lossFunction(outputs, truth)
        loss.backward()
        self.optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == truth.data).sum()/truth.shape[0]
        return (loss.data[0], accuracy)
    def eval(self, ECALs, HCALs, truth):
        self.net.eval()
        outputs = self.net(ECALs)
        loss = self.lossFunction(outputs, truth)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == truth.data).sum()/truth.shape[0]
        return (loss.data[0], accuracy)
    def save(self, path):
        torch.save(self.net, path)
