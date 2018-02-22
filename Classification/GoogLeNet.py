import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Classification #
##################

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes, eps=epsilon),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192, eps=epsilon),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# class ResBlock(nn.Module):
#     def __init__(self, NumChannels): 
#         super().__init__()
#         self.conv0 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
#         self.conv1 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
#         self.selu0 = nn.SELU()
#         self.selu1 = nn.SELU()
#     def forward(self, x):
#         y = self.conv0(x)
#         y = self.selu0(y)
#         y = self.conv1(y)
#         return self.selu1(torch.add(y, x))

# class ResNet(nn.Module):
#     def build_layer(self, NumLayers, NumChannels):
#         layers = []
#         for _ in range(NumLayers):
#             layers.append(ResBlock(NumChannels))
#         return nn.Sequential(*layers)
#     def __init__(self):
#         super().__init__()
#         self.conv0 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
#         self.conv1 = nn.Conv3d(32, 64, 3, stride=2)
#         self.conv2 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
#         self.conv4 = nn.Conv3d(128, 192, 3, stride=1)
#         self.fc1 = nn.Linear(192, 192)
#         self.fc2 = nn.Linear(192, 2)
#         self.selu0 = nn.SELU()
#         self.selu1 = nn.SELU()
#         self.selu2 = nn.SELU()
#         self.selu3 = nn.SELU()
#         self.selu4 = nn.SELU()
#         self.selu5 = nn.SELU()
#         self.selu6 = nn.SELU()
#         self.block0 = self.build_layer(4, 32)
#         self.block1 = self.build_layer(4, 64)
#         self.block2 = self.build_layer(4, 96)
#     def forward(self, x, _):
#         x = x.view(-1, 1, 25, 25, 25)
#         x = self.selu0(x)
#         x = self.conv0(x)
#         x = self.selu1(x)
#         x = self.block0(x)
#         x = self.conv1(x)
#         x = self.selu2(x)
#         x = self.block1(x)
#         x = self.conv2(x)
#         x = self.selu3(x)
#         x = self.block2(x)
#         x = self.conv3(x)
#         x = self.selu4(x)
#         x = self.conv4(x)
#         x = self.selu5(x)
#         x = x.view(-1, 192)
#         x = self.fc1(x)
#         x = self.selu6(x)
#         x = self.fc2(x)
#         return x

class Classifier():
    def __init__(self, learningRate, decayRate):
        self.net = ResNet()
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
