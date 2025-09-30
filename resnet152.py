import torch
import torch.nn as nn

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=3, input_channels=8, dropout=0.5):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=dropout)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x  # Apply activation externally based on your task


def ResNet152_1D(num_classes=3, input_channels=8, dropout=0.5):
    return ResNet1D(Bottleneck1D, [3, 8, 36, 3],
                    num_classes=num_classes,
                    input_channels=input_channels,
                    dropout=dropout)


def ResNet50_1D(num_classes=3, input_channels=8, dropout=0.5):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3],
                    num_classes=num_classes,
                    input_channels=input_channels,
                    dropout=dropout)                   