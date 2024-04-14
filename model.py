import torch.nn as nn
from utils import operate_elif, operate
from torch import Tensor
import torch


def conv1x1(in_channels:int, out_channels: int, bias: bool) -> nn.Conv2d:
   return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

def conv3x3(in_channels: int, out_channels: int, bias: bool) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool,
            batch_normal: bool,
            downsample: bool
    ) -> None:
        super(BottleneckBlock, self).__init__()
        norm2d = operate(batch_normal == True, nn.BatchNorm2d, nn.InstanceNorm2d)
        self.conv1 = conv1x1(in_channels, out_channels, bias)
        self.bn1 = norm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, bias)
        self.bn2 = norm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels*self.expansion, bias)
        self.bn3 = norm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if downsample == True:
          self.down = nn.Sequential(
              conv1x1(in_channels, out_channels*self.expansion, bias),
              norm2d(out_channels*self.expansion)
          )
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample == True:
            x += self.down(x)
            
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(
            self,
            channels: int,
            num_classes: int,
            layer: list,
            bias: bool,
            batch_normal: bool,
            init_weights: bool
    ) -> None:
        super(ResNet, self).__init__()
        norm2d = operate(batch_normal == True, nn.BatchNorm2d, nn.InstanceNorm2d)
        self.conv1 = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = norm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(1, len(layer) + 1):
            bottleneck = []

            for j in range(layer[i - 1]):
                middle = int(channels*(2**i)/2)
                head = operate_elif(
                    i == 1 and j == 0, channels,
                    i != 0 and j == 0, middle*2, middle*BottleneckBlock.expansion
                )
                down = operate(j == 0, True, False)
                bottleneck.append(BottleneckBlock(head, middle, bias, batch_normal, down))

            layers.append(bottleneck)

        self.layer1 = nn.Sequential(*layers[0])
        self.layer2 = nn.Sequential(*layers[1])
        self.layer3 = nn.Sequential(*layers[2])
        self.layer4 = nn.Sequential(*layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(channels**1.5), num_classes)

        if init_weights == True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif isinstance(m, norm2d):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not True:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RPNBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ratio: list,
            anchor: list,
            stride: int,
    ) -> None:
        super(RPNBlock, self).__init__()
        