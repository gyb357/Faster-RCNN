import torch.nn as nn
from torch import Tensor
from typing import Tuple


class ResNetBackbone(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


class RPN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_anchors: int = 9
    ) -> None:
        super(RPN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls = nn.Conv2d(in_channels, num_anchors*2, kernel_size=1)
        self.bbox = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.relu(x)
        x = self.conv(x)
        cls = self.cls(x)
        bbox = self.bbox(x)
        return cls, bbox


class RoIPool(nn.Module):
    def __init__(self, output_size: Tuple[int, int]) -> None:
        super(RoIPool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, features: Tensor, proposals: Tensor) -> Tensor:
        for i, proposal in enumerate(proposals):
            pass