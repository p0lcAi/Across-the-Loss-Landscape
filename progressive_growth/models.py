from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes: int = 100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MLP2D(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 512,
        depth: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("MLP2D depth must be >= 1")

        layers = []
        d_in = int(input_dim)

        for _ in range(int(depth)):
            layers.append(nn.Linear(d_in, int(hidden_dim)))
            layers.append(nn.ReLU(inplace=True))
            d_in = int(hidden_dim)

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.fc(x)
        return x

@dataclass
class ModelConfig:
    arch: str = "resnet18"
    num_classes: int = 100
    pretrained: bool = False
    vit_name: str = "vit_tiny_patch16_224"
    drop_path_rate: float = 0.0
    img_size: int = 32
    input_dim: int = 2
    hidden_dim: int = 512
    depth: int = 4



def build_model(cfg: ModelConfig) -> nn.Module:
    arch = cfg.arch.lower()
    if arch == "resnet18":
        return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.num_classes)
    if arch in {"vit_tiny", "tinyvit", "vit"}:
        try:
            import timm
        except ImportError as e:
            raise ImportError("Building the ViT model requires timm. Please pip install timm.") from e
        model = timm.create_model(
            cfg.vit_name,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            img_size=cfg.img_size,
            drop_path_rate=cfg.drop_path_rate,
        )
        return model
    if arch in {"mlp", "mlp2d", "overparam_mlp"}:
        return MLP2D(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            num_classes=cfg.num_classes,
        )
    raise ValueError(f"Unsupported architecture: {cfg.arch}")
