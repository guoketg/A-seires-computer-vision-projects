import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data_cifar10, train


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, in_channels, out_channels, expansion_factor, kernel_size, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        expanded_channels = in_channels * expansion_factor

        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                      stride, kernel_size // 2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )

        # Squeeze and Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, expanded_channels, 1),
            nn.Sigmoid()
        )

        # Output phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x

        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)  # SE attention
        x = self.project(x)

        if self.use_residual:
            x = x + residual

        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # Base configuration for EfficientNet-B0
        base_config = [
            # expansion, out_channels, kernel_size, stride, num_repeats
            [1, 16, 3, 1, 1],
            [6, 24, 3, 2, 2],
            [6, 40, 5, 2, 2],
            [6, 80, 3, 2, 3],
            [6, 112, 5, 1, 3],
            [6, 192, 5, 2, 4],
            [6, 320, 3, 1, 1]
        ]

        # Adjust channels and repeats based on width and depth multipliers
        adjusted_config = []
        for exp, out, k, s, rep in base_config:
            out_channels = self._adjust_channels(out, width_mult)
            num_repeats = self._adjust_repeats(rep, depth_mult)
            adjusted_config.append([exp, out_channels, k, s, num_repeats])

        # Build stem
        stem_channels = self._adjust_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )

        # Build blocks
        blocks = []
        in_channels = stem_channels
        for exp, out_channels, k, s, rep in adjusted_config:
            for i in range(rep):
                stride = s if i == 0 else 1
                blocks.append(MBConvBlock(in_channels, out_channels, exp, k, stride))
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        # Build head
        head_channels = self._adjust_channels(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(head_channels, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _adjust_channels(self, channels, width_mult):
        return int(channels * width_mult)

    def _adjust_repeats(self, repeats, depth_mult):
        return int(repeats * depth_mult)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def efficientnet_b0(num_classes=10):
    """EfficientNet-B0 configuration"""
    return EfficientNet(num_classes=num_classes, width_mult=1.0, depth_mult=1.0)


def efficientnet_b1(num_classes=10):
    """EfficientNet-B1 configuration"""
    return EfficientNet(num_classes=num_classes, width_mult=1.0, depth_mult=1.1)


# Training script
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    resize = 128  # EfficientNet typically uses 224x224 input

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path="D:\code\python\data\CIFAR10"
    train_loader, test_loader = load_data_cifar10(batch_size, path, resize)
    model = efficientnet_b0(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train(model, train_loader, test_loader, num_epochs, criterion, optimizer, device)
    torch.save(model.state_dict(),"D:\code\python\save_position\\trained_efficientnet.pth")