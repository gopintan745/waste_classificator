import torch
from torch import nn
from torch.nn.modules import MaxPool2d


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1, bias=False) -> None:
        super().__init__(ConvBlock, self)
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x):
        return self.block(x)



class WasteClassifierCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3, base_filters=32, dropout=0.3) -> None:
        super().__init__(WasteClassifierCNN, self)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.block1 = ConvBlock(base_filters, base_filters * 2, dropout=dropout),
        self.block2 = ConvBlock(base_filters * 2, base_filters*4, dropout=dropout),
        self.block3 = ConvBlock(base_filters * 4, base_filters * 8, dropout=dropout),
        self.block4 = ConvBlock(base_filters * 8, base_filters * 16, dropout=dropout),
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 16 *2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.cat([self.gap(x), self.gmp(x)])
        return self.head(x)


if __name__ == "__main__":
    m = WasteClassifierCNN(num_classes=6)
    from torchsummary import summary
    summary(m, (3, 224, 224))