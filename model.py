import torch
import torch.nn as nn

class ImprovedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = block(512 + 256, 256)
        self.dec2 = block(256 + 128, 128)
        self.dec3 = block(128 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d1 = self.dec1(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d3)
