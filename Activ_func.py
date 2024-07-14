import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class fwd0(nn.Module):
    def __init__(self, inCh, outCh):
        super(fwd0, self).__init__()
        self.conv0 = nn.Conv2d(inCh, outCh, kernel_size=1)

    def forward(self, x):
        return self.conv0(x)

class fwd1(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(inCh, outCh, kernel_size = 3, padding = 158, bias = False),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv3x3_1(x)

class fwd2(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(inCh, outCh, kernel_size = 3, padding = 0, bias = False),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv3x3_2(x)
    

    
class down(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.max_conv = nn.Sequential(
            nn.MaxPool2d(2),
            fwd2(inCh, outCh)
        )

    def forward(self, x):
        return self.max_conv(x)
    
class up(nn.Module):

    def __init__(self, inCh, outCh):
        super().__init__()

        
        self.up = nn.ConvTranspose2d(inCh, inCh // 2, kernel_size=2, stride=2)
        self.conv = fwd2(inCh, outCh)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

