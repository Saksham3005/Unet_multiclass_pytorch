import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from Activ_func import fwd1, down, up, fwd0

class UNet_model(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear

        self.inc = (fwd1(n_channels, 64))
        self.down1 = (down(64, 128))
        self.down2 = (down(128, 256))
        self.down3 = (down(256, 512))
        self.down4 = (down(512, 1024))
        self.up1 = (up(1024, 512))
        self.up2 = (up(512, 256))
        self.up3 = (up(256, 128))
        self.up4 = (up(128, 64))
        self.outc = (fwd0(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    

model = UNet_model(3, 5)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)


x = torch.randn(1, 3, 256, 256)
y = torch.randint(0, 5, (1, 256, 256)) 


output = model(x)


loss = criterion(output, y)


print(f'Initial loss: {loss.item()}')


optimizer.zero_grad()
loss.backward()
optimizer.step()

output = model(x)
loss = criterion(output, y)
print(f'Loss after one optimization step: {loss.item()}')


summary(model, (3, 256, 256))

