import torch.nn as nn

import common.model.unet as unet


class PostNet(nn.Module):

    def __init__(self, in_channels, nb_classes, nb_convs=3, dropout=None):
        super().__init__()
        convs = [unet.Conv2dBnRelu(in_channels, in_channels, dropout, kernel=1, padding=0) for _ in range(nb_convs)]
        self.convs = nn.Sequential(*convs)
        self.conv_logits = nn.Conv2d(in_channels, nb_classes, 1)

    def forward(self, x):
        x = self.convs(x)
        x = self.conv_logits(x)
        return x

