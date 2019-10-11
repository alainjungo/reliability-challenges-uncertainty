import torch
import torch.nn as nn
import torch.nn.functional as F

import common.model.helpers as helper


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=None, bn=True, activation=True, kernel=3, padding=1):
        super().__init__()
        self.conv2d_batch_relu = nn.Sequential()
        self.conv2d_batch_relu.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel, padding=padding))
        if dropout is not None:
            self.conv2d_batch_relu.add_module('dropout', nn.Dropout2d(p=dropout))
        if bn:
            self.conv2d_batch_relu.add_module('bn', nn.BatchNorm2d(out_ch))
        if activation:
            self.conv2d_batch_relu.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv2d_batch_relu(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=None, dropout_mode='all', bn=True, repetitions=2):
        super().__init__()
        convs = []
        in_ch_temp = in_ch
        for i in range(repetitions):
            do = _get_dropout(dropout, dropout_mode, i, repetitions)
            convs.append(Conv2dBnRelu(in_ch_temp, out_ch, do, bn))
            in_ch_temp = out_ch
        self.block = nn.Sequential(*convs)

    def forward(self, x):
        return self.block(x)


class ConvResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=None, dropout_mode='all', bn=True, repetitions=2):
        super().__init__()
        convs = []
        in_ch_temp = in_ch
        for i in range(repetitions - 1):
            do = _get_dropout(dropout, dropout_mode, i, repetitions)
            convs.append(Conv2dBnRelu(in_ch_temp, out_ch, do, bn))
            in_ch_temp = out_ch

        do = _get_dropout(dropout, dropout_mode, repetitions - 1, repetitions)
        convs.append(Conv2dBnRelu(in_ch_temp, out_ch, do, bn, activation=False))
        self.block = nn.Sequential(*convs)
        self.residual = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.block(x) + self.residual(x)
        return x


def _get_dropout(dropout, dropout_mode, i, repetitions):
    if dropout_mode == 'all':
        return dropout
    if dropout_mode == 'first' and i == 0:
        return dropout
    if dropout_mode == 'last' and i == repetitions - 1:
        return dropout
    if dropout_mode == 'no':
        return None
    return None


def _get_dropout_mode(dropout_center, curr_depth, depth, is_down):
    if dropout_center is None:
        return 'all'
    if curr_depth == depth:
        return 'no'
    if curr_depth + dropout_center >= depth:
        return 'last' if is_down else 'first'
    return 'no'


class DownConv(nn.Module):
    def __init__(self, block):
        super().__init__()

        self.block = block
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_x = self.block(x)
        x = self.pool(skip_x)
        return x, skip_x


class UpConv(nn.Module):
    def __init__(self, block, in_ch, out_ch, transpose=False):
        super().__init__()
        self.block = block
        if transpose:
            self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        else:
            self.upconv = nn.Sequential(helper.InterpolateWrapper(scale_factor=2), nn.Conv2d(in_ch, out_ch, 3, padding=1))

    def forward(self, x, skip_x):
        up = self.upconv(x)

        up_shape, skip_shape = up.size()[-2:], skip_x.size()[-2:]
        if up_shape < skip_shape:
            x_diff = skip_shape[-2] - up_shape[-2]
            y_diff = skip_shape[-1] - up_shape[-1]
            x_pad = (x_diff // 2, x_diff // 2 + (x_diff % 2))
            y_pad = (y_diff // 2, y_diff // 2 + (y_diff % 2))
            up = F.pad(up, y_pad + x_pad)

        x = torch.cat((up, skip_x), 1)
        x = self.block(x)
        return x


class UNet(nn.Module):
    DEFAULT_DEPTH = 4
    DEFAULT_START_FILTERS = 16
    DEFAULT_DROPOUT = 0.2

    def __init__(self, nb_classes, in_channels, depth=DEFAULT_DEPTH,
                 start_filters=DEFAULT_START_FILTERS, dropout=DEFAULT_DROPOUT, dropout_center:int=None, residual=False,
                 sigma_out=False, provide_features=False, bn=True):
        super().__init__()
        block_cls = ConvResidualBlock if residual else ConvBlock

        self.down_convs = nn.ModuleList()
        self.provide_features = provide_features
        self.features = None

        in_ch = in_channels
        out_ch = start_filters

        for i in range(depth):
            do_mode = _get_dropout_mode(dropout_center, i, depth, True)
            down_conv = DownConv(block_cls(in_ch, out_ch, dropout, do_mode, bn))
            self.down_convs.append(down_conv)
            in_ch = out_ch
            out_ch *= 2

        do_mode = _get_dropout_mode(dropout_center, depth, depth, True)
        self.bottom_convs = block_cls(in_ch, out_ch, dropout, do_mode, bn)

        self.up_convs = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            in_ch = out_ch
            out_ch = in_ch // 2
            do_mode = _get_dropout_mode(dropout_center, i, depth, False)
            up_conv = UpConv(block_cls(2*out_ch, out_ch, dropout, do_mode, bn), in_ch, out_ch)
            self.up_convs.append(up_conv)

        in_ch = out_ch
        self.conv_cls = nn.Sequential(Conv2dBnRelu(in_ch, in_ch, dropout, bn), nn.Conv2d(in_ch, nb_classes, 1))
        self.conv_sigma = None
        if sigma_out:
            self.conv_sigma = nn.Sequential(Conv2dBnRelu(in_ch, in_ch, dropout, bn), nn.Conv2d(in_ch, nb_classes, 1))

    def forward(self, x):
        skip_xs = []
        for down_conv in self.down_convs:
            x, skip_x = down_conv(x)
            skip_xs.append(skip_x)

        x = self.bottom_convs(x)

        for inv_depth, up_conv in enumerate(self.up_convs, 1):
            skip_x = skip_xs[-inv_depth]
            x = up_conv(x, skip_x)

        if self.provide_features:
            self.features = x

        out_logits = self.conv_cls(x)
        if self.conv_sigma is None:
            return out_logits

        out_sigma = self.conv_sigma(x)
        return out_logits, out_sigma

