import random

import torch
import torch.nn as nn
import numpy as np

channel_dim = 1


def channel_to_end(tensor: torch.Tensor):
    return tensor.permute(*get_channel_to_end_permutation(tensor.dim()))


def end_to_channel(tensor: torch.Tensor):
    return tensor.permute(*get_end_to_channel_permutation(tensor.dim()))


def get_channel_to_end_permutation(dimensions: int):
    return (0,) + tuple(range(channel_dim + 1, dimensions)) + (1,)


def get_end_to_channel_permutation(dimensions: int):
    return (0, dimensions-1) + tuple(range(channel_dim, dimensions - 1))


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return optimizer


def do_seed(seed: int, with_cudnn: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes it perfectly deterministic but slower (without is already very good)
    if with_cudnn:
        torch.backends.cudnn.deterministic = True


def set_dropout_mode(model, is_train=True):
    for i, m in enumerate(model.modules()):
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if is_train:
                m.train()
            else:
                m.eval()


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)




