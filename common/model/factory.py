import torch.optim as optim

import common.model.unet as unet
import common.model.postnet as postnet
import common.configuration.config as cfg


def get_model(model_params: cfg.DictableParameter):
    return model_registry[model_params.type](**model_params.params)


model_registry = {
    'unet': unet.UNet,
    'postnet': postnet.PostNet
}


def get_optimizer(params, optim_params: cfg.DictableParameter):
    return optimizer_registry[optim_params.type](params, **optim_params.params)


optimizer_registry = {'adam': optim.Adam, 'sgd': optim.SGD}
