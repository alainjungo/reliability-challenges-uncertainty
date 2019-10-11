import torch
import torch.nn.functional as F


def smooth_dice(prediction, target, smooth=1.):
    _check_tensor_type(prediction, torch.float)
    _check_tensor_type(target, torch.float)

    iflat = prediction.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice


def nll(probabilities, target, do_log=True):
    _check_tensor_type(probabilities, torch.float)
    _check_tensor_type(target, torch.long)
    # log since results are probabilities and nll needs log_softmax inputs
    if do_log:
        probabilities = probabilities.log()

    target = target.view(-1)
    probabilities = probabilities.view(-1, probabilities.size()[-1])

    return F.nll_loss(probabilities, target)


def _check_tensor_type(obj, type_):
    if obj.dtype != type_:
        raise ValueError('object of type "{}" should be "{}"'.format(obj.dtype, type_))
