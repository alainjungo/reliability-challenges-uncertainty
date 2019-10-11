import os
import glob
import json

import torch
import torch.nn as nn
import torch.optim as optim

import common.utils.filehelper as fh
import common.model.factory as factory
import common.configuration.config as cfg


class ModelFiles:
    CHECKPOINT_PLACEHOLDER = 'checkpoint{postfix}_ep{epoch:03d}.pth'
    BEST_PLACEHOLDER = 'checkpoint{postfix}_ep{epoch:03d}-best.pth'
    MODELDIR_PREFIX = 'model_'

    def __init__(self, root_model_dir: str, identifier: str) -> None:
        self.identifier = identifier
        self.root_model_dir = root_model_dir

    @classmethod
    def from_model_dir(cls, model_dir: str):
        if model_dir.endswith('/'):
            model_dir = model_dir[:-1]
        root_dir = os.path.dirname(model_dir)
        model_id = os.path.basename(model_dir)[len(cls.MODELDIR_PREFIX):]
        return cls(root_dir, model_id)

    @property
    def model_dir(self) -> str:
        return os.path.join(self.root_model_dir, '{}{}'.format(self.MODELDIR_PREFIX, self.identifier))

    @property
    def weight_checkpoint_dir(self) -> str:
        return os.path.join(self.model_dir, 'checkpoints')

    def model_path(self, postfix='') -> str:
        if len(postfix) > 0:
            postfix = '-{}'.format(postfix)
        return os.path.join(self.model_dir, 'model{}.json'.format(postfix))

    def build_checkpoint_path(self, epoch: int, is_best=False, postfix=''):
        if len(postfix) > 0:
            postfix = '-{}'.format(postfix)
        if is_best:
            return os.path.join(self.weight_checkpoint_dir,
                                ModelFiles.BEST_PLACEHOLDER).format(epoch=epoch, postfix=postfix)
        return os.path.join(self.weight_checkpoint_dir,
                            ModelFiles.CHECKPOINT_PLACEHOLDER).format(epoch=epoch, postfix=postfix)


class _ModelService:

    @staticmethod
    def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer=None):
        if not os.path.exists(checkpoint_path):
            raise ValueError('missing checkpoint file {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint.pop('state_dict'))
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        return checkpoint

    @staticmethod
    def load_model_from_parameters(model_path: str, with_optimizer=False, with_others=False):
        if not os.path.exists(model_path):
            raise ValueError('missing model file {}'.format(model_path))
        with open(model_path, 'r') as f:
            d = json.load(f)  # type: dict
        model_params = cfg.DictableParameter()
        model_params.from_dict(d.pop('model'))
        model = factory.get_model(model_params)

        ret_val = [model]
        if with_optimizer:
            optim_params = cfg.DictableParameter()
            optim_params.from_dict(d.pop('optimizer'))
            optimizer = factory.get_optimizer(model.parameters(), optim_params)
            ret_val.append(optimizer)

        if with_others:
            others = {k: v for k, v in d.items()}
            ret_val.append(others)

        return ret_val[0] if len(ret_val) == 1 else tuple(ret_val)

    @staticmethod
    def backup_model_parameters(model_path: str, model_params: cfg.DictableParameter,
                                optimizer_params: cfg.DictableParameter, **others) -> None:
        fh.create_dir_if_not_exists(model_path, is_file=True)
        with open(model_path, 'w') as f:
            json.dump({'model': model_params.to_dict(), 'optimizer': optimizer_params.to_dict(), **others}, f)

    @staticmethod
    def save_checkpoint(checkpoint_path: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer, **others) -> None:
        fh.create_dir_if_not_exists(checkpoint_path, is_file=True)
        save_dict = {'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), **others}
        torch.save(save_dict, checkpoint_path)

    @staticmethod
    def delete_checkpoint(checkpoint_dir: str, epoch_or_best_or_last) -> None:
        checkpoint_files = _ModelService.find_checkpoint_files(checkpoint_dir, epoch_or_best_or_last)

        for checkpoint_file in checkpoint_files:
            os.remove(checkpoint_file)

    @staticmethod
    def find_checkpoint_files(checkpoint_dir: str, epoch_or_best_or_last, epoch_can_be_best=False):
        if not isinstance(epoch_or_best_or_last, (str, int)):
            raise AttributeError('Expected epoch_or_best_or_last types are (string, int), not {}'.
                                 format(type(epoch_or_best_or_last)))

        epoch = epoch_or_best_or_last
        if isinstance(epoch_or_best_or_last, str):
            if epoch_or_best_or_last == 'last':
                epoch = _ModelService.find_last_checkpoint_epoch(checkpoint_dir)
            elif epoch_or_best_or_last == 'best':
                epoch = _ModelService.find_best_checkpoint_epoch(checkpoint_dir)
            else:
                raise ValueError('allowed string values for epoch are ({})'.format(('last', 'best')))

        if epoch is None:
            return []

        best_postfix = ''
        if epoch_or_best_or_last == 'best':
            best_postfix = '-best'
        elif epoch_can_be_best:
            best_postfix = '*'
        results = glob.glob(checkpoint_dir + '/checkpoint*ep*{:03d}{}.pth'.format(epoch, best_postfix))
        return results

    @staticmethod
    def find_checkpoint_file(checkpoint_dir: str, epoch_or_best_or_last, postfix=''):
        checkpoint_files = _ModelService.find_checkpoint_files(checkpoint_dir, epoch_or_best_or_last)

        if len(postfix) > 0:
            postfix = '-{}'.format(postfix)
        results = [f for f in checkpoint_files if os.path.basename(f).startswith('checkpoint{}'.format(postfix))]

        if len(results) == 0:
            return None
        return results[0]

    @staticmethod
    def find_best_checkpoint_epoch(checkpoint_dir: str):
        results = glob.glob(checkpoint_dir + '/checkpoint*ep*-best.pth')
        if len(results) == 0:
            return None
        epoch = int(os.path.basename(results[0])[-len('-best.pth') - 3: -len('-best.pth')])
        return epoch

    @staticmethod
    def find_last_checkpoint_epoch(checkpoint_dir: str):
        results = glob.glob(checkpoint_dir + '/checkpoint*ep{}.pth'.format(3*'[0-9]'))
        if len(results) == 0:
            return None
        epochs = [int(os.path.basename(r)[-len('.pth') - 3: -len('.pth')]) for r in results]
        return max(epochs)


model_service = _ModelService()

