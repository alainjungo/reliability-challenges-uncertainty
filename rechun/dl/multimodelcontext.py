import torch

import common.trainloop.context as ctx
import common.trainloop.factory as factory
import common.model.management as mgt
import common.utils.torchhelper as th


class MultiModelTorchTrainContext(ctx.TorchTrainContext):

    def __init__(self, device_str) -> None:
        super().__init__(device_str)
        self.additional_models = {}
        self.additional_optimizers = {}

    def load_from_new(self):
        super().load_from_new()  # retrieval of the first/standard model

        if not hasattr(self.config.others, 'model_names'):
            raise ValueError('model_names entry missing in others section of configuration')

        if not hasattr(self.config.others, 'additional_models'):
            raise ValueError('additional_models entry missing in others section of configuration')

        if not hasattr(self.config.others, 'additional_optimizers'):
            raise ValueError('additional_optimizers entry missing in others section of configuration')

        for i, name in enumerate(self.config.others.model_names):
            model = factory.get_model(self.config.others.additional_models[i])
            model = self._multi_gpu_if_available(model)
            model.to(self.device)
            self.additional_models[name] = model

            optimizer = factory.get_optimizer(model.parameters(), self.config.others.additional_optimizers[i])
            self.additional_optimizers[name] = optimizer

            mgt.model_service.backup_model_parameters(self.model_files.model_path(postfix=name),
                                                      self.config.model.to_dictable_parameter(),
                                                      self.config.optimizer.to_dictable_parameter())

    def save_to_checkpoint(self, epoch: int, is_best=False):
        super().save_to_checkpoint(epoch, is_best)

        for name in self.additional_models:
            checkpoint_path = self.model_files.build_checkpoint_path(epoch, is_best=is_best, postfix=name)
            mgt.model_service.save_checkpoint(checkpoint_path, epoch, self.additional_models[name],
                                              self.additional_optimizers[name])

    def load_from_checkpoint(self, epoch):
        super().load_from_checkpoint(epoch)

        for name in self.additional_models:
            # build, since we know it is a int epoch
            checkpoint_path = self.model_files.build_checkpoint_path(epoch, postfix=name)

            model, optimizer = mgt.model_service.load_model_from_parameters(self.model_files.model_path(postfix=name),
                                                                            with_optimizer=True)
            mgt.model_service.load_checkpoint(checkpoint_path, model, optimizer)

            model = self._multi_gpu_if_available(model)
            self.additional_models[name] = model.to(self.device)
            self.additional_optimizers[name] = th.optimizer_to_device(optimizer, self.device)

    def set_mode(self, is_train: bool) -> None:
        self.is_train = is_train
        if self.is_train:
            self.model.train()
            for model in self.additional_models.values():
                model.train()

        else:
            self.model.eval()
            for model in self.additional_models.values():
                model.eval()

        torch.set_grad_enabled(self.is_train)
