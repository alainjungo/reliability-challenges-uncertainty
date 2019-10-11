import os
import abc
import logging
import typing
import shutil

import torch
import torch.nn as nn
import tensorboardX

import common.utils.logginghelper as log_help
import common.trainloop.factory as factory
import common.data.split as split
import common.model.management as mgt
import common.utils.idhelper as idh
import common.utils.torchhelper as th
import common.utils.filehelper as fh
import common.utils.messages as msg
import common.trainloop.config as cfg
import common.trainloop.data as data


class Context(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.config = None

    @abc.abstractmethod
    def load_from_config(self, config_file: str) -> None:
        pass

    @abc.abstractmethod
    def load_from_checkpoint(self, epoch: int) -> None:
        pass

    @abc.abstractmethod
    def do_seed(self, seed: int) -> None:
        pass

    @abc.abstractmethod
    def get_seed(self) -> typing.Union[int, None]:
        pass

    @abc.abstractmethod
    def setup_directory(self) -> None:
        pass

    @abc.abstractmethod
    def setup_logging(self) -> None:
        pass


class TrainContext(Context, abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.best_score = None
        self.is_train = True
        self.resume_train_at = None
        self.train_data = None
        self.valid_data = None

    @abc.abstractmethod
    def save_to_checkpoint(self, epoch: int, is_best=False) -> None:
        pass

    @abc.abstractmethod
    def load_from_new(self):
        pass

    @abc.abstractmethod
    def load_train_and_valid_data(self, build_train: data.BuildData, build_valid: data.BuildData) -> None:
        pass

    @abc.abstractmethod
    def get_resume_at(self) -> typing.Union[int, None]:
        pass

    @abc.abstractmethod
    def get_task_context(self, epoch: int):
        pass

    @abc.abstractmethod
    def need_validation(self, epoch: int) -> bool:
        pass

    @abc.abstractmethod
    def set_mode(self, is_train: bool) -> None:
        pass


class TorchTrainContext(TrainContext):

    def __init__(self, device_str) -> None:
        super().__init__()
        self.train_id = ''
        self.train_dir = ''
        self.valid_dir = ''
        self.log_file = ''

        self.model_files = None  # type: mgt.ModelFiles
        self.optimizer = None

        self.device = torch.device(device_str)
        self.tb = None  # type: tensorboardX.SummaryWriter
        self.config_file_path = None

    def load_from_config(self, config_file: str):
        self.config_file_path = config_file
        config = cfg.load(self.config_file_path, cfg.TrainConfiguration)
        if not isinstance(config, cfg.TrainConfiguration):
            raise ValueError(msg.get_type_error_msg(config, cfg.TrainConfiguration))
        self.config = config

        id_ = idh.extract_leading_identifier(self.config.train_name)
        name = self.config.train_name
        if id_:
            resume = True
            name = self.config.train_name.replace(id_ + '_', '')
        else:
            resume = False
            id_ = idh.get_unique_identifier()
        self.train_id = id_
        self.train_dir = os.path.join(self.config.train_dir, '{}_{}'.format(self.train_id, name))
        self.valid_dir = os.path.join(self.train_dir, 'validation')
        self.log_file = os.path.join(self.train_dir, 'log.txt')
        self.model_files = mgt.ModelFiles(self.train_dir, self.train_id)

        if resume:
            last_epoch = mgt.model_service.find_last_checkpoint_epoch(self.model_files.weight_checkpoint_dir)
            if last_epoch is not None:
                self.resume_train_at = last_epoch

    def setup_directory(self):
        fh.create_and_clean_dir(self.train_dir)
        fh.create_dir_if_not_exists(self.valid_dir)

        copy_config_name = 'config{}'.format(os.path.splitext(self.config_file_path)[1])
        # default parameters only in save (therefore not copying)
        cfg.save(os.path.join(self.train_dir, copy_config_name), self.config)

        if self.config.split:
            shutil.copy(self.config.split,
                        os.path.join(self.train_dir, os.path.basename(self.config.split)))

    def setup_logging(self):
        self.tb = tensorboardX.SummaryWriter(log_dir=self.train_dir)
        log_help.setup_file_logging(self.log_file)

    def load_train_and_valid_data(self, build_train: data.BuildData, build_valid: data.BuildData):
        train_params, valid_params = {}, {}
        if self.config.split:
            split_k = None
            if hasattr(self.config.others, 'split_k'):
                split_k = self.config.others.split_k
            train_entries, valid_entries, _ = split.load_split(self.config.split, split_k)
            train_params['entries'] = train_entries
            valid_params['entries'] = valid_entries

        self.train_data = build_train(self.config.train_data, **train_params)
        self.valid_data = build_valid(self.config.valid_data, **valid_params)

    def load_from_new(self):
        self.model = factory.get_model(self.config.model)
        self.model = self._multi_gpu_if_available(self.model)
        self.model.to(self.device)

        self.optimizer = factory.get_optimizer(self.model.parameters(), self.config.optimizer)

        mgt.model_service.backup_model_parameters(self.model_files.model_path(),
                                                  self.config.model.to_dictable_parameter(),
                                                  self.config.optimizer.to_dictable_parameter())

    def save_to_checkpoint(self, epoch: int, is_best=False):
        checkpoint_path = self.model_files.build_checkpoint_path(epoch, is_best=is_best)
        mgt.model_service.save_checkpoint(checkpoint_path, epoch, self.model, self.optimizer,
                                          best_score=self.best_score)

    def load_from_checkpoint(self, epoch):
        checkpoint_path = self.model_files.build_checkpoint_path(epoch)  # build, since we know it is a int epoch

        model, optimizer = mgt.model_service.load_model_from_parameters(self.model_files.model_path(),
                                                                        with_optimizer=True)
        others = mgt.model_service.load_checkpoint(checkpoint_path, model, optimizer)

        self.model = self._multi_gpu_if_available(model)
        self.model = self.model.to(self.device)
        self.optimizer = th.optimizer_to_device(optimizer, self.device)

        if 'best_score' not in others:
            logging.warning('could not find "best_score" in the checkpoint')
        else:
            self.best_score = others['best_score']

    def get_task_context(self, epoch: int):
        if self.is_train:
            return TaskContext(epoch, self.train_data, self.config.train_data)
        else:
            return TaskContext(epoch, self.valid_data, self.config.valid_data)

    def do_seed(self, seed: int):
        th.do_seed(seed, with_cudnn=False)

    def get_seed(self) -> typing.Union[int, None]:
        return self.config.seed

    def get_resume_at(self) -> typing.Union[int, None]:
        return self.resume_train_at

    def need_validation(self, epoch: int) -> bool:
        return ((epoch + 1) % self.config.valid_every_nth) == 0

    def set_mode(self, is_train: bool) -> None:
        self.is_train = is_train
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()
        torch.set_grad_enabled(self.is_train)

    @staticmethod
    def _multi_gpu_if_available(model):
        supported_cuda_devices = []
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_capability(i)[0] >= 6:
                supported_cuda_devices.append(i)

        if len(supported_cuda_devices) > 1:
            logging.info('-- use dataparallel since {} gpus visible'.format(len(supported_cuda_devices)))
            model = nn.DataParallel(model, device_ids=supported_cuda_devices)
        return model


class TestContext(Context, abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.test_data = None
        self.model = None

    @abc.abstractmethod
    def load_test_data(self, build_test: data.BuildData) -> None:
        pass

    @abc.abstractmethod
    def get_test_at(self) -> typing.Union[int, str]:
        pass

    @abc.abstractmethod
    def get_task_context(self):
        pass


class TorchTestContext(TestContext):

    def __init__(self, device_str: str) -> None:
        super().__init__()
        self.test_id = ''
        self.test_dir = ''
        self.log_file = ''

        self.model_files = None  # type: mgt.ModelFiles

        self.device = torch.device(device_str)
        self.config_file_path = None

    def load_from_config(self, config_file: str) -> None:
        self.config_file_path = config_file
        config = cfg.load(self.config_file_path, cfg.TestConfiguration)
        if not isinstance(config, cfg.TestConfiguration):
            raise ValueError(msg.get_type_error_msg(config, cfg.TestConfiguration))
        self.config = config

        test_dir = self.config.test_dir
        if self.config.test_dir is None or len(self.config.test_dir) == 0:
            # if not defined in config file, take the train directory as test directory
            train_dir = os.path.dirname(config.model_dir)
            test_dir = os.path.join(train_dir, 'tests')

        self.test_id = idh.get_unique_identifier()
        self.test_dir = os.path.join(test_dir, '{}_{}'.format(self.test_id, self.config.test_name))
        self.log_file = os.path.join(self.test_dir, 'log.txt')
        self.model_files = mgt.ModelFiles.from_model_dir(self.config.model_dir)

    def setup_directory(self) -> None:
        fh.create_dir_if_not_exists(self.test_dir)

        copy_config_name = 'config{}'.format(os.path.splitext(self.config_file_path)[1])
        # default parameters only in save (therefore not copying)
        cfg.save(os.path.join(self.test_dir, copy_config_name), self.config)

        if self.config.split:
            shutil.copy(self.config.split, os.path.join(self.test_dir, os.path.basename(self.config.split)))

    def load_test_data(self, build_test: data.BuildData) -> None:
        test_params = {}

        if self.config.split:
            split_k = None
            if hasattr(self.config.others, 'split_k'):
                split_k = self.config.others.split_k
            _, _, test_entries = split.load_split(self.config.split, split_k)
            test_params['entries'] = test_entries
        self.test_data = build_test(self.config.test_data, **test_params)

    def get_test_at(self) -> typing.Union[int, str]:
        return self.config.test_at

    def get_task_context(self):
        return TaskContext(0, self.test_data, self.config.test_data)

    def load_from_checkpoint(self, epoch: int) -> None:
        checkpoint_path = mgt.model_service.find_checkpoint_file(self.model_files.weight_checkpoint_dir, epoch)

        model = mgt.model_service.load_model_from_parameters(self.model_files.model_path(), with_optimizer=False)
        mgt.model_service.load_checkpoint(checkpoint_path, model)
        self.model = model.to(self.device)

        self.model.eval()
        torch.set_grad_enabled(False)

    def do_seed(self, seed: int) -> None:
        th.do_seed(seed, with_cudnn=False)

    def get_seed(self) -> typing.Union[int, None]:
        return self.config.seed

    def setup_logging(self) -> None:
        log_help.setup_file_logging(self.log_file)


class BatchContext:

    def __init__(self, batch: dict, batch_index: int) -> None:
        self.input = batch
        self.batch_index = batch_index
        self.output = {}
        self.metrics = {}
        self.score = None
        self.more = {}


class TaskContext:

    def __init__(self, epoch: int, task_data, task_data_config) -> None:
        self.epoch = epoch
        self.data = task_data
        self.data_config = task_data_config
        self.history = History()
        self.scores = []
        self.more = {}


class SubjectContext:

    def __init__(self, subject_index, subject_data: dict) -> None:
        self.subject_index = subject_index
        self.subject_data = subject_data
        self.metrics = {}
        self.score = None
        self.more = {}


class History:

    def __init__(self) -> None:
        self.categories = {}

    def add_entry(self, id_, value, category: str):
        self.categories.setdefault(category, {}).setdefault(id_, []).append(value)

    def add(self, entries: dict, category: str) -> None:
        for k, v in entries.items():
            self.add_entry(k, v, category)

    def get_entries_keys(self, category: str = None) -> tuple:
        return tuple(self.categories[category].keys())

    def get_entries(self, entry_key: str, category: str) -> list:
        return self.categories[category][entry_key]

    def get_entry_size(self, entry_key: str, category: str):
        return len(self.get_entries(entry_key, category))

    def get_entries_by_index(self, index:int, category: str):
        entries = {}
        for entry_key in self.get_entries_keys(category):
            entries[entry_key] = self.get_entries(entry_key, category)[index]
        return entries

    def get_tasks(self) -> tuple:
        return tuple(self.categories.keys())

    def clear(self, category: str, entry_key=None):
        if entry_key is not None:
            del self.categories[category][entry_key]
        else:
            del self.categories[category]
