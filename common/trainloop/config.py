import typing

from common.configuration.config import *
import common.configuration.config as cfg


class TrainConfiguration(cfg.ConfigurationBase):

    @classmethod
    def version(cls) -> int:
        return 0

    @classmethod
    def type(cls) -> str:
        return 'train-config'

    def __init__(self) -> None:
        super().__init__()

        self.epochs = 100
        self.valid_every_nth = 1
        self.log_every_nth = 1
        self.optimizer = None
        self.model = None
        self.seed = 20
        self.split = ''
        self.train_dir = ''
        self.train_name = ''
        self.train_data = DataConfiguration()
        self.valid_data = DataConfiguration()
        self.others = OtherParameters()

    def to_dict(self, **kwargs):
        return member_to_dict_with_parametric(self, TrainConfiguration._parametric_members())

    def from_dict(self, d: dict, **kwargs):
        dict_to_member_with_parametric(self, d, TrainConfiguration._parametric_members())

    @staticmethod
    def _parametric_members():
        return 'model', 'optimizer'


class TestConfiguration(cfg.ConfigurationBase):

    @classmethod
    def version(cls) -> int:
        return 0

    @classmethod
    def type(cls) -> str:
        return 'test-config'

    def __init__(self) -> None:
        super().__init__()

        self.seed = 20
        self.split = ''
        self.model_dir = ''
        self.test_name = ''
        self.test_dir = None
        # 'best', 'last' or int epoch values
        self.test_at = ''  # type: typing.Union[int, str]
        self.test_data = DataConfiguration()
        self.others = OtherParameters()

    def to_dict(self, **kwargs):
        return member_to_dict_with_parametric(self, TestConfiguration._parametric_members())

    def from_dict(self, d: dict, **kwargs):
        dict_to_member_with_parametric(self, d, TestConfiguration._parametric_members())

    @staticmethod
    def _parametric_members():
        return tuple()


class DataConfiguration(cfg.Dictable):

    def __init__(self) -> None:
        super().__init__()

        self.dataset = ''

        self.batch_size = 10
        self.num_workers = 1
        self.extractor = None
        self.transform = None
        self.indexing = None
        self.selection_strategy = None
        self.selection_extractor = None
        self.shuffle = True

        self.direct_extractor = None
        self.direct_transform = None
        self.others = OtherParameters()

    def to_dict(self, **kwargs):
        return member_to_dict_with_parametric(self, DataConfiguration._parametric_members())

    def from_dict(self, d: dict, **kwargs):
        dict_to_member_with_parametric(self, d, DataConfiguration._parametric_members())

    @staticmethod
    def _parametric_members():
        return ('extractor', 'transform', 'indexing', 'selection_strategy', 'selection_extractor',
                'direct_extractor', 'direct_transform')


class OtherParameters(cfg.Dictable):

    def to_dict(self, **kwargs):
        return member_to_dict_with_parametric(self, OtherParameters._parametric_members(), allow_missing_member=True)

    def from_dict(self, d: dict, **kwargs):
        self.__dict__ = d  # required because to members before
        dict_to_member_with_parametric(self, d, OtherParameters._parametric_members(), allow_missing_member=True)

    @staticmethod
    def _parametric_members():
        return 'model', 'transform', 'additional_models', 'additional_optimizers'


def member_to_dict_with_parametric(obj, parametric_members, allow_missing_member=False):
    def from_parameter(p: cfg.DictableParameterExt):
        d_ = p.to_dict()
        if not p.params:
            # only consider type if no parameter
            d_ = next(iter(d_))
        return d_

    d = cfg.member_to_dict(obj)

    # handle the list case
    for param_member in parametric_members:
        if allow_missing_member and not hasattr(obj, param_member):
            # parametric member is not in the object
            continue
        value = getattr(obj, param_member)
        if isinstance(value, list):
            dict_value = []
            for i in range(len(value)):
                assert isinstance(value[i], cfg.DictableParameterExt)
                dict_value.append(from_parameter(value[i]))
            d[param_member] = dict_value
    return d


def dict_to_member_with_parametric(obj, d: dict, parametric_members, allow_missing_member=False):
    def to_parameter(d_):
        param = cfg.DictableParameterExt()
        param.from_dict(d_)
        return param

    def recursive_string_to_dict(entry):
        if isinstance(entry, dict):
            return entry
        if isinstance(entry, list):
            l = []
            for val in entry:
                l.append(recursive_string_to_dict(val))
            return l
        if isinstance(entry, str):
            return {entry: {}}

    for param_member in parametric_members:
        if param_member in d:
            d[param_member] = recursive_string_to_dict(d[param_member])

    # extract what is possible and already known
    dict_to_member(obj, d)

    # handle the list case
    for param_member in parametric_members:
        if allow_missing_member and not hasattr(obj, param_member):
            # parametric member was not in the config
            continue
        value = getattr(obj, param_member)
        if isinstance(value, Dictable) or value is None:
            # already handled entry by dict_to_member
            continue
        if isinstance(value, list):
            # list of configurable entries
            param_value = []
            for i in range(len(value)):
                param_value.append(to_parameter(value[i]))
            setattr(obj, param_member, param_value)
        elif isinstance(value, dict):
            # did not get captured by the dict_to_member function -> entry was none on the object
            setattr(obj, param_member, to_parameter(value))




