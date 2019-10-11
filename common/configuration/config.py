from pymia.config.configuration import *


class DictableParameter(Dictable):

    def __init__(self, type_: str=None, **kwargs) -> None:
        super().__init__()
        self.type = type_
        self.params = kwargs

    def to_dict(self, **kwargs):
        return vars(self)

    def from_dict(self, d: dict, **kwargs):
        if 'type' in d:
            self.type = d.get('type')
        if 'params' in d:
            self.params = d.get('params')


class DictableParameterExt(Dictable):

    def __init__(self, type_: str=None, **kwargs) -> None:
        super().__init__()

        self.type = type_
        self.params = kwargs

    def to_dict(self, **kwargs):
        return {self.type: self.params}

    def from_dict(self, d: dict, **kwargs):
        assert len(d) == 1
        self.type = next(iter(d))
        self.params = d[self.type]

    def to_dictable_parameter(self):
        return DictableParameter(self.type, **self.params)
