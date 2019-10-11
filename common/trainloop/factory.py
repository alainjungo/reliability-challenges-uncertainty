import typing

import pymia.data.transformation as tfm
import pymia.data.extraction as extr

import common.configuration.config as cfg
import common.model.factory as model_factory


transform_registry = {'size': tfm.SizeCorrection,
                      'permute': tfm.Permute,
                      'squeeze': tfm.Squeeze,
                      'unsqueeze': tfm.UnSqueeze,
                      'rescale': tfm.IntensityRescale,
                      'relabel': tfm.Relabel}


def get_transform(transform_params: typing.Union[cfg.DictableParameterExt, list, tuple]) -> tfm.Transform:
    if isinstance(transform_params, (list, tuple)):
        transforms = []
        for transform_param in transform_params:
            transforms.append(get_transform(transform_param))
        return tfm.ComposeTransform(transforms)

    if transform_params.type not in transform_registry:
        raise ValueError('transform type "{}" unknown'.format(transform_params.type))
    return transform_registry[transform_params.type](**transform_params.params)


extractor_registry = {'names': extr.NamesExtractor,
                      'data': extr.DataExtractor,
                      'pad': extr.PadDataExtractor,
                      'shape': extr.ImageShapeExtractor,
                      'properties': extr.ImagePropertiesExtractor,
                      'files': extr.FilesExtractor,
                      'indexing': extr.IndexingExtractor,
                      'random': extr.RandomDataExtractor,
                      'selective': extr.SelectiveDataExtractor,
                      'subject': extr.SubjectExtractor}


def get_extractor(extractor_params: typing.Union[cfg.DictableParameterExt, list, tuple]) -> extr.Extractor:
    if isinstance(extractor_params, (list, tuple)):
        extractors = []
        for extractor_param in extractor_params:
            extractors.append(get_extractor(extractor_param))
        return extr.ComposeExtractor(extractors)

    if extractor_params.type not in extractor_registry:
        raise ValueError('extractor type "{}" unknown'.format(extractor_params.type))
    if extractor_params.type.startswith('pad'):
        if 'extractor' not in extractor_params.params:
            raise ValueError('pad data extractor need a extractor parameter')
        inner_params = cfg.DictableParameterExt()
        inner_params.from_dict(extractor_params.params['extractor'])
        extractor = get_extractor(inner_params)
        extractor_params.params['extractor'] = extractor

    return extractor_registry[extractor_params.type](**extractor_params.params)


indexing_registry = {'slice': extr.SliceIndexing, 'empty': extr.EmptyIndexing, 'patch': extr.PatchWiseIndexing}


def get_indexing(indexing_params: cfg.DictableParameterExt) -> extr.IndexingStrategy:
    if indexing_params.type not in indexing_registry:
        raise ValueError('indexing type "{}" unknown'.format(indexing_params.type))
    return indexing_registry[indexing_params.type](**indexing_params.params)


selection_registry = {'none-black': extr.NonBlackSelection, 'with-foreground': extr.WithForegroundSelection}


def get_selection_strategy(selection_params: typing.Union[cfg.DictableParameterExt, list, tuple]) -> extr.SelectionStrategy:
    if isinstance(selection_params, (list, tuple)):
        selections = []
        for selection_param in selection_params:
            selections.append(get_selection_strategy(selection_param))
        return extr.ComposeSelection(selections)

    if selection_params.type not in selection_registry:
        raise ValueError('selection type "{}" unknown'.format(selection_params.type))
    return selection_registry[selection_params.type](**selection_params.params)


def get_model(model_params: cfg.DictableParameterExt):
    if model_params.type not in model_factory.model_registry:
        raise ValueError('model type "{}" unknown'.format(model_params.type))

    return model_factory.get_model(model_params.to_dictable_parameter())


def get_optimizer(params, optimizer_params: cfg.DictableParameterExt):
    if optimizer_params.type not in model_factory.optimizer_registry:
        raise ValueError('model type "{}" unknown'.format(optimizer_params.type))

    return model_factory.get_optimizer(params, optimizer_params.to_dictable_parameter())
