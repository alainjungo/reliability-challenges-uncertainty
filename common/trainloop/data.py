import abc

import pymia.data.extraction as extr
import pymia.data.transformation as tfm

import common.data.selectionhelper as select
import common.data.collate as collate
import common.trainloop.config as cfg
import common.trainloop.factory as factory


class Data:

    def __init__(self, dataset, loader) -> None:
        self.dataset = dataset
        self.loader = loader
        self.nb_batches = len(loader)


class BuildDataset(abc.ABC):

    @abc.abstractmethod
    def __call__(self, data_config: cfg.DataConfiguration, **kwargs):
        pass


class BuildParametrizableDataset(BuildDataset):

    def __call__(self, data_config: cfg.DataConfiguration, **kwargs):
        entries = None
        if 'entries' in kwargs:
            entries = kwargs['entries']
        init_reader_once = True
        if 'init_reader_once' in kwargs:
            init_reader_once = kwargs['init_reader_once']

        indexing = factory.get_indexing(data_config.indexing)
        extractor = factory.get_extractor(data_config.extractor)
        transform = factory.get_transform(data_config.transform)

        return extr.ParameterizableDataset(
            dataset_path=data_config.dataset,
            indexing_strategy=indexing,
            extractor=extractor,
            transform=transform,
            subject_subset=entries,
            init_reader_once=init_reader_once
        )


class BuildLoader(abc.ABC):

    @abc.abstractmethod
    def __call__(self, dataset: extr.Dataset, sampler: extr.Sampler, data_config: cfg.DataConfiguration, **kwargs):
        pass


class BuildDefaultLoader:

    def __call__(self, dataset: extr.Dataset, sampler: extr.Sampler, data_config: cfg.DataConfiguration, **kwargs):
        if 'collate_entries' in kwargs:
            collate_fn = collate.CollateDict(entries=kwargs['collate_entries'])
        else:
            collate_fn = collate.CollateDict()
        return extr.DataLoader(dataset=dataset, batch_size=data_config.batch_size, sampler=sampler,
                               num_workers=data_config.num_workers, collate_fn=collate_fn)


class BuildSampler(abc.ABC):

    @abc.abstractmethod
    def __call__(self, dataset: extr.Dataset, data_config: cfg.DataConfiguration, **kwargs):
        pass


class BuildDefaultSampler(BuildSampler):

    def __call__(self, dataset: extr.Dataset, data_config: cfg.DataConfiguration, **kwargs):
        if data_config.shuffle:
            return extr.RandomSampler(dataset)
        return extr.SequentialSampler(dataset)


class BuildSubsetSampler(BuildSampler):

    def __call__(self, dataset: extr.Dataset, data_config: cfg.DataConfiguration, **kwargs):
        if 'entries' not in kwargs:
            raise ValueError('"entries" needed in kwargs to build sampler')

        entries = kwargs['entries']
        if data_config.shuffle:
            return extr.SubsetRandomSampler(indices=entries)
        return extr.SubsetSequentialSampler(indices=entries)


class BuildSelectionSampler(BuildSampler):

    def __call__(self, dataset: extr.Dataset, data_config: cfg.DataConfiguration, **kwargs):
        if not isinstance(dataset, extr.ParameterizableDataset):
            raise ValueError('dataset neeeds to be of type {}'.format(extr.ParameterizableDataset.__class__.__name__))

        selection_params = data_config.selection_strategy
        if not isinstance(selection_params, (list, tuple)):
            selection_params = [selection_params]

        subject_selection_params = [p for p in selection_params if p.type == 'subject']
        if len(subject_selection_params) > 0:
            if 'entries' not in kwargs:
                raise ValueError('"entries" needed in kwargs to build sampler')
            entries = kwargs['entries']

            assert len(subject_selection_params) == 1
            subject_selection_param = subject_selection_params[0]  # type: cfg.ParameterClass

            # add the subject entries to the parameters of the type
            subject_selection_param.params = entries

        selection_extractor = factory.get_extractor(data_config.selection_extractor)
        selection_strategy = factory.get_selection_strategy(selection_params)

        indices = self.get_indices(dataset, selection_extractor, selection_strategy)
        if data_config.shuffle:
            return extr.SubsetRandomSampler(indices=indices)
        return extr.SubsetSequentialSampler(indices=indices)

    @staticmethod
    def get_indices(dataset: extr.ParameterizableDataset, extractor: extr.Extractor, selection: extr.SelectionStrategy):
        previous_raise_config = tfm.raise_error_if_entry_not_extracted
        previous_extractor = dataset.extractor

        tfm.raise_error_if_entry_not_extracted = False
        dataset.extractor = extractor
        indices = select.calculate_or_load_indices(dataset, selection)

        tfm.raise_error_if_entry_not_extracted = previous_raise_config
        dataset.extractor = previous_extractor
        return indices


class BuildData:

    def __init__(self, build_dataset, build_loader=BuildDefaultLoader(), build_sampler=BuildDefaultSampler(), **kwargs) -> None:
        self.build_dataset = build_dataset
        self.build_loader = build_loader
        self.build_sampler = build_sampler
        self.kwargs = kwargs

    def __call__(self, data_config: cfg.DataConfiguration, **kwargs) -> Data:
        kwargs = {**self.kwargs, **kwargs}
        dataset = self.build_dataset(data_config, **kwargs)
        sampler = self.build_sampler(dataset, data_config, **kwargs)
        loader = self.build_loader(dataset, sampler, data_config, **kwargs)

        return Data(dataset=dataset, loader=loader)

