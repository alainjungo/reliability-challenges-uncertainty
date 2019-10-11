import logging
import json
import zlib
import os

import pymia.data.extraction as extr


def save_indices(file_path: str, indices: list):
    config = {'indices': indices}
    with open(file_path, 'w') as f:
        json.dump(config, f)


def load_indices(file_path: str):
    with open(file_path, 'r') as f:
        config = json.load(f)
        return config['indices']


def calculate_or_load_indices(dataset: extr.ParameterizableDataset, selection: extr.SelectionStrategy):

    to_hash = os.path.basename(dataset.dataset_path) + ''.join(sorted(dataset.subject_subset)) + \
              repr(dataset.indexing_strategy) + repr(selection)
    crc32 = hex(zlib.crc32(bytes(to_hash, encoding='utf-8')) & 0xffffffff)

    indices_dir = os.path.join(os.path.dirname(dataset.dataset_path), 'indices')
    file_path = os.path.join(indices_dir, '{}.json'.format(crc32))
    if os.path.exists(file_path):
        return load_indices(file_path)

    logging.info('\t- need to calculate indices: {}'.format(repr(selection)))
    indices = extr.select_indices(dataset, selection)

    if not os.path.isdir(indices_dir):
        os.makedirs(indices_dir)

    save_indices(file_path, indices)
    logging.info('\t- written to file {}'.format(file_path))

    return indices



