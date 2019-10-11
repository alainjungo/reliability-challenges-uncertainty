import json
import operator

import numpy as np
import sklearn.model_selection as model_selection

import common.utils.filehelper as fh


def split_subjects(subjects: list, sizes: tuple) -> tuple:
    nb_total = len(subjects)
    counts = _normalize_sizes(sizes, nb_total)

    nb_train, nb_valid = counts[0], counts[1]
    train_subjects = subjects[:nb_train]
    valid_subjects = subjects[nb_train:nb_train + nb_valid]

    ret = [train_subjects, valid_subjects]
    with_test = len(counts) == 3
    if with_test:
        nb_test = counts[2]
        test_subjects = subjects[-nb_test:]
        ret.append(test_subjects)
    return tuple(ret)


def split_subjects_k_fold(subjects: list, k: int) -> list:
    no_subjects = len(subjects)
    if no_subjects % k != 0:
        raise ValueError('Number of subjects ({}) must be a multiple of k ({})'.format(no_subjects, k))

    subjects_per_fold = no_subjects // k

    splits = []
    for i in range(0, no_subjects, subjects_per_fold):
        valid_subjects = subjects[i:i + subjects_per_fold]
        train_subjects = subjects[0:i] + subjects[i + subjects_per_fold:]
        splits.append((train_subjects, valid_subjects))
    return splits


def split_subject_k_fold_stratified(subjects: list, stratification: list, k: int) -> list:
    # note: folds may not be of same size

    select = model_selection.StratifiedKFold(n_splits=k)

    folds = []
    for train_indices, valid_indices in select.split(subjects, stratification):
        train_names = operator.itemgetter(*train_indices)(subjects)
        valid_names = operator.itemgetter(*valid_indices)(subjects)
        folds.append((train_names, valid_names))

    return folds


def create_stratified_shuffled_split(subjects: list, stratification: list, counts: tuple, seed=100):

    valid_cnt = counts[1]
    res = model_selection.train_test_split(subjects, stratification, test_size=valid_cnt, random_state=seed,
                                           shuffle=True, stratify=np.asarray(stratification))
    tt_subjects, valid_subjects = res[:2]
    tt_stratification, _ = res[2:]

    if len(counts) == 3:
        test_cnt = counts[2]
        res = model_selection.train_test_split(tt_subjects, test_size=test_cnt, random_state=seed,
                                               shuffle=True, stratify=np.asarray(tt_stratification))
        train_subjects, test_subjects = res
        return train_subjects, valid_subjects, test_subjects
    else:
        train_subjects = tt_subjects
        return train_subjects, valid_subjects


def save_split(file: str, train_subjects: list, valid_subjects: list, test_subjects: list = None):
    fh.remove_if_exists(file)

    write_dict = {'train': train_subjects, 'valid': valid_subjects, 'test': test_subjects}

    with open(file, 'w') as f:
        json.dump(write_dict, f)


def load_split(file: str, k=None):
    with open(file, 'r') as f:
        read_dict = json.load(f)

    train_subjects, valid_subjects, test_subjects = read_dict['train'], read_dict['valid'], read_dict['test']
    if k is not None:
        train_subjects, valid_subjects = train_subjects[k], valid_subjects[k]
        test_subjects = [] if test_subjects is None else test_subjects[k]

    return train_subjects, valid_subjects, test_subjects


def _normalize_sizes(sizes, nb_total):
    if isinstance(sizes[0], int):
        if nb_total != sum(sizes):
            raise ValueError('int sizes ({}) do not sum to number of subjects ({})'.format(sizes, nb_total))
        nb_train = sizes[0]
        nb_valid = sizes[1]
    elif isinstance(sizes[0], float):
        if sum(sizes) != 1.0:
            raise ValueError('float sizes ({}) do not sum up to 1'.format(sizes))
        nb_train = int(nb_total * sizes[0])
        nb_valid = int(nb_total * sizes[1])
    else:
        raise ValueError('size values must be float or int, found {}'.format(type(sizes[0])))

    counts = [nb_train, nb_valid]

    with_test = len(sizes) == 3
    if with_test:
        nb_test = nb_total - nb_train - nb_valid
        counts.append(nb_test)

    return tuple(counts)
