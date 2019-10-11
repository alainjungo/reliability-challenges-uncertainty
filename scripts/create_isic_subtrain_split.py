import os
import argparse
import random

import utils.filehelper as fh
import common.data.collector as collect
import data.split as split
import rechun.directories as dirs


def main(split_type: str):

    if split_type not in ('ensemble', 'k-fold', 'resplit-train'):
        raise ValueError('invalid split type "{}"'.format(split_type))

    data_dir_with_prefix = dirs.ISIC_PREPROCESSED_TRAIN_DATA_DIR
    out_dir = dirs.SPLITS_DIR
    fh.create_dir_if_not_exists(out_dir)

    collector = collect.IsicCollector(data_dir_with_prefix, True)
    subject_files = collector.get_subject_files()
    subject_names = [sf.subject for sf in subject_files]

    # only training data that should be randomized
    train_names = subject_names

    if split_type == 'ensemble':

        k = 10
        splits = split.split_subjects_k_fold(train_names, k)
        train_names_k, _ = zip(*splits)
        valid_names_k = k * [None]
        test_names_k = None

        file_name = 'split_isic-train_k{}_{}-{}-{}.json'.format(k, len(train_names_k[0]), 0, 0)
        split.save_split(os.path.join(out_dir, file_name), train_names_k, valid_names_k, test_names_k)

    elif split_type == 'k-fold':
        k = 5

        splits = split.split_subjects_k_fold(train_names, k)
        train_names_k, valid_names_k = zip(*splits)

        nb_valid = int(len(train_names) / k)
        nb_train = len(train_names) - nb_valid

        file_name = 'split_isic_cv_k{}_{}-{}-{}.json'.format(k, nb_train, nb_valid, nb_valid)
        # valid is test, too for cross-validation
        split.save_split(os.path.join(out_dir, file_name), train_names_k, valid_names_k, valid_names_k)

    elif split_type == 'resplit-train':
        nb_train_percentage = 0.10
        nb_train_new = int(len(train_names)*nb_train_percentage)

        state = random.getstate()
        random.seed(100)
        random.shuffle(train_names)
        random.setstate(state)

        counts = (nb_train_new, len(train_names) - nb_train_new)
        new_train_names, _ = split.split_subjects(train_names, counts)

        file_name = 'split_isic_sub_{}-{}-{}.json'.format(nb_train_new, 0, 0)
        split.save_split(os.path.join(out_dir, file_name), new_train_names, [None], None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BraTS split file creation')
    parser.add_argument('--type', type=str, help='the split type to create')
    args = parser.parse_args()

    s_type = args.type
    if s_type is None:
        s_type = 'k-fold'

    print('\n**************************************')
    print('split type: {}'.format(s_type))
    print('**************************************\n')

    main(s_type)

