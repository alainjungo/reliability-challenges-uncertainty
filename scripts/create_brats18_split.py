import os
import argparse

import common.utils.filehelper as fh
import common.data.collector as collect
import common.data.split as split
import rechun.directories as dirs


def main(split_type: str):

    if split_type not in ('default', 'ensemble', 'k-fold', 'resplit-train'):
        raise ValueError('invalid split type "{}"'.format(split_type))

    data_dir = dirs.BRATS_ORIG_DATA_DIR
    out_dir = dirs.SPLITS_DIR
    fh.create_dir_if_not_exists(out_dir)

    collector = collect.Brats17Collector(data_dir)
    subject_files = collector.get_subject_files()
    subject_names = [sf.subject for sf in subject_files]

    def get_grade_from_subject_file(subject_file):
        first_image_path = list(subject_file.categories['images'].entries.values())[0]
        grade = os.path.basename(os.path.dirname(os.path.dirname(first_image_path)))
        return grade

    grades = [get_grade_from_subject_file(sf) for sf in subject_files]

    nb_train = 100
    nb_valid = 25
    nb_test = len(subject_names) - nb_train - nb_valid
    counts = (nb_train, nb_valid, nb_test)

    grade_ints = [0 if g == 'HGG' else 1 for g in grades]
    train_names, valid_names, test_names = split.create_stratified_shuffled_split(subject_names, grade_ints, counts,
                                                                                  seed=100)

    if split_type == 'default':
        file_name = 'split_brats18_{}-{}-{}.json'.format(nb_train, nb_valid, nb_test)
        split.save_split(os.path.join(out_dir, file_name), train_names, valid_names, test_names)
    elif split_type == 'ensemble':
        k = 10
        splits = split.split_subjects_k_fold(train_names, k)
        train_names_k, _ = zip(*splits)
        valid_names_k = k*[valid_names]
        test_names_k = k*[test_names]

        file_name = 'split_brats18_k{}_{}-{}-{}.json'.format(k, len(train_names)-k, nb_valid, nb_test)
        split.save_split(os.path.join(out_dir, file_name), train_names_k, valid_names_k, test_names_k)
    elif split_type == 'k-fold':
        k = 5

        name_grade_dict = {s: g for s, g in zip(subject_names, grade_ints)}

        to_fold = train_names + valid_names
        to_fold_grades = [name_grade_dict[s] for s in to_fold]

        splits = split.split_subject_k_fold_stratified(to_fold, to_fold_grades, k)
        train_names_k, valid_names_k = zip(*splits)

        nb_valid = int(len(to_fold) / k)
        nb_train = len(to_fold) - nb_valid

        file_name = 'split_brats18_cv_k{}_{}-{}-{}.json'.format(k, nb_train, nb_valid, nb_valid)
        # valid is test, too for cross-validation
        split.save_split(os.path.join(out_dir, file_name), train_names_k, valid_names_k, valid_names_k)
    elif split_type == 'resplit-train':
        # note: not stratified, since we want same subject to be include in a larger train_new split
        nb_train_new = 10

        counts = (nb_train_new, len(train_names) - nb_train_new)
        new_train_names, _ = split.split_subjects(train_names, counts)

        file_name = 'split_brats18_sub_{}-{}-{}.json'.format(nb_train_new, nb_valid, nb_test)
        split.save_split(os.path.join(out_dir, file_name), new_train_names, valid_names, test_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BraTS split file creation')
    parser.add_argument('--type', type=str, help='the split type to create')
    args = parser.parse_args()

    s_type = args.type
    if s_type is None:
        s_type = 'default'

    print('\n**************************************')
    print('split type: {}'.format(s_type))
    print('**************************************\n')

    main(s_type)
