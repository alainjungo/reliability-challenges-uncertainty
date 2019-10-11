import typing
import os
import argparse
import glob

import SimpleITK as sitk
import numpy as np
import pymia.data.creation as crt
import pymia.data.transformation as tfm
import pymia.data.conversion as conv
import pymia.data.subjectfile as subj
import pymia.data.indexexpression as expr
import pymia.data.creation.writer as wr

import common.utils.filehelper as fh
import common.data.collector as collect
import common.data.split as split
import rechun.directories as dirs


class LoadSubject(crt.Load):

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:

        if category == 'images':
            img = sitk.ReadImage(file_name, sitk.sitkFloat32)  # load as float32 (otherwise is int)
            return sitk.GetArrayFromImage(img), conv.ImageProperties(img)
        else:
            img = sitk.ReadImage(file_name, sitk.sitkUInt8)
            return sitk.GetArrayFromImage(img), conv.ImageProperties(img)


def to_binary(arr: np.ndarray):
    arr[arr != 0] = 1
    return arr


class BuildParameters:

    def __init__(self, in_dir: str, out_file: str, transforms: list, split_file: str = None, is_train_data=True,
                 prediction_path: str = None, add_prediction_fn = None) -> None:
        super().__init__()
        self.in_dir = in_dir
        self.out_file = out_file
        self.transforms = transforms
        self.split_file = split_file
        self.is_train_data = is_train_data
        self.prediction_path = prediction_path
        self.add_prediction_fn = add_prediction_fn


def get_params(out_file: str, in_dir=dirs.BRATS_ORIG_DATA_DIR,
               split_file=os.path.join(dirs.SPLITS_DIR, 'split_brats18_100-25-160.json'), is_train_data=True,
               prediction_path=None, label_fn=None, add_prediction_fn = None):

    if add_prediction_fn is None:
        add_prediction_fn = add_predictions

    params = BuildParameters(
        in_dir=in_dir,
        out_file=out_file,
        transforms=[tfm.IntensityNormalization(loop_axis=-1)],
        split_file=split_file,
        is_train_data=is_train_data,
        prediction_path =prediction_path,
        add_prediction_fn = add_prediction_fn
    )

    if label_fn is not None:
        params.transforms.append(tfm.LambdaTransform(label_fn, entries=('labels',)))
    return params


def main(creation_type):

    if creation_type not in ('train', 'test', 'train_with_predictions', 'test_with_predictions'):
        raise ValueError('invalid cration type "{}"'.format(creation_type))

    if creation_type == 'train':
        build_binarized_train()
    elif creation_type == 'test':
        build_binarized_test()
    elif creation_type == 'train_with_predictions':
        build_binarized_with_predictions_train()
    elif creation_type == 'test_with_predictions':
        build_binarized_with_prediction_test()


def build_binarized_with_prediction_test():
    params = get_params(
        os.path.join(dirs.DATASET_DIR, 'brats18_test_wpred_reduced_norm.h5'),
        is_train_data=False,
        label_fn=to_binary,
        prediction_path=os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_BASELINE_PREDICT)
    )
    build_brats_dataset(params)


def build_binarized_with_predictions_train():
    params = get_params(
        os.path.join(dirs.DATASET_DIR, 'brats18_train_wpred_reduced_norm.h5'),
        label_fn=to_binary,
        prediction_path=os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_CV_PREDICT)
    )
    build_brats_dataset(params)


def build_binarized_train():
    params = get_params(
        os.path.join(dirs.DATASET_DIR, 'brats18_train_reduced_norm.h5'),
        label_fn=to_binary,
    )
    build_brats_dataset(params)


def build_binarized_test():
    params = get_params(
        os.path.join(dirs.DATASET_DIR, 'brats18_test_reduced_norm.h5'),
        is_train_data=False,
        label_fn=to_binary,
    )
    build_brats_dataset(params)


def build_brats_dataset(params: BuildParameters):
    collector = collect.Brats17Collector(params.in_dir)  # 17 is same dataset as 18
    subject_files = collector.get_subject_files()

    if params.split_file is not None:
        if params.is_train_data:
            train_subjects, valid_subjects, _ = split.load_split(params.split_file)
            selection = train_subjects + valid_subjects
        else:
            _, _, selection = split.load_split(params.split_file)

        subject_files = [sf for sf in subject_files if sf.subject in selection]
        assert len(subject_files) == len(selection)

    # sort the subject files according to the subject name (identifier)
    subject_files.sort(key=lambda sf: sf.subject)

    if params.prediction_path is not None:
        subject_files = params.add_prediction_fn(subject_files, params.prediction_path)

    fh.create_dir_if_not_exists(params.out_file, is_file=True)
    fh.remove_if_exists(params.out_file)

    with crt.get_writer(params.out_file) as writer:
        callbacks = [crt.MonitoringCallback(), crt.WriteNamesCallback(writer), crt.WriteFilesCallback(writer),
                     crt.WriteDataCallback(writer), crt.WriteSubjectCallback(writer),
                     crt.WriteImageInformationCallback(writer),
                     ]

        has_grade = params.in_dir.endswith('Training')
        if has_grade:
            callbacks.append(WriteGradeCallback(writer))
        callback = crt.ComposeCallback(callbacks)

        traverser = crt.SubjectFileTraverser()
        traverser.traverse(subject_files, callback=callback, load=LoadSubject(),
                           transform=tfm.ComposeTransform(params.transforms))


def add_predictions(subject_files: list, prediction_path: str):
    paths = glob.glob(prediction_path + '/*_prediction.nii.gz')
    paths = [os.path.abspath(p) for p in paths]

    prediction_dict = {os.path.basename(p)[:-len('_prediction.nii.gz')]: p for p in paths}

    assert set(s.subject for s in subject_files) == set(prediction_dict.keys())

    for sf in subject_files:
        corresponding_pred_path = prediction_dict[sf.subject]
        sf.categories['labels'].entries['prediction'] = corresponding_pred_path
    return subject_files


class WriteGradeCallback(crt.Callback):

    def __init__(self,  writer: wr.Writer) -> None:
        super().__init__()
        self.writer = writer

    def on_start(self, params: dict):
        subject_count = len(params['subject_files'])
        self.writer.reserve('meta/grades', (subject_count,), dtype=str)

    def on_subject(self, params: dict):
        subject_files = params['subject_files']
        subject_index = params['subject_index']

        subject_file = subject_files[subject_index]  # type: subj.SubjectFile
        first_image_path = list(subject_file.categories['images'].entries.values())[0]
        grade_str = os.path.basename(os.path.dirname(os.path.dirname(first_image_path)))
        self.writer.fill('meta/grades', grade_str, expr.IndexExpression(subject_index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BraTS h5 dataset creation')
    parser.add_argument('--type', type=str, help='the creation type')
    args = parser.parse_args()

    c_type = args.type
    if c_type is None:
        c_type = 'train'

    print('\n**************************************')
    print('split type: {}'.format(c_type))
    print('**************************************\n')

    main(c_type)
