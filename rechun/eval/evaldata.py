import os
import typing

import common.data.split as split
import common.data.collector as collect
import rechun.directories as dirs


class EvalData:

    def __init__(self, id_, eval_path, confidence_entry='probabilities', subject_files=None) -> None:
        super().__init__()
        self.id_ = id_
        self.eval_path = eval_path
        self.confidence_entry = confidence_entry
        if subject_files is None:
            subject_files = []
        self.subject_files = subject_files


brats_eval_data = {
    'baseline': EvalData('baseline', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_BASELINE_PREDICT)),
    'baseline_mc': EvalData('baseline_mc', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_BASELINE_MC_PREDICT)),
    'center': EvalData('center', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_CENTER_PREDICT)),
    'center_mc': EvalData('center_mc', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_CENTER_MC_PREDICT)),
    'ensemble': EvalData('ensemble', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_ENSEMBLE_PREDICT)),
    'auxiliary_feat': EvalData('auxiliary_feat', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_AUX_FEAT_PREDICT), 'confidence'),
    'auxiliary_segm': EvalData('auxiliary_segm', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_AUX_SEGM_PREDICT), 'confidence'),
    'aleatoric': EvalData('aleatoric', os.path.join(dirs.BRATS_PREDICT_DIR, dirs.BRATS_ALEATORIC_PREDICT), 'sigma')
}


def get_brats_eval_data(to_eval: list):
    eval_data = [brats_eval_data[e] for e in to_eval]
    return get_brats_data(eval_data)


isic_eval_data = {
    'baseline': EvalData('baseline', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_BASELINE_PREDICT)),
    'baseline_mc': EvalData('baseline_mc', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_BASELINE_MC_PREDICT)),
    'center': EvalData('center', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_CENTER_PREDICT)),
    'center_mc': EvalData('center_mc', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_CENTER_MC_PREDICT)),
    'ensemble': EvalData('ensemble', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_ENSEMBLE_PREDICT)),
    'auxiliary_feat': EvalData('auxiliary_feat', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_AUX_FEAT_PREDICT), 'confidence'),
    'auxiliary_segm': EvalData('auxiliary_segm', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_AUX_SEGM_PREDICT), 'confidence'),
    'aleatoric': EvalData('aleatoric', os.path.join(dirs.ISIC_PREDICT_DIR, dirs.ISIC_ALEATORIC_PREDICT), 'sigma')
}


def get_isic_eval_data(to_eval: list):
    eval_data = [isic_eval_data[e] for e in to_eval]
    return get_isic_data(eval_data)


def get_brats_data(eval_data: typing.Union[EvalData, list], in_dir=dirs.BRATS_ORIG_DATA_DIR,
                   split_file=os.path.join(dirs.SPLITS_DIR, 'split_brats18_100-25-160.json')):
    was_list = True
    if isinstance(eval_data, EvalData):
        was_list = False
        eval_data = [eval_data]

    # get all file path of the training set
    collector = collect.Brats17Collector(in_dir)  # 17 is same dataset as 18
    gt_subject_files = collector.get_subject_files()

    _, _, test_subjects = split.load_split(split_file)

    for entry in eval_data:
        post_fixes = ['prediction', entry.confidence_entry]
        cats = ['labels', 'misc']
        prediction_collector = collect.PostfixPredictionCollector(entry.eval_path, post_fixes, cats)
        prediction_subject_files = prediction_collector.get_subject_files()

        prediction_subject_files = collect.combine(gt_subject_files, prediction_subject_files)
        assert set(test_subjects) == set([sf.subject for sf in prediction_subject_files])

        entry.subject_files = prediction_subject_files

    return eval_data if was_list else eval_data[0]


def get_isic_data(eval_data: typing.Union[EvalData, list],
                  in_dir=dirs.ISIC_PREPROCESSED_TEST_DATA_DIR):
    was_list = True
    if isinstance(eval_data, EvalData):
        was_list = False
        eval_data = [eval_data]

    collector = collect.IsicCollector(in_dir)
    gt_subject_files = collector.get_subject_files()

    for entry in eval_data:
        post_fixes = ['prediction', entry.confidence_entry]
        cats = ['labels', 'misc']
        prediction_collector = collect.PostfixPredictionCollector(entry.eval_path, post_fixes, cats)
        prediction_subject_files = prediction_collector.get_subject_files()

        prediction_subject_files = collect.combine(gt_subject_files, prediction_subject_files)
        assert set([sf.subject for sf in gt_subject_files]) == set([sf.subject for sf in prediction_subject_files])

        entry.subject_files = prediction_subject_files

    return eval_data if was_list else eval_data[0]
