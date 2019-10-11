import abc
import os

import SimpleITK as sitk
import numpy as np
import pymia.data.conversion as conversion

import common.evalutation.numpyfunctions as np_fn
import common.utils.labelhelper as lh
import rechun.eval.helper as helper
import rechun.eval.evaldata as evdata
import rechun.directories as dirs


class Loader:

    class Params:

        def __init__(self, misc_entry='probabilities', need_target=True, need_prediction=True, need_t2_mask=False,
                     need_prediction_dist_and_boarder=False, need_gt_dist_and_boarder=False, images_needed: list=None,
                     need_img_props=False) -> None:
            super().__init__()
            self.misc_entry = misc_entry
            self.need_target = need_target
            self.need_prediction = need_prediction
            self.need_t2_mask = need_t2_mask
            self.need_gt_dist_and_boarder = need_gt_dist_and_boarder
            self.need_prediction_dist_and_boarder = need_prediction_dist_and_boarder
            self.images_needed = images_needed
            self.need_img_props = need_img_props

    def __init__(self) -> None:
        super().__init__()
        self.cached_entries = {}
        self.cached_subject_id = None

    def get_data(self, subject_file, params: Params):
        if subject_file.subject != self.cached_subject_id:
            self.cached_entries.clear()
            self.cached_subject_id = subject_file.subject

        to_eval = {}
        misc_np, props = self._get_misc_entry(subject_file, params.misc_entry, 'img_properties')
        to_eval[params.misc_entry] = misc_np
        if params.need_img_props:
            to_eval['img_properties'] = props

        if params.need_target:
            to_eval['target'] = self._get_target(subject_file, 'target')

        if params.need_prediction:
            to_eval['prediction'] = self._get_prediction(subject_file, 'prediction')

        if params.need_gt_dist_and_boarder:
            mask, distance = self._get_dist_and_boarder(subject_file, 'target_boarder', 'target_distance',
                                                        'target')
            to_eval['target_boarder'] = mask
            to_eval['target_distance'] = distance

        if params.need_prediction_dist_and_boarder:
            mask, distance = self._get_dist_and_boarder(subject_file, 'prediction_boarder', 'prediction_distance',
                                                        'prediction')
            to_eval['prediction_boarder'] = mask
            to_eval['prediction_distance'] = distance

        if params.need_t2_mask:
            to_eval['mask'] = self._get_t2_mask(subject_file, 'mask')

        if params.images_needed:
            for image_type in params.images_needed:
                to_eval[image_type] = self._get_image(subject_file, image_type)

        return to_eval

    def _get_misc_entry(self, subject_file, entry: str, property_entry: str):
        if entry in self.cached_entries:
            return self.cached_entries[entry].copy(), self.cached_entries[property_entry] # copy just to be sure that is hadn't been modified
        file_path = subject_file.categories['misc'].entries[entry]
        np_misc, props = conversion.SimpleITKNumpyImageBridge.convert(sitk.ReadImage(file_path))
        self.cached_entries[entry] = np_misc
        self.cached_entries[property_entry] = props
        return self.cached_entries[entry], self.cached_entries[property_entry]

    def _get_target(self, subject_file, entry):
        if entry in self.cached_entries:
            return self.cached_entries[entry].copy() # copy just to be sure that is hadn't been modified
        file_path = subject_file.categories['labels'].entries['gt']
        target_np = sitk.GetArrayFromImage(sitk.ReadImage(file_path, sitk.sitkUInt8))
        target_np[target_np > 0] = 1  # the labels are 0 to 4 but we only do 0 and 1
        self.cached_entries[entry] = target_np
        return target_np

    def _get_image(self, subject_file, entry):
        if entry in self.cached_entries:
            return self.cached_entries[entry].copy() # copy just to be sure that is hadn't been modified
        file_path = subject_file.categories['images'].entries[entry]
        image_np = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        self.cached_entries[entry] = image_np
        return image_np

    def _get_prediction(self, subject_file, entry):
        if entry in self.cached_entries:
            return self.cached_entries[entry].copy()
        file_path = subject_file.categories['labels'].entries[entry]
        prediction_np = sitk.GetArrayFromImage(sitk.ReadImage(file_path, sitk.sitkUInt8))
        self.cached_entries[entry] = prediction_np
        return prediction_np

    def _get_dist_and_boarder(self, subject_file, boarder_entry, dist_entry, prediction_entry):
        if boarder_entry in self.cached_entries and dist_entry in self.cached_entries:
            return self.cached_entries[boarder_entry].copy(), self.cached_entries[dist_entry].copy()
        prediction_np = self._get_prediction(subject_file, prediction_entry)
        distance, mask = lh.boarder_mask(prediction_np.astype(np.bool), distance_in=1, distance_out=1)
        self.cached_entries[boarder_entry] = mask
        self.cached_entries[dist_entry] = distance
        return mask, distance

    def _get_t2_mask(self, subject_file, entry):
        if entry in self.cached_entries:
            return self.cached_entries[entry].copy()  # copy just to be sure that is hadn't been modified
        file_path = subject_file.categories['images'].entries['t2']
        t2_np = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        mask_np = t2_np > 0
        self.cached_entries[entry] = mask_np
        return mask_np


class PrepareData(abc.ABC):

    @abc.abstractmethod
    def __call__(self, to_eval: dict) -> dict:
        pass


class ComposePreparation(PrepareData):

    def __init__(self, prepare_data_list: list) -> None:
        super().__init__()
        self.prepare_data_list = prepare_data_list

    def __call__(self, to_eval: dict) -> dict:
        for prepare_data in self.prepare_data_list:
            to_eval = prepare_data(to_eval)
        return to_eval


class AddBackgroundProbabilities(PrepareData):

    def __call__(self, to_eval: dict) -> dict:
        to_eval['probabilities'] = helper.add_background_probability(to_eval['probabilities'])
        return to_eval


class RescaleLinear(PrepareData):

    def __init__(self, entry: str, min_: float, max_: float, epsilon=1e-5) -> None:
        self.entry = entry
        self.min = min_
        self.max = max_
        self.epsilon = epsilon  # epsilon is used to have probs != 0 or 1

    def __call__(self, to_eval: dict) -> dict:
        prob_np = helper.rescale_uncertainties(to_eval[self.entry], self.min, self.max, self.epsilon)
        to_eval[self.entry] = prob_np
        return to_eval


class RescaleSubjectMinMax(PrepareData):

    def __init__(self, entry: str, epsilon=1e-5) -> None:
        self.entry = entry
        self.epsilon = epsilon  # epsilon is used to have probs != 0 or 1

    def __call__(self, to_eval: dict) -> dict:
        entry_np = to_eval[self.entry]
        prob_np = helper.rescale_uncertainties(entry_np, entry_np.min(), entry_np.max(), self.epsilon)
        to_eval[self.entry] = prob_np
        return to_eval


class ToForegroundProbabilities(PrepareData):

    def __call__(self, to_eval: dict) -> dict:
        prob_np = helper.uncertainty_to_foreground_probabilities(to_eval['probabilities'], to_eval['prediction'])
        to_eval['probabilities'] = prob_np
        return to_eval


class ToEntropy(PrepareData):

    def __init__(self, entropy_entry='uncertainty') -> None:
        super().__init__()
        self.nb_classes = 2  # everything is binary until now
        self.entropy_entry = entropy_entry

    def __call__(self, to_eval: dict) -> dict:
        prob_np = to_eval['probabilities']
        if prob_np.shape[-1] != self.nb_classes:
            raise ValueError('last dimension of probability array ({}) must be equal to nb_classes ({})'
                             .format(prob_np.shape, self.nb_classes))
        to_eval[self.entropy_entry] = np_fn.entropy(prob_np) / np.log(self.nb_classes)
        helper.check_min_max(to_eval[self.entropy_entry], only_warn=True)
        return to_eval


class MoveEntry(PrepareData):

    def __init__(self, from_entry: str, to_entry: str) -> None:
        super().__init__()
        self.from_entry = from_entry
        self.to_entry = to_entry

    def __call__(self, to_eval: dict) -> dict:
        to_eval[self.to_entry] = to_eval[self.from_entry]
        return to_eval


def get_probability_preparation(eval_data: evdata.EvalData, rescale_confidence='subject', rescale_sigma='subject',
                                min_max_dir: str = None):
    prepare = []

    if eval_data.confidence_entry == 'probabilities':
        prepare.append(AddBackgroundProbabilities())
        return ComposePreparation(prepare), eval_data.id_
    if eval_data.confidence_entry == 'confidence':
        id_ = eval_data.id_
        prep, prep_id = _get_rescale_prep_and_idstr(eval_data, rescale_confidence, min_max_dir)
        if prep is not None:
            prepare.append(prep)
            id_ += prep_id
        prepare.extend([
            MoveEntry(eval_data.confidence_entry, 'probabilities'),
            ToForegroundProbabilities(),
            AddBackgroundProbabilities()
        ])
        return ComposePreparation(prepare), id_
    # if sigma or log-var
    id_ = eval_data.id_
    prep, prep_id = _get_rescale_prep_and_idstr(eval_data, rescale_sigma, min_max_dir)
    if prep is not None:
        prepare.append(prep)
        id_ += prep_id
    prepare.extend([
        MoveEntry(eval_data.confidence_entry, 'probabilities'),
        ToForegroundProbabilities(),
        AddBackgroundProbabilities()
    ])
    return ComposePreparation(prepare), id_


def get_uncertainty_preparation(eval_data: evdata.EvalData, rescale_confidence='', rescale_sigma='global',
                                min_max_dir: str = None):
    prepare = []

    if eval_data.confidence_entry == 'probabilities':
        prepare.append(AddBackgroundProbabilities())
        prepare.append(ToEntropy())
        return ComposePreparation(prepare), eval_data.id_
    if eval_data.confidence_entry == 'confidence':
        id_ = eval_data.id_
        prep, prep_id = _get_rescale_prep_and_idstr(eval_data, rescale_confidence, min_max_dir)
        if prep is not None:
            prepare.append(prep)
            id_ += prep_id
        prepare.append(MoveEntry(eval_data.confidence_entry, 'uncertainty'))
        return ComposePreparation(prepare), id_
    # sigma or log-var
    id_ = eval_data.id_
    prep, prep_id = _get_rescale_prep_and_idstr(eval_data, rescale_sigma, min_max_dir)
    if prep is not None:
        prepare.append(prep)
        id_ += prep_id
    prepare.append(MoveEntry(eval_data.confidence_entry, 'uncertainty'))
    return ComposePreparation(prepare), id_


def _get_rescale_prep_and_idstr(eval_data: evdata.EvalData, rescale_type: str, min_max_dir: str = None):
    if rescale_type == 'global':
        min_max_path = os.path.join(min_max_dir, dirs.MINMAX_PLACEHOLDER.format(eval_data.id_))
        min_, max_ = helper.read_min_max(min_max_path)
        return RescaleLinear(eval_data.confidence_entry, min_, max_), '_globalrescale'
    elif rescale_type == 'subject':
        return RescaleSubjectMinMax(eval_data.confidence_entry), '_rescale'
    else:
        return None, ''


def get_confidence_entry_preparation(eval_data: evdata.EvalData, to_entry):
    if eval_data.confidence_entry == 'probabilities':
        return MoveEntry('probabilities', to_entry), eval_data.id_
    if eval_data.confidence_entry == 'confidence':
        return MoveEntry(eval_data.confidence_entry, to_entry), eval_data.id_
    # sigma or log-var
    return MoveEntry(eval_data.confidence_entry, to_entry), eval_data.id_
