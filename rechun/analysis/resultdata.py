import abc
import re
import os
import glob
import warnings

import rechun.directories as dirs

brats_selection_id_name_dict = {
    'baseline': 'baseline',
    'center': 'center',
    'baseline_mc': 'baseline+MC',
    'center_mc': 'center+MC',
    'ensemble': 'ensemble',
    'auxiliary_feat_rescale': 'auxiliary feat.',
    'auxiliary_segm_rescale': 'auxiliary segm.',
    'aleatoric_globalrescale': 'aleatoric',
}

isic_selection_id_name_dict = {
    'baseline': 'baseline',
    'center': 'center',
    'baseline_mc': 'baseline+MC',
    'center_mc': 'center+MC',
    'ensemble': 'ensemble',
    'auxiliary_feat_rescale': 'auxiliary feat.',
    'auxiliary_segm_rescale': 'auxiliary segm.',
    'aleatoric_globalrescale': 'aleatoric',
}


def _place_holder_to_regex(place_holder: str):
    return place_holder.replace('{}', '(.*)')


def _place_holder_to_glob(place_holder: str):
    return place_holder.replace('{}', '*')


def _get_files_in_dir(base_dir: str, dir_name: str, placeholder: str):
    directory = os.path.join(base_dir, dir_name)
    has_threshold = placeholder.count('{}') == 2

    if not directory.endswith('/'):
        directory += '/'
    all_files = glob.glob(directory + _place_holder_to_glob(placeholder))

    id_file_dict = {}
    for file_ in all_files:
        basename = os.path.basename(file_)
        m = re.match(_place_holder_to_regex(placeholder), basename)

        if has_threshold:
            id_, threshold = m.group(1), m.group(2)
            id_file_dict.setdefault(id_, {})[threshold] = file_
        else:
            id_ = m.group(1)
            id_file_dict[id_] = file_

    return id_file_dict


def _combine_file_dicts(category_dict_: dict):
    id_category_dict_ = {}
    for category, id_file_dict in category_dict_.items():
        for id_, file_ in id_file_dict.items():
            id_category_dict_.setdefault(id_, {})[category] = file_
    return id_category_dict_


class ResultsData(abc.ABC):

    def __init__(self, base_dir) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.category_dict = self._get_cat_dict()
        self.id_category_dict = _combine_file_dicts(self.category_dict)

    @abc.abstractmethod
    def _get_cat_dict(self) -> dict:
        pass

    def get_files(self, ids: list, categories: list, thresholds: list = None):
        files_and_info = []

        for id_ in ids:
            if id_ not in self.id_category_dict:
                warnings.warn('missing id "{}"'.format(id_))
                continue

            for category in categories:
                if category not in self.id_category_dict[id_]:
                    warnings.warn('missing category "{}" in id "{}"'.format(category, id_))
                    continue

                file_ = self.id_category_dict[id_][category]
                if isinstance(file_, dict):
                    # has thresholds
                    if thresholds is not None:
                        for threshold in thresholds:
                            if threshold not in file_:
                                warnings.warn(
                                    'missing threshold "{}" in category "{}" in id "{}"'.format(threshold, category,
                                                                                                id_))
                                continue
                            files_and_info.append((file_[threshold], id_, category, threshold))
                    else:
                        for threshold, f in file_.items():
                            files_and_info.append((f, id_, category, threshold))
                else:
                    # has not thresholds
                    files_and_info.append((file_, id_, category, None))

        files, file_ids, file_categories, file_thresholds = zip(*files_and_info)
        return files, file_ids, file_categories, file_thresholds

    def get_ids(self):
        return list(self.id_category_dict.keys())

    def get_categories(self,):
        return list(self.category_dict.keys())


class BratsResultData(ResultsData):

    def __init__(self, base_dir=dirs.BRATS_EVAL_DIR) -> None:
        super().__init__(base_dir)

    def _get_cat_dict(self) -> dict:
        return {
            dirs.CALIB_NAME: _get_files_in_dir(self.base_dir, dirs.CALIB_NAME, dirs.CALIBRATION_PLACEHOLDER),
            dirs.ECE_FOREGROUND_NAME: _get_files_in_dir(self.base_dir, dirs.ECE_FOREGROUND_NAME, dirs.ECE_PLACEHOLDER),
            dirs.MINMAX_NAME: _get_files_in_dir(self.base_dir, dirs.MINMAX_NAME, dirs.MINMAX_PLACEHOLDER),
            dirs.UNCERTAINTY_NAME: _get_files_in_dir(self.base_dir, dirs.UNCERTAINTY_NAME, dirs.UNCERTAINTY_PLACEHOLDER),
        }


class IsicResultData(ResultsData):

    def __init__(self, base_dir=dirs.ISIC_EVAL_DIR) -> None:
        super().__init__(base_dir)

    def _get_cat_dict(self) -> dict:
        return {
            dirs.CALIB_NAME: _get_files_in_dir(self.base_dir, dirs.CALIB_NAME, dirs.CALIBRATION_PLACEHOLDER),
            dirs.ECE_NAME: _get_files_in_dir(self.base_dir, dirs.ECE_NAME, dirs.ECE_PLACEHOLDER),
            dirs.MINMAX_NAME: _get_files_in_dir(self.base_dir, dirs.MINMAX_NAME, dirs.MINMAX_PLACEHOLDER),
            dirs.UNCERTAINTY_NAME: _get_files_in_dir(self.base_dir, dirs.UNCERTAINTY_NAME, dirs.UNCERTAINTY_PLACEHOLDER),
        }
