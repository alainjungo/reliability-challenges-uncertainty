import abc
import typing
import glob
import os

import pymia.data as data
import pymia.data.subjectfile as subj


class Collector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_subject_files(self) -> typing.List[data.SubjectFile]:
        pass


class Brats17Collector(Collector):

    def __init__(self, root_dir: str, with_grade=False, crop_brats_prefix=False) -> None:
        if root_dir.endswith('/'):
            root_dir = root_dir[:-1]
        self.root_dir = root_dir
        self.with_grade = with_grade
        self.crop_brats_prefix = crop_brats_prefix
        self.subject_files = []
        self._collect()

    def get_subject_files(self) -> typing.List[data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        flair_paths = glob.glob(self.root_dir + '/**/*_flair.nii.gz', recursive=True)
        t1_paths = glob.glob(self.root_dir + '/**/*_t1.nii.gz', recursive=True)
        t2_paths = glob.glob(self.root_dir + '/**/*_t2.nii.gz', recursive=True)
        t1c_paths = glob.glob(self.root_dir + '/**/*_t1ce.nii.gz', recursive=True)
        label_paths = glob.glob(self.root_dir + '/**/*_seg.nii.gz', recursive=True)

        flair_paths.sort()
        t1_paths.sort()
        t2_paths.sort()
        t1c_paths.sort()
        label_paths.sort()

        if not (len(flair_paths) == len(t1_paths) == len(t2_paths) == len(t1c_paths)):
            raise ValueError('all sequences must have same amount of files in the dataset')

        has_gt = len(label_paths) > 0
        if has_gt and len(flair_paths) != len(label_paths):
            raise ValueError('label must have same amount of files as other sequences')

        for subject_index in range(len(flair_paths)):
            subject_dir = os.path.dirname(flair_paths[subject_index])
            identifier = os.path.basename(subject_dir)
            if self.crop_brats_prefix:
                identifier = identifier[len('BratsXX_'):]
            if self.with_grade:
                grade = os.path.basename(os.path.dirname(subject_dir))
                identifier = '{}_{}'.format(identifier, grade)

            image_files = {'flair': flair_paths[subject_index],
                           't1': t1_paths[subject_index],
                           't2': t2_paths[subject_index],
                           't1c': t1c_paths[subject_index]}

            label_files = {}
            if has_gt:
                label_files['gt'] = label_paths[subject_index]

            sf = data.SubjectFile(identifier, images=image_files, labels=label_files)
            self.subject_files.append(sf)


class IsicCollector(Collector):
    LABEL_DIR_POST_FIX = '_Part1_GroundTruth'
    IMAGE_DIR_POST_FIX = '_Data'

    def __init__(self, root_dir_with_prefix: str, with_super_pixels=False) -> None:
        self.root_dir_with_prefix = root_dir_with_prefix
        self.with_super_pixels = with_super_pixels

        self.subject_files = []
        self._collect()

    def get_subject_files(self) -> typing.List[data.SubjectFile]:
        return self.subject_files

    def get_img_and_label_dirs(self):
        return (self.root_dir_with_prefix + IsicCollector.IMAGE_DIR_POST_FIX,
                self.root_dir_with_prefix + IsicCollector.LABEL_DIR_POST_FIX)

    def _collect(self):
        self.subject_files.clear()

        img_dir, label_dir = self.get_img_and_label_dirs()
        assert os.path.exists(img_dir)
        assert os.path.exists(label_dir)

        files_by_id = {}
        for file_path in glob.glob(img_dir + '/*') + glob.glob(label_dir + '/*'):
            base_name = os.path.basename(file_path)
            id_ = base_name[:12]

            if base_name.endswith('_superpixels.png'):
                files_by_id.setdefault(id_, {})['superpixel'] = file_path
            elif base_name.endswith('_segmentation.png'):
                files_by_id.setdefault(id_, {})['gt'] = file_path
            elif base_name.endswith('.jpg'):
                files_by_id.setdefault(id_, {})['image'] = file_path

        for id_, files in files_by_id.items():
            assert len(files) == 3, 'id "{}" has not 3 entries'.format(id_)

            params = {'images': {'image': files['image']}, 'labels': {'gt': files['gt']}}
            if self.with_super_pixels:
                params['misc'] = {'superpixel': files['superpixel']}
            sf = data.SubjectFile(id_, **params)
            self.subject_files.append(sf)


class PostfixPredictionCollector(Collector):

    def __init__(self, prediction_path: str, post_fixes: list, post_fix_categories: list = None) -> None:
        super().__init__()
        # if not prediction_path.endswith('/'):
        #     prediction_path = prediction_path + '/'
        self.prediction_path = prediction_path
        self.post_fixes = post_fixes
        if post_fix_categories is None:
            post_fix_categories = ['prediction' for _ in range(len(post_fixes))]
        if len(post_fix_categories) != len(post_fixes):
            raise ValueError('post_fix_categories argument must be of same length then post_fixes')
        self.post_fix_to_category = {pf: cat for pf, cat in zip(post_fixes, post_fix_categories)}

        self.subject_files = []
        self._collect()

    def get_subject_files(self) -> typing.List[data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        files_by_id = {}
        for post_fix in self.post_fixes:
            post_fix_paths = glob.glob(self.prediction_path + '/**/*_{}.nii.gz'.format(post_fix), recursive=True)

            for path_ in post_fix_paths:
                id_ = os.path.basename(path_)[:-len('_{}.nii.gz'.format(post_fix))]
                files_by_id.setdefault(id_, {})[post_fix] = path_

        for id_, files in files_by_id.items():
            assert set(files.keys()) == set(self.post_fixes), \
                'id "{}" has not all required entries "({})"'.format(id_, list(self.post_fixes))

            categories = {}
            for post_fix, category in self.post_fix_to_category.items():
                categories.setdefault(category, {})[post_fix] = files[post_fix]
            sf = data.SubjectFile(id_, **categories)
            self.subject_files.append(sf)


def combine(subject_files_from: typing.List[data.SubjectFile], subject_files_to: typing.List[data.SubjectFile]):
    sf_from_by_id = {sf.subject: sf for sf in subject_files_from}

    for sf_to in subject_files_to:
        sf_from = sf_from_by_id[sf_to.subject]  # type: data.SubjectFile

        for category in sf_from.categories:
            for k, v in sf_from.categories[category].entries.items():
                sf_to.categories.setdefault(category, subj.FileCategory()).entries[k] = v

    return subject_files_to
