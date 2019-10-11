from PIL import Image
import SimpleITK as sitk
import numpy as np
import torch.utils.data as torch_data

import common.data.collector as collect
import common.trainloop.factory as factory
import common.trainloop.config as cfg
import common.trainloop.data as data


class IsicDataset(torch_data.Dataset):

    LABEL_DIR_POST_FIX = '_Part1_GroundTruth'
    IMAGE_DIR_POST_FIX = '_Data'

    def __init__(self, data_dir_with_task_prefix: str, transform=None, with_super_pixels=False,
                 with_file_paths=True, subject_subset: list=None, prediction_subject_files: list=None) -> None:
        super().__init__()
        self.data_dir_with_task_prefix = data_dir_with_task_prefix
        self.transform = transform
        self.with_super_pixels = with_super_pixels
        self.with_file_paths = with_file_paths
        self.subject_files_by_id = {}
        self.ids = []
        self.with_predictions = prediction_subject_files is not None
        self._collect(subject_subset, prediction_subject_files)

    def _collect(self, subject_subset: list=None, prediction_subject_files=None):
        collector = collect.IsicCollector(self.data_dir_with_task_prefix, with_super_pixels=True)
        subject_files = collector.get_subject_files()

        if prediction_subject_files is not None:
            subject_files = collect.combine(prediction_subject_files, subject_files)
            assert set([sf.subject for sf in prediction_subject_files]) == set([sf.subject for sf in subject_files])

        if subject_subset is not None:
            self.subject_files_by_id = {sf.subject: sf for sf in subject_files if sf.subject in subject_subset}
        else:
            self.subject_files_by_id = {sf.subject: sf for sf in subject_files}

        self.ids = list(self.subject_files_by_id.keys())
        self.ids.sort()

    def __getitem__(self, index):
        id_ = self.ids[index]
        files = self.subject_files_by_id[id_].get_all_files()

        sample = {'ids': id_}

        label = np.asarray(Image.open(files['gt']).convert('L'))[..., np.newaxis].astype(np.uint8)
        label.setflags(write=True)
        sample['labels'] = label

        image = np.asarray(Image.open(files['image'])).astype(np.float32)
        image.setflags(write=True)
        sample['images'] = image

        if self.with_super_pixels:
            superpixel = np.asarray(Image.open(files['superpixel']).convert('L'))[..., np.newaxis].astype(np.uint8)
            superpixel.setflags(write=True)
            sample['superpixels'] = superpixel

        if self.with_predictions:
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(files['prediction']))
            # labels max is 255
            prediction = prediction*255
            labels = np.concatenate((sample['labels'], prediction[..., np.newaxis]), axis=-1)
            sample['labels'] = labels

        if self.with_file_paths:
            self._add_files_to_sampler_sample(files, sample)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_files_by_id(self, id_: str):
        files = self.subject_files_by_id[id_].get_all_files()
        file_dict = {}
        self._add_files_to_sampler_sample(files, file_dict)
        return file_dict

    def _add_files_to_sampler_sample(self, files: dict, sample: dict):
        sample['image_paths'] = files['image']
        sample['label_paths'] = files['gt']
        if self.with_super_pixels:
            sample['superpixel_paths'] = files['superpixel']

    def __len__(self):
        return len(self.ids)

    def get_img_and_label_dirs(self):
        return (self.data_dir_with_task_prefix + IsicDataset.IMAGE_DIR_POST_FIX,
                self.data_dir_with_task_prefix + IsicDataset.LABEL_DIR_POST_FIX)


class BuildIsicDataset(data.BuildDataset):
    def __call__(self, data_config: cfg.DataConfiguration, **kwargs):
        transform = factory.get_transform(data_config.transform)
        subset = None
        if 'entries' in kwargs:
            subset = kwargs['entries']

        prediction_subject_files = None
        if 'prediction_dir' in kwargs:
            post_fixes = ['prediction']
            cats = ['labels']
            prediction_collector = collect.PostfixPredictionCollector(kwargs['prediction_dir'], post_fixes, cats)
            prediction_subject_files = prediction_collector.get_subject_files()
        return IsicDataset(data_config.dataset, transform=transform, with_super_pixels=True, subject_subset=subset,
                           prediction_subject_files=prediction_subject_files)
