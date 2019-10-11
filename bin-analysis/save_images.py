import os
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar

import rechun.eval.evaldata as evdata
import rechun.eval.analysis as analysis
import rechun.directories as dirs
import common.utils.filehelper as fh


def main(dataset: str, to_plot: list):

    if dataset not in ('brats', 'isic'):
        raise ValueError('Invalid dataset "{}". Chose "brats" or "isic"'.format(dataset))
    task = dataset

    if task == 'brats':
        eval_data_list = evdata.get_brats_eval_data(to_plot)
        subjects = ['Brats18_TCIA01_390_1', 'Brats18_CBICA_AUN_1', 'Brats18_CBICA_ASY_1']
        min_max_dir = os.path.join(dirs.BRATS_EVAL_DIR, dirs.MINMAX_NAME)
        plot_dir = os.path.join(dirs.BRATS_PLOT_DIR, 'images')
        img_key = 'flair'
    else:
        eval_data_list = evdata.get_isic_eval_data(to_plot)
        subjects = ['ISIC_0012388', 'ISIC_0012654', 'ISIC_0012447']
        min_max_dir = os.path.join(dirs.ISIC_EVAL_DIR, dirs.MINMAX_NAME)
        plot_dir = os.path.join(dirs.ISIC_PLOT_DIR, 'images')
        img_key = 'image'

    fh.create_dir_if_not_exists(plot_dir)
    writer = OutWriterPng(plot_dir, task, img_key)

    for entry in eval_data_list:
        prepare, id_ = analysis.get_uncertainty_preparation(entry, rescale_confidence='subject', rescale_sigma='global',
                                                            min_max_dir=min_max_dir)
        print(id_)
        subject_files = [sf for sf in entry.subject_files if sf.subject in subjects]
        for sf in subject_files:
            subject_dir = os.path.join(plot_dir, sf.subject)
            if not os.path.isdir(subject_dir):
                os.makedirs(subject_dir)

            loader = analysis.Loader()
            d = loader.get_data(sf, analysis.Loader.Params(entry.confidence_entry, need_target=True,
                                                           need_prediction=True, images_needed=[img_key]))
            d = prepare(d)

            writer.on_new_subject(sf.subject, d)
            writer.on_test_id(entry.id_, d)


def get_slice_and_str(data: dict, task: str):
    if task == 'isic':
        return slice(None)
    else:
        max_gt_slice = np.argmax(data['target'].sum(axis=(1, 2)))
        return max_gt_slice


class OutWriterPng:

    def __init__(self, out_dir: str, task: str, img_key, get_slice_and_str_fn=get_slice_and_str) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.task = task
        self.img_key = img_key
        self.subject_dir = None
        self.selected_slice = None
        self.bbox = None
        self.get_slice_and_str_fn = get_slice_and_str_fn

        self.img_to_overlay = None

        self.gt_colors = [(0, 1, 0), (0, 1, 0), (0, 1, 0)]   # all regions red
        self.gt_cm = colors.LinearSegmentedColormap.from_list('my_cmap', self.gt_colors, N=3)
        self.pred_colors = [(1, 0, 0), (1, 0, 0), (1, 0, 0)]   # all regions red
        self.pred_cm = colors.LinearSegmentedColormap.from_list('my_cmap', self.pred_colors, N=3)
        self.norm = colors.Normalize(1, 3)

        self.uncert_norm = colors.Normalize(0.0, 1.0)
        self._save_color_bar(os.path.join(out_dir, 'colorbar.png'))

    def on_new_subject(self, subject_name: str, img_data: dict):
        self.subject_dir = os.path.join(self.out_dir, subject_name)
        fh.create_dir_if_not_exists(self.subject_dir)

        self.selected_slice = self.get_slice_and_str_fn(img_data, self.task)
        slice_str = '_sl{}'.format(self.selected_slice) if not isinstance(self.selected_slice, slice) else ''

        if self.task == 'isic':
            self.bbox = self._get_bbox(img_data[self.img_key][self.selected_slice], squared='min', dims=2)
        else:
            self.bbox = self._get_bbox(img_data[self.img_key][self.selected_slice], squared='max')

        gt = self._apply_bbox(img_data['target'].astype(np.uint8)[self.selected_slice])
        ma_gt = np.ma.masked_where(gt == 0, gt)

        img_path = os.path.join(self.subject_dir, '{}{}.png'.format(self.img_key, slice_str))
        self._save_image(self._apply_bbox(img_data[self.img_key][self.selected_slice]), img_path)

        overlay_path = os.path.join(self.subject_dir, '{}_gt_overlay{}.png'.format(self.img_key, slice_str))
        self.img_to_overlay = self._apply_bbox(img_data[self.img_key][self.selected_slice])

        self._save_label_overlay(self.img_to_overlay, ma_gt, overlay_path, alpha=0.5, cm=self.gt_cm)

    def on_test_id(self, id_: str, pred_data: dict):
        prediction = self._apply_bbox(pred_data['prediction'].astype(np.uint8)[self.selected_slice])
        ma_prediction = np.ma.masked_where(prediction == 0, prediction)

        slice_str = '_sl{}'.format(self.selected_slice) if isinstance(self.selected_slice, str) else ''

        overlay_path = os.path.join(self.subject_dir, '{}_{}_pred_overlay{}.png'.format(id_, self.img_key, slice_str))
        self._save_label_overlay(self.img_to_overlay, ma_prediction, overlay_path, alpha=0.5, cm=self.pred_cm)

        uncertainty = self._apply_bbox(pred_data['uncertainty'][self.selected_slice])
        uncertainty_path = os.path.join(self.subject_dir, '{}_uncert{}.png'.format(id_, slice_str))

        self._save_image(uncertainty, uncertainty_path, cmap='inferno')

    def _save_color_bar(self, out_path: str, cmap='inferno', orientation='vertical'):
        figsize = (1, 10) if orientation == 'vertical' else (10, 1)
        fig, ax = plt.subplots(figsize=figsize)
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cb = colorbar.ColorbarBase(ax, cmap=cmap, orientation=orientation, ticks=[0, 1])
        cb.set_ticklabels(['', ''])
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    def _save_label_overlay(self, np_img: np.ndarray, np_over: np.ma.MaskedArray, out_path: str, alpha: float, cm):
        plt.imshow(np_img, 'gray', interpolation='none')
        ax_img = plt.imshow(np_over, cmap=cm, norm=self.norm, interpolation='none', alpha=alpha)
        plt.axis('off')
        ax_img.axes.get_xaxis().set_visible(False)
        ax_img.axes.get_yaxis().set_visible(False)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _save_image(self, np_img: np.ndarray, out_path: str, cmap='gray'):
        ax_img = plt.imshow(np_img, cmap, interpolation='none')
        plt.axis('off')
        ax_img.axes.get_xaxis().set_visible(False)
        ax_img.axes.get_yaxis().set_visible(False)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _get_bbox(self, np_img: np.ndarray, squared='max', dims=None):
        if dims is None:
            dims = np_img.ndim
        bbox = []

        for i, ax in enumerate(itertools.combinations(range(dims), dims - 1)):
            nonzero = np.any(np_img, axis=ax)
            min_, max_ = tuple(np.where(nonzero)[0][[0, -1]].tolist())
            if min_ - 10 >= 0:
                min_ = min_ - 10
            if max_ + 10 <= nonzero.shape[0]:
                max_ = max_ + 10
            bbox.append((min_, max_))

        bbox = bbox[::-1]

        if squared == 'max':
            max_dist = max(ma - mi for mi, ma in bbox)
            for i in range(len(bbox)):
                min_, max_ = bbox[i]
                dist_diff = max_dist - (max_ - min_)
                min_add = dist_diff//2
                max_add = dist_diff - min_add
                bbox[i] = min_ - min_add, max_ + max_add
                assert bbox[i][0] >= 0 and bbox[i][1] < np_img.shape[i]
        elif squared == 'min':
            min_dist = min(ma - mi for mi, ma in bbox)
            for i in range(len(bbox)):
                min_, max_ = bbox[i]
                dist_diff = (max_ - min_) - min_dist
                min_remove = dist_diff // 2
                max_remove = dist_diff - min_remove
                bbox[i] = min_ + min_remove, max_ - max_remove
                assert bbox[i][0] >= 0 and bbox[i][1] < np_img.shape[i]

        return bbox

    def _apply_bbox(self, np_img: np.ndarray):
        return np_img[self.bbox[0][0]:self.bbox[0][1], self.bbox[1][0]:self.bbox[1][1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, nargs='?', help='the dataset to evaluate the runs on')
    parser.add_argument('--ids', type=str, nargs='*', help='the ids of the runs to be evaluated')
    args = parser.parse_args()

    ds = args.ds
    if ds is None:
        ds = 'isic'

    plot_ids = args.ids
    if plot_ids is None:
        # no command line arguments given
        plot_ids = [
            'baseline',
            'baseline_mc',
            'center',
            'center_mc',
            'ensemble',
            'auxiliary_feat',
            'auxiliary_segm',
            'aleatoric',
        ]

    print('\n**************************************')
    print('dataset: {}'.format(ds))
    print('to_plot: {}'.format(plot_ids))
    print('**************************************\n')

    main(ds, plot_ids)
