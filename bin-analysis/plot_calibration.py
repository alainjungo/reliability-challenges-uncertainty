import argparse
import os
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rechun.analysis.resultdata as resdata
import rechun.directories as dirs
import common.utils.filehelper as fh


def main(dataset):

    if dataset not in ('brats', 'isic'):
        raise ValueError('dataset must be "isic" or "brats"')

    if dataset == 'brats':
        data, ids_names_dict, task = get_brats_data()
        subjects = ['Brats18_TCIA01_390_1', 'Brats18_CBICA_AUN_1', 'Brats18_CBICA_ASY_1']
        out_dir = os.path.join(dirs.BRATS_PLOT_DIR, dirs.CALIB_NAME)
    else:
        data, ids_names_dict, task = get_isic_data()
        subjects = ['ISIC_0012388', 'ISIC_0012654', 'ISIC_0012447']
        out_dir = os.path.join(dirs.ISIC_PLOT_DIR, dirs.CALIB_NAME)

    fh.create_dir_if_not_exists(out_dir)

    run_ids = []
    frames = []
    for run_id, file_path in data:
        frames.append(pd.read_csv(file_path))
        run_ids.append(run_id)

    df = pd.concat(frames, keys=run_ids, names=['run_id'])

    # create a calibration plot with all run_ids
    create_pdf_all_run_id(df, out_dir, ids_names_dict, task, legend=False)
    # create a calibration pdf for each subject
    create_subject_pdfs(df, out_dir, ids_names_dict, subjects, legend=False)
    create_legend_only(ids_names_dict, out_dir)

    miscalibration_percentage(df, ids_names_dict, task)


def miscalibration_percentage(df: pd.DataFrame, id_names_dict, task):

    voxelwise_calib_errors = []
    for run_id, name in id_names_dict.items():
        run_df = df.loc[run_id]
        bins_all_avg_confidence, bins_all_positive_fraction, bins_all_counts = get_bins(run_df)
        corr_all_avg_confidence = bins_all_avg_confidence * bins_all_counts
        corr_all_positive_fraction = bins_all_positive_fraction * bins_all_counts

        bin_sum = bins_all_counts.sum(axis=0)
        avg_conf = corr_all_avg_confidence.sum(axis=0).compressed() / bin_sum.compressed()
        pos_frac = corr_all_positive_fraction.sum(axis=0).compressed() / bin_sum.compressed()

        calibration_error = (pos_frac - avg_conf).mean()  # not ece: tells whether over or undercalibrated
        voxelwise_calib_errors.append(calibration_error)

    miscalibration_threshold = np.percentile(np.asarray(voxelwise_calib_errors), 90)

    subjectwise_calib_error = []
    for subject_name, group in df.groupby('subject_name'):
        run_calib_errors = []
        for run_id, name in id_names_dict.items():
            run_df = group.loc[run_id]
            bins_avg_confidence, bins_positive_fraction, bins_counts = get_bins(run_df)

            calibration_error = (bins_positive_fraction - bins_avg_confidence).mean()  # not ece: tells whether over or undercalibrated
            run_calib_errors.append(calibration_error)
        subjectwise_calib_error.append(run_calib_errors)

    calib_arr = np.asarray(subjectwise_calib_error)
    calib_mean = calib_arr.mean(axis=1)

    ratio_underconfident = (calib_mean > miscalibration_threshold).sum() / calib_mean.size
    ratio_overconfident = (calib_mean < -miscalibration_threshold).sum() / calib_mean.size
    ratio_calibrated = np.logical_and(calib_mean <= miscalibration_threshold, calib_mean >= -miscalibration_threshold).sum() / calib_mean.size

    print('[{}] overall_err:{:.3f} \t overconfident: {:.2f} \t underconfident: {:.2f}  \t well-calibrated: {:.2f}'.
          format(task, miscalibration_threshold, ratio_overconfident, ratio_underconfident, ratio_calibrated))


def create_legend_only(id_names_dict, base_dir):
    fig_legend = plt.figure(figsize=(11.5, 0.5))
    fig, ax = plt.subplots()
    line_ranges = (2*len(id_names_dict))*[range(2)]
    bars = ax.plot(*line_ranges, label=list(id_names_dict.keys()))
    fig_legend.legend(bars, list(id_names_dict.values()), loc='center', ncol=len(id_names_dict), frameon=False)
    # fig_legend.show()
    fig_legend.savefig(os.path.join(base_dir, 'legend.svg'), bbox_inches='tight')


def create_subject_pdfs(df: pd.DataFrame, base_dir, id_names_dict, subjects, legend=False):
    for subject_name in subjects:
        group = df.loc[df['subject_name'] == subject_name]

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xlabel('confidence', fontsize=18)
        plt.ylabel('accuracy', fontsize=18)
        ax.plot([0, 1], [0, 1], '--', label=None, color='Black')
        for run_id, name in id_names_dict.items():
            run_group = group.loc[run_id]
            bins_avg_confidence, bins_positive_fraction, bins_counts = get_bins(run_group)
            ax.plot(bins_avg_confidence.compressed(), bins_positive_fraction.compressed(), "-", label=name)
        if legend:
            plt.legend()
        # plt.title(subject_name)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(os.path.join(base_dir, 'subject_{}.svg'.format(subject_name)), bbox_inches='tight')
        plt.close()


def create_pdf_all_run_id(df: pd.DataFrame, base_dir, ids_names_dict, task, legend=False):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.xlabel('confidence', fontsize=18)
    plt.ylabel('accuracy', fontsize=18)

    ax.plot([0, 1], [0, 1], '--', label=None, color='Black')
    # names.append('ideal')

    for run_id, name in ids_names_dict.items():
        group = df.loc[run_id]
        bins_all_avg_confidence, bins_all_positive_fraction, bins_all_counts = get_bins(group)

        # voxel-wise (corrected with count)
        corr_all_avg_confidence = bins_all_avg_confidence * bins_all_counts
        corr_all_positive_fraction = bins_all_positive_fraction * bins_all_counts

        bin_sum = bins_all_counts.sum(axis=0)
        avg_conf = corr_all_avg_confidence.sum(axis=0).compressed() / bin_sum.compressed()
        pos_frac = corr_all_positive_fraction.sum(axis=0).compressed() / bin_sum.compressed()

        bin_proportions = bins_all_counts / bins_all_counts.sum(axis=1, keepdims=True)
        ece = (np.abs(bins_all_avg_confidence - bins_all_positive_fraction) * bin_proportions).sum(axis=1)

        # verify if ece is the same
        allclose = np.allclose(ece.data, group['ece'].values)
        assert allclose

        ax.plot(avg_conf, pos_frac, "-", label=name)

    ax.tick_params(axis='both', which='major', labelsize=14)
    if legend:
        plt.legend()
    plt.savefig(os.path.join(base_dir, 'summary_all_{}.svg'.format(task)), bbox_inches='tight')
    plt.close(fig)


def get_bins(df: pd.DataFrame):
    bins_avg_confidence = df.loc[:, 'bins_avg_confidence_00': 'bins_avg_confidence_09']
    bins_positive_fraction = df.loc[:, 'bins_positive_fraction_00': 'bins_positive_fraction_09']
    bins_nonzero = df.loc[:, 'bins_non_zero_00': 'bins_non_zero_09']
    bins_counts = df.loc[:, 'bins_count_00': 'bins_count_09']

    bins_avg_confidences = bins_avg_confidence.values
    bins_positive_fractions = bins_positive_fraction.values
    bins_nonzeros = bins_nonzero.values
    bins_counts = bins_counts.values

    bins_avg_confidences = np.ma.array(bins_avg_confidences, mask=~bins_nonzeros)
    bins_positive_fractions = np.ma.array(bins_positive_fractions, mask=~bins_nonzeros)
    bins_counts = np.ma.array(bins_counts, mask=~bins_nonzeros)

    return bins_avg_confidences, bins_positive_fractions, bins_counts


def get_brats_data():
    data = resdata.BratsResultData()
    ids_names_dict = collections.OrderedDict(resdata.brats_selection_id_name_dict)

    files, file_ids, _, _ = data.get_files(list(ids_names_dict.keys()), [dirs.CALIB_NAME])
    return list(zip(file_ids, files)), ids_names_dict, 'brats'


def get_isic_data():
    data = resdata.IsicResultData()
    ids_names_dict = collections.OrderedDict(resdata.isic_selection_id_name_dict)

    files, file_ids, _, _ = data.get_files(list(ids_names_dict.keys()), [dirs.CALIB_NAME])
    return list(zip(file_ids, files)), ids_names_dict, 'isic'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, nargs='?', help='the dataset used to for analysis')
    args = parser.parse_args()

    ds = args.ds
    if ds is None:
        ds = 'brats'

    print('\n**************************************')
    print('dataset: {}'.format(ds))
    print('**************************************\n')

    main(ds)
