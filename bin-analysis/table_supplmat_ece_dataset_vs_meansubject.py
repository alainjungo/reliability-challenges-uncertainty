import collections

import pandas as pd
import numpy as np

import rechun.analysis.resultdata as resdata
import rechun.directories as dirs


def main():

    precisions = {'ece': 3, 'ds_ece': 3}

    brats_df = gather_information('brats')
    prep_brats_df = prepare_for_print(brats_df, precisions)

    isic_df = gather_information('isic')
    prep_isic_df = prepare_for_print(isic_df, precisions)

    combined = pd.concat([prep_brats_df, prep_isic_df], axis=1, keys=['brats', 'isic'])

    latex_str = combined.to_latex()
    print(latex_str)


def gather_information(task: str):
    if task == 'brats':
        data, ids_names_dict = get_brats_data()
    else:
        data, ids_names_dict = get_isic_data()

    run_ids = []
    frames = []
    for run_id, file_path in data:
        frames.append(pd.read_csv(file_path))
        run_ids.append(run_id)

    df = pd.concat(frames, keys=run_ids, names=['run_id'])
    df.reset_index(level=1, drop=True, inplace=True)

    mean_df = data_set_vs_mean_subject_ece(df, ids_names_dict)
    return mean_df


def prepare_for_print(df: pd.DataFrame, precisions: dict):
    do_percent = True

    if do_percent:
        df['ece'] *= 100
        df['ds_ece'] *= 100

    for entry, prec in precisions.items():
        df[entry] = df[entry].round(prec)

    print(df.to_latex())
    return df


def data_set_vs_mean_subject_ece(df: pd.DataFrame, ids_names_dict):

    def mean_and_ds_mean_ece(frame):
        bins_all_avg_confidence, bins_all_positive_fraction, bins_all_counts = get_bins(frame)

        # voxel-wise (corrected with count)
        corr_all_avg_confidence = bins_all_avg_confidence * bins_all_counts
        corr_all_positive_fraction = bins_all_positive_fraction * bins_all_counts

        bin_sum = bins_all_counts.sum(axis=0)
        avg_conf = corr_all_avg_confidence.sum(axis=0) / bin_sum
        pos_frac = corr_all_positive_fraction.sum(axis=0) / bin_sum

        bin_proportions = bins_all_counts / bins_all_counts.sum(axis=1, keepdims=True)
        ece = (np.abs(bins_all_avg_confidence - bins_all_positive_fraction) * bin_proportions).sum(axis=1)

        # verify if ece is the same
        allclose = np.allclose(ece.data, frame['ece'].values)
        assert allclose

        mean_ece = ece.mean()
        ds_ece = (np.abs(avg_conf - pos_frac) * bin_sum / bin_sum.sum()).sum()

        return pd.Series({'ece': mean_ece, 'ds_ece': ds_ece})

    mean_ece_df = df.groupby('test_id').apply(mean_and_ds_mean_ece)
    mean_ece_df = mean_ece_df.reindex(list(ids_names_dict.keys())).rename(index=ids_names_dict)
    return mean_ece_df


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
    return list(zip(file_ids, files)), ids_names_dict


def get_isic_data():
    data = resdata.IsicResultData()
    ids_names_dict = collections.OrderedDict(resdata.isic_selection_id_name_dict)

    files, file_ids, _, _ = data.get_files(list(ids_names_dict.keys()), [dirs.CALIB_NAME])
    return list(zip(file_ids, files)), ids_names_dict


if __name__ == '__main__':
    main()
