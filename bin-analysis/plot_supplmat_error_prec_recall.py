import argparse
import os
import collections

import pandas as pd
import matplotlib.pyplot as plt

import rechun.directories as dirs
import rechun.analysis.resultdata as resdata
import rechun.eval.helper as anl_help
import common.utils.filehelper as fh


def main(ds):
    if ds not in ('brats', 'isic'):
        raise ValueError('dataset must be "isic" or "brats"')

    task = ds

    if task == 'brats':
        out_dir = os.path.join(dirs.BRATS_PLOT_DIR, 'suppl_mat')
        data, ids_names_dict = get_brats_data()
    else:
        out_dir = os.path.join(dirs.ISIC_PLOT_DIR, 'suppl_mat')
        data, ids_names_dict = get_isic_data()

    fh.create_dir_if_not_exists(out_dir)

    df = gather_base(data)
    out_file = os.path.join(out_dir, 'error_prec_recall_{}.svg'.format(task))
    plot_precision_recall(df, ids_names_dict, out_file)

    create_legend_only(ids_names_dict, out_dir)


def gather_base(data):

    run_ids = []
    frames = []
    for id_, file_path, threshold in data:
        frame = pd.read_csv(file_path)
        frames.append(frame)
        run_id = '{}_th{}'.format(id_, threshold)
        run_ids.append(run_id)

    df = pd.concat(frames, keys=run_ids, names=['run_id'])

    thresholds = [float(s[-3:]) / 100 for s in list(df.index.get_level_values(0))]
    df['threshold'] = pd.Series(thresholds, index=df.index)

    return df


def create_legend_only(id_names_dict, base_dir):
    fig_legend = plt.figure(figsize=(11.5, 0.5))
    fig, ax = plt.subplots()
    line_ranges = (2*len(id_names_dict))*[range(2)]
    bars = ax.plot(*line_ranges, label=list(id_names_dict.keys()))
    fig_legend.legend(bars, list(id_names_dict.values()), loc='center', ncol=len(id_names_dict), frameon=False)
    fig_legend.savefig(os.path.join(base_dir, 'legend.svg'), bbox_inches='tight')


def plot_precision_recall(df, ids_names_dict, outfile, with_legend=False):
    df['ue_sens'] = anl_help.pandas_error_recall(df['fp'], df['fn'], df['fpu'], df['fnu'])
    df['ue_prec'] = anl_help.pandas_error_precision(df['tpu'], df['tnu'], df['fpu'], df['fnu'])

    fig, ax = plt.subplots()
    for run_id, name in ids_names_dict.items():
        group = df[df['test_id'] == run_id].groupby('threshold')[['ue_prec', 'ue_sens']].mean()
        group = group.sort_values('ue_prec')
        group.reset_index(inplace=True)
        group.plot('ue_prec', 'ue_sens', kind='line', ax=ax, label=name, marker='.', markersize='6', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if not with_legend:
        ax.get_legend().remove()

    ax.set_xlim(0.0, 0.6)

    plt.xlabel('precision', fontsize=14)
    plt.ylabel('recall', fontsize=14)
    plt.savefig(outfile)


def get_brats_data():
    data = resdata.BratsResultData()
    ids_names_dict = collections.OrderedDict(resdata.brats_selection_id_name_dict)

    files, file_ids, _, file_thresholds = data.get_files(list(ids_names_dict.keys()), [dirs.UNCERTAINTY_NAME])
    return list(zip(file_ids, files, file_thresholds)), ids_names_dict


def get_isic_data():
    data = resdata.IsicResultData()
    ids_names_dict = collections.OrderedDict(resdata.isic_selection_id_name_dict)

    files, file_ids, _, file_thresholds = data.get_files(list(ids_names_dict.keys()), [dirs.UNCERTAINTY_NAME])
    return list(zip(file_ids, files, file_thresholds)), ids_names_dict


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
