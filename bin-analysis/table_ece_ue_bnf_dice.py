import collections
import re

import pandas as pd
import rechun.analysis.resultdata as resdata
import rechun.directories as dirs


def main():
    entries, precision = ['ece', 'error', 'benefit', 'dice'], [3, 3, 2, 3]
    brats_df = gather_information('brats')
    prep_brats_df = prepare_for_print(brats_df, entries, precision)

    isic_df = gather_information('isic')
    prep_isic_df = prepare_for_print(isic_df, entries, precision)

    combined = pd.concat([prep_brats_df, prep_isic_df], axis=1, keys=['brats', 'isic'])

    latex_str = combined.to_latex().replace('±', '$\pm$').replace('rank', 'r')
    latex_str = re.sub('([0-9]*\.?[0-9]*) \(1\)', r'\\textbf{\g<1>} (1)', latex_str)
    print(latex_str)


def gather_information(task):
    if task == 'brats':
        data, ece_dict, ids_names_dict = get_brats_data()
    else:
        data, ece_dict, ids_names_dict = get_isic_data()

    run_ids = []
    frames = []
    for id_, file_path, threshold in data:
        frame = pd.read_csv(file_path)
        frame = frame[['test_id', 'subject_name', 'corrected_dice', 'fp', 'fn', 'fnu', 'fpu', 'tnu', 'tpu', 'dice']]

        # add the dice info to the frame
        if id_ not in ece_dict:
            print('missing ece id_ {}'.format(id_))
            continue
        # add same ece to all the uncertainty thresholds
        ece_frame = pd.read_csv(ece_dict[id_])
        assert (frame['dice'] == ece_frame['dice']).all()
        frame.drop(columns='dice', inplace=True)

        frame = pd.merge(frame, ece_frame[['subject_name', 'ece', 'dice']], on=['subject_name'])

        frames.append(frame)
        run_id = '{}_th{}'.format(id_, threshold)
        run_ids.append(run_id)

    df = pd.concat(frames, keys=run_ids, names=['run_id'])

    thresholds = [float(s[-3:]) / 100 for s in list(df.index.get_level_values(0))]
    df['threshold'] = pd.Series(thresholds, index=df.index)

    df['dice_diff'] = df['corrected_dice'] - df['dice']
    df['benefit'] = df['dice_diff'] > 0

    df['error'] = (2 * (df['fnu'] + df['fpu'])) / (df['fn'] + df['fp'] + df['fnu'] + df['fpu'] + df['tnu'] + df['tpu'])

    best_benefit = get_best_thresholds(df[['test_id', 'subject_name', 'threshold', 'benefit']], 'benefit')
    best_benefit = best_benefit.rename(columns={'threshold': 'benefit_threshold'})
    best_error = get_best_thresholds(df[['test_id', 'subject_name', 'threshold', 'error']], 'error')
    best_error = best_error.rename(columns={'threshold': 'error_threshold'})

    df = df[['test_id', 'subject_name', 'ece', 'dice']]
    df = pd.merge(df, best_benefit, on=['test_id', 'subject_name'])
    df = pd.merge(df, best_error, on=['test_id', 'subject_name'])

    df = df.groupby('test_id').mean()

    df = df.reindex(list(ids_names_dict.keys())).rename(index=ids_names_dict)
    return df


def prepare_for_print(df, entries: list, precisions: list):
    # preferable to df[entries] because otherwise we get warning afterwards
    df = df.loc[:, entries]

    rel_diff = False
    if rel_diff:
        baseline = df.loc['baseline']
        for e in entries:
            df[e] = (df[e] - baseline[e]) / baseline[e]

    ece_in_percent = True
    if 'ece' in entries and ece_in_percent:
        df['ece'] = df['ece']*100

    rank_dense = True
    rank_method = 'dense' if rank_dense else 'min'
    for e, prec in zip(entries, precisions):
        df[e] = df[e].round(prec)
        ascending = True if e == 'ece' else False
        df['{}_rank'.format(e)] = df[e].rank(method=rank_method, ascending=ascending).astype(int)

    rank_sum = sum(df['{}_rank'.format(e)] for e in entries)
    df['rank'] = rank_sum.rank(method=rank_method).astype(int)

    def with_rank(y):
        if len(y) > 1:
            return '{} ({})'.format(y[0], int(y[1]))
        return int(y[0])

    def no_rank(y):
        if len(y) > 1:
            return '{}'.format(y[0])
        return int(y[0])

    show_rank = True
    rank_fn = with_rank if show_rank else no_rank
    cleanedforexport = df.groupby(df.columns.str[:3], axis=1).apply(lambda x: x.apply(rank_fn, axis=1))

    show_overall_rank = False

    rename_dict = {e[:3]: e for e in entries}
    if show_overall_rank:
        rename_dict['ran'] = 'rank'
    cleanedforexport = cleanedforexport.rename(columns=rename_dict)

    col_order = entries
    if show_overall_rank:
        col_order = col_order + ['rank']
    cleanedforexport = cleanedforexport[col_order]
    print(cleanedforexport)

    print(cleanedforexport.to_latex())

    return cleanedforexport


def get_best_thresholds(df: pd.DataFrame, entry):

    best_thresholds = []
    run_ids = []
    for name, group in df.groupby('test_id'):
        run_id_max = group.groupby('run_id').mean()[entry].idxmax()
        best_threshold = df.loc[run_id_max]
        best_thresholds.append(best_threshold)
        run_ids.append(run_id_max)

    best_stacked = pd.concat(best_thresholds, keys=run_ids, names=['run_id'])
    return best_stacked


def get_brats_data():
    data = resdata.BratsResultData()
    ids_names_dict = collections.OrderedDict(resdata.brats_selection_id_name_dict)

    files, file_ids, _, file_thresholds = data.get_files(list(ids_names_dict.keys()), [dirs.UNCERTAINTY_NAME])

    files_ece, file_ids_ece, _, _ = data.get_files(list(ids_names_dict.keys()), [dirs.ECE_FOREGROUND_NAME])
    ece_dict = {id_: f for id_, f in zip(file_ids_ece, files_ece)}

    return list(zip(file_ids, files, file_thresholds)), ece_dict, ids_names_dict


def get_isic_data():
    data = resdata.IsicResultData()
    ids_names_dict = collections.OrderedDict(resdata.isic_selection_id_name_dict)

    files, file_ids, _, file_thresholds = data.get_files(list(ids_names_dict.keys()), [dirs.UNCERTAINTY_NAME])

    files_ece, file_ids_ece, _, _ = data.get_files(list(ids_names_dict.keys()), [dirs.ECE_NAME])
    ece_dict = {id_: f for id_, f in zip(file_ids_ece, files_ece)}

    return list(zip(file_ids, files, file_thresholds)), ece_dict, ids_names_dict


if __name__ == '__main__':
    main()
