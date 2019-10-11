import argparse
import os
import abc
import time

import common.evalutation.eval as ev
import rechun.eval.analysis as analysis
import rechun.eval.hook as hooks
import rechun.eval.evaldata as evdata
import rechun.directories as dirs


def main(dataset, to_eval, action_names):

    if dataset not in ('brats', 'isic'):
        raise ValueError('chose "brats" or "isic" as dataset')
    prefix = dataset

    if prefix == 'brats':
        eval_data_list = evdata.get_brats_eval_data(to_eval)
        ece_details = 'foreground'
        base_dir = dirs.BRATS_EVAL_DIR
    else:
        eval_data_list = evdata.get_isic_eval_data(to_eval)
        ece_details = ''
        base_dir = dirs.ISIC_EVAL_DIR

    min_max_dir = os.path.join(base_dir, dirs.MINMAX_NAME)

    actions = get_actions(action_names, min_max_dir, base_dir, ece_details)

    for entry in eval_data_list:
        for action in actions:
            action.setup_eval(entry)

        for action in actions:
            action.start_eval()

        for i, sf in enumerate(entry.subject_files):
            print('[{}/{}] {}'.format(i + 1, len(entry.subject_files), sf.subject), end=' ', flush=True)
            loader = analysis.Loader()
            start = time.time()

            for action in actions:
                action.eval_subject(sf, loader)

            print('({}s)'.format(time.time() - start))

        for action in actions:
            action.finish_eval()


def _make_dir_if_not_exists(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


class EvalCase(abc.ABC):

    def __init__(self, metric, hook, id_='') -> None:
        super().__init__()
        self.result_history = {}
        self.metric = metric
        self.hook = hook
        self.id_ = id_

    def do_eval(self, to_eval, subject_name, id_):
        results = {}
        self.metric(to_eval, results)

        self.hook.on_subject(results, subject_name, id_)

        for k, v in results.items():
            self.result_history.setdefault(k, []).append(v)


class EvalAction(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.load_params = None
        self.prepare = None
        self.eval_cases = []
        self.id_ = ''

    @abc.abstractmethod
    def _setup_eval(self, eval_data: evdata.EvalData):
        pass

    def setup_eval(self, eval_data: evdata.EvalData):
        self._setup_eval(eval_data)

    def start_eval(self):
        sub_ids = ', '.join(s.id_ for s in self.eval_cases if s.id_ != '')
        print(self.id_ + sub_ids)
        for eval_case in self.eval_cases:
            eval_case.hook.on_run_start(self.id_)

    def eval_subject(self, sf, loader):
        to_eval = loader.get_data(sf, self.load_params)
        if self.prepare:
            to_eval = self.prepare(to_eval)

        for eval_case in self.eval_cases:
            eval_case.do_eval(to_eval, sf.subject, self.id_)

    def finish_eval(self):
        for eval_case in self.eval_cases:
            eval_case.hook.on_run_end(eval_case.result_history, self.id_)


class EceCalibrationAction(EvalAction):

    def __init__(self, base_dir: str, details: str = '', rescale_confidence='subject', rescale_sigma='subject',
                 min_max_dir: str = None) -> None:
        super().__init__()
        self.need_mask = details == 'foreground'
        self.rescale_confidence = rescale_confidence
        self.rescale_sigma = rescale_sigma
        self.min_max_dir = min_max_dir
        self.out_dir = os.path.join(base_dir, dirs.CALIB_NAME)
        _make_dir_if_not_exists(self.out_dir)

    def _setup_eval(self, eval_data: evdata.EvalData):
        self.prepare, self.id_ = analysis.get_probability_preparation(eval_data,
                                                                      rescale_confidence=self.rescale_confidence,
                                                                      rescale_sigma=self.rescale_sigma,
                                                                      min_max_dir=self.min_max_dir)
        self.load_params = analysis.Loader.Params(eval_data.confidence_entry, need_t2_mask=self.need_mask)

        metric = ev.ComposeEvaluation([
            ev.EceBinaryNumpy(threshold_range=None, return_bins=True, with_mask=self.need_mask),
            ev.DiceNumpy()
        ])
        hook = hooks.ReducedComposeEvalHook([
            hooks.WriteBinsCsvHook(os.path.join(self.out_dir, dirs.CALIBRATION_PLACEHOLDER.format(self.id_)))
        ])
        self.eval_cases = [EvalCase(metric, hook)]


class EceAction(EvalAction):

    def __init__(self, base_dir: str, details: str, rescale_confidence='subject', rescale_sigma='subject',
                 min_max_dir: str = None) -> None:
        super().__init__()
        self.rescale_confidence = rescale_confidence
        self.rescale_sigma = rescale_sigma
        self.min_max_dir = min_max_dir
        self.need_t2_mask = details == 'foreground'

        if details == 'foreground':
            self._m = [ev.EceBinaryNumpy(threshold_range=None, with_mask=True)]
            self.out_dir = os.path.join(base_dir, dirs.ECE_FOREGROUND_NAME)
        else:
            self._m = [ev.EceBinaryNumpy(threshold_range=None)]
            self.out_dir = os.path.join(base_dir, dirs.ECE_NAME)

        self.ece_entries = ['ece']
        _make_dir_if_not_exists(self.out_dir)

    def _setup_eval(self, eval_data: evdata.EvalData):
        self.prepare, self.id_ = analysis.get_probability_preparation(eval_data,
                                                                      rescale_confidence=self.rescale_confidence,
                                                                      rescale_sigma=self.rescale_sigma,
                                                                      min_max_dir=self.min_max_dir)
        self.load_params = analysis.Loader.Params(eval_data.confidence_entry, need_t2_mask=self.need_t2_mask)

        metric = ev.ComposeEvaluation([*self._m, ev.DiceNumpy(), ev.ConfusionMatrix()])
        hook = hooks.ReducedComposeEvalHook([
            hooks.WriteCsvHook(os.path.join(self.out_dir, dirs.ECE_PLACEHOLDER.format(self.id_)),
                               entries=(*self.ece_entries, 'dice', 'tp', 'tn', 'fp', 'fn', 'n'))
        ])
        self.eval_cases = [EvalCase(metric, hook)]


class CorrectionAction(EvalAction):

    def __init__(self, thresholds: list, base_dir: str, rescale_confidence='', rescale_sigma='global',
                 min_max_dir: str = None) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.rescale_confidence = rescale_confidence
        self.rescale_sigma = rescale_sigma
        self.min_max_dir = min_max_dir
        self.out_dir = os.path.join(base_dir, dirs.UNCERTAINTY_NAME)
        _make_dir_if_not_exists(self.out_dir)

    def _setup_eval(self, eval_data: evdata.EvalData):
        self.prepare, self.id_ = analysis.get_uncertainty_preparation(eval_data,
                                                                      rescale_confidence=self.rescale_confidence,
                                                                      rescale_sigma=self.rescale_sigma,
                                                                      min_max_dir=self.min_max_dir)
        self.load_params = analysis.Loader.Params(eval_data.confidence_entry)

        self.eval_cases = []
        for threshold in self.thresholds:
            metric = ev.UncertaintyAndCorrectionEvalNumpy(threshold)
            threshold_str = '{:.2f}'.format(threshold).replace('.', '')

            out_csv = os.path.join(self.out_dir, dirs.UNCERTAINTY_PLACEHOLDER.format(self.id_, threshold_str))
            hook = hooks.WriteCsvHook(out_csv, None)
            self.eval_cases.append(EvalCase(metric, hook))


class SaveMinMaxAction(EvalAction):

    def __init__(self, min_max_dir: str) -> None:
        super().__init__()
        self.min_max_dir = min_max_dir
        _make_dir_if_not_exists(min_max_dir)

    def _setup_eval(self, eval_data: evdata.EvalData):

        self.prepare, self.id_ = analysis.get_confidence_entry_preparation(eval_data, 'probabilities')
        self.load_params = analysis.Loader.Params(eval_data.confidence_entry)

        metric = ev.ComposeEvaluation([
            ev.LambdaEvaluation(lambda x: x.min(), ('probabilities',), 'min'),
            ev.LambdaEvaluation(lambda x: x.max(), ('probabilities',), 'max')
        ])
        hook = hooks.WriteSummaryCsvHook(os.path.join(self.min_max_dir, dirs.MINMAX_PLACEHOLDER.format(self.id_)),
                                         confidence_entry=eval_data.confidence_entry)
        self.eval_cases = [EvalCase(metric, hook)]


def get_actions(action_names, min_max_dir, base_dir, ece_details):
    actions = []
    for action_name in action_names:
        action = None
        if action_name == 'minmax':
            action = SaveMinMaxAction(min_max_dir)
        elif action_name == 'ece_dice':
            action = EceAction(base_dir, ece_details, rescale_confidence='subject', rescale_sigma='global',
                               min_max_dir=min_max_dir)
        elif action_name == 'calib':
            action = EceCalibrationAction(base_dir, ece_details, rescale_confidence='subject', rescale_sigma='global',
                                          min_max_dir=min_max_dir)
        elif action_name == 'bnf_ue':
            action = CorrectionAction([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], base_dir,
                                      rescale_confidence='subject', rescale_sigma='global', min_max_dir=min_max_dir)
        if action is not None:
            actions.append(action)

    return actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, nargs='?', help='the dataset to evaluate the runs on')
    parser.add_argument('--ids', type=str, nargs='*', help='the ids of the runs to be evaluated')
    parser.add_argument('--act', type=str, nargs='*', help='the names of the evaluation configuration')
    args = parser.parse_args()

    ds = args.ds
    if ds is None:
        ds = 'brats'

    to_evaluate = args.ids
    if to_evaluate is None:
        # no command line arguments given
        to_evaluate = [
            'baseline',
            'baseline_mc',
            'center',
            'center_mc',
            'ensemble',
            'auxiliary_feat',
            'auxiliary_segm',
            'aleatoric',
        ]

    action_ids = args.act
    if action_ids is None:
        # no command line arguments given
        action_ids = [
            'minmax',
            'ece_dice',
            'calib',
            'bnf_ue'
        ]

    print('\n**************************************')
    print('dataset: {}'.format(ds))
    print('to_evaluate: {}'.format(to_evaluate))
    print('eval_actions: {}'.format(action_ids))
    print('**************************************\n')

    main(ds, to_evaluate, action_ids)
