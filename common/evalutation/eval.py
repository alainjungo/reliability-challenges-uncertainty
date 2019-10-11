import abc

import numpy as np

import common.evalutation.torchfunctions as t_fn
import common.evalutation.numpyfunctions as np_fn


class EvaluationStrategy(metaclass=abc.ABCMeta):

    def __init__(self, result_entry=None) -> None:
        self.result_entry = result_entry

    @abc.abstractmethod
    def __call__(self, to_evaluate: dict, results: dict) -> None:
        pass


class EmptyEvaluation(EvaluationStrategy):
    def __call__(self, to_evaluate: dict, results: dict) -> None:
        pass


class ComposeEvaluation(EvaluationStrategy):

    def __init__(self, eval_strategies) -> None:
        super().__init__()
        self.eval_strategies = eval_strategies

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        for eval_ in self.eval_strategies:
            eval_(to_evaluate, results)


class LambdaEvaluation(EvaluationStrategy):

    def __init__(self, lambda_fn, entry_keys:tuple, result_entry) -> None:
        super().__init__(result_entry)
        self.lamda_fn = lambda_fn
        self.entry_keys = entry_keys

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        entries = []
        for entry_key in self.entry_keys:
            entries.append(to_evaluate[entry_key])
        results[self.result_entry] = self.lamda_fn(*entries)


class TorchEvaluationStrategy(EvaluationStrategy, metaclass=abc.ABCMeta):
    pass


class SmoothDice(TorchEvaluationStrategy):

    def __init__(self, result_entry='smooth_dice') -> None:
        super().__init__(result_entry)

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        prediction = to_evaluate['prediction'].float()
        target = to_evaluate['target'].float()
        dice = t_fn.smooth_dice(prediction, target).item()
        results[self.result_entry] = dice


class Nll(TorchEvaluationStrategy):

    def __init__(self, do_log=True, result_entry='nll') -> None:
        super().__init__(result_entry)
        # log since results are probabilities and nll needs log_softmax inputs
        self.do_log = do_log

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        probs = to_evaluate['probabilities'].float()
        target = to_evaluate['target'].long()

        ce = t_fn.nll(probs, target, self.do_log).item()
        results[self.result_entry] = ce


class NumpyEvaluationStrategy(EvaluationStrategy, metaclass=abc.ABCMeta):
    pass


class LogLossSklearn(NumpyEvaluationStrategy):

    def __init__(self, result_entry='ce', labels=None) -> None:
        super().__init__(result_entry)
        self.labels = labels

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        results[self.result_entry] = np_fn.log_loss_sklearn(to_evaluate['probabilities'], to_evaluate['target'],
                                                            self.labels)


class DiceNumpy(NumpyEvaluationStrategy):

    def __init__(self, result_entry='dice') -> None:
        super().__init__(result_entry)

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        results[self.result_entry] = np_fn.dice(to_evaluate['prediction'], to_evaluate['target'])


class ConfusionMatrix(NumpyEvaluationStrategy):

    def __init__(self, result_entries=('tp', 'tn', 'fp', 'fn', 'n')) -> None:
        super().__init__(result_entries)

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        tp, tn, fp, fn, n = np_fn.confusion_matrx(to_evaluate['prediction'], to_evaluate['target'])
        results[self.result_entry[0]] = tp
        results[self.result_entry[1]] = tn
        results[self.result_entry[2]] = fp
        results[self.result_entry[3]] = fn
        results[self.result_entry[4]] = n


class EceBinaryNumpy(NumpyEvaluationStrategy):

    def __init__(self, n_bins=10, result_entry='ece', threshold_range:tuple=None, with_mask=False,
                 return_bins=False, bin_weighting='proportion') -> None:
        super().__init__(result_entry)
        self.n_bins = n_bins
        self.threshold_range = threshold_range
        self.with_mask = with_mask
        self.return_bins = return_bins
        self.bin_weighting = bin_weighting

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        target = to_evaluate['target']
        probabilities = to_evaluate['probabilities']
        mask = None
        if self.with_mask:
            mask = to_evaluate['mask']

        out_bins = None
        if self.return_bins:
            out_bins = results
        ece = np_fn.ece_binary(probabilities, target, self.n_bins, self.threshold_range, mask, out_bins,
                               self.bin_weighting)

        results[self.result_entry] = ece


class UncertaintyErrorDiceNumpy(NumpyEvaluationStrategy):

    def __init__(self, uncertainty_threshold, result_prefix: str=None, with_mask=False) -> None:
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        if result_prefix is None:
            result_prefix = ''
        else:
            result_prefix += '_'
        self.prefix = result_prefix
        self.with_mask = with_mask

    def __call__(self, to_evaluate: dict, results: dict):
        # entropy should be normalized [0,1] before calling this evaluation
        target = to_evaluate['target'].astype(np.bool)
        prediction = to_evaluate['prediction'].astype(np.bool)
        uncertainty = to_evaluate['uncertainty']

        mask = None
        if self.with_mask:
            mask = ~to_evaluate['target_boarder']

        thresholded_uncertainty = uncertainty > self.uncertainty_threshold
        tp, tn, fp, fn, tpu, tnu, fpu, fnu = \
            np_fn.uncertainty(prediction, target, thresholded_uncertainty, mask=mask)

        results['{}precision'.format(self.prefix)] = np_fn.error_precision(tpu, tnu, fpu, fnu)
        results['{}recall'.format(self.prefix)] = np_fn.error_recall(fp, fn, fpu, fnu)
        results['{}dice'.format(self.prefix)] = np_fn.error_dice(fp, fn, tpu, tnu, fpu, fnu)


class UncertaintyAndCorrectionEvalNumpy(NumpyEvaluationStrategy):

    def __init__(self, uncertainty_threshold) -> None:
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold

    def __call__(self, to_evaluate: dict, results: dict) -> None:
        # entropy should be normalized [0,1] before calling this evaluation
        target = to_evaluate['target'].astype(np.bool)
        prediction = to_evaluate['prediction'].astype(np.bool)
        uncertainty = to_evaluate['uncertainty']

        thresholded_uncertainty = uncertainty > self.uncertainty_threshold
        tp, tn, fp, fn, tpu, tnu, fpu, fnu = \
            np_fn.uncertainty(prediction, target, thresholded_uncertainty)

        results['tpu'] = tpu
        results['tnu'] = tnu
        results['fpu'] = fpu
        results['fnu'] = fnu

        results['tp'] = tp
        results['tn'] = tn
        results['fp'] = fp
        results['fn'] = fn

        tpu_fpu_ratio = results['tpu'] / results['fpu']
        jaccard_index = results['tp'] / (results['tp'] + results['fp'] + results['fn'])
        results['dice_benefit'] = tpu_fpu_ratio < jaccard_index
        results['accuracy_benefit'] = tpu_fpu_ratio < 1

        results['dice'] = np_fn.dice(prediction, target)
        results['accuracy'] = np_fn.accuracy(prediction, target)

        corrected_prediction = prediction.copy()
        # correct to background
        corrected_prediction[thresholded_uncertainty] = 0

        results['corrected_dice'] = np_fn.dice(corrected_prediction, target)
        results['corrected_accuracy'] = np_fn.accuracy(corrected_prediction, target)

        results['dice_benefit_correct'] = (results['corrected_dice'] > results['dice']) == results['dice_benefit']
        results['accuracy_benefit_correct'] = (results['corrected_accuracy'] > results['accuracy']) == results[
            'accuracy_benefit']

        corrected_prediction = prediction.copy()
        # correct to foreground
        corrected_prediction[thresholded_uncertainty] = 1

        results['corrected_add_dice'] = np_fn.dice(corrected_prediction, target)
        results['corrected_add_accuracy'] = np_fn.accuracy(corrected_prediction, target)


def _check_and_return(obj, type_):
    if not isinstance(obj, type_):
        raise ValueError("entry must be '{}'".format(type_.__name__))
    return obj
