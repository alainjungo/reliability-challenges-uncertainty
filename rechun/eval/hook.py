import csv

import numpy as np
import matplotlib.backends.backend_pdf as pdf
import matplotlib.pyplot as plt

import common.trainloop.hooks as ho


class EvalHook:

    def on_run_start(self, run_id: str):
        pass

    def on_subject(self, results: dict, subject_name: str, run_id: str):
        pass

    def on_run_end(self, results_history: dict, run_id: str):
        pass


class ReducedComposeEvalHook(EvalHook):
    def __init__(self, hooks: list) -> None:
        super().__init__()
        ho.make_reduce_compose(self, EvalHook, hooks)


class WriteCsvHook(EvalHook):

    def __init__(self, file_path: str, entries=None) -> None:
        super().__init__()
        self.file_path = file_path
        self.rows = []
        self.entries = None if entries is None else list(entries)
        self.header = None

    def on_subject(self, results: dict, subject_name: str, run_id: str):
        results = self._unfold_results(results)

        if self.entries is None:
            self.entries = list(results.keys())
        results = self._select_results(results, self.entries)

        if self.header is None:
            self.header = ['test_id', 'subject_name'] + self.entries

        self.rows.append([run_id, subject_name] + [results[e] for e in self.entries])

    @staticmethod
    def _select_results(results: dict, entries: list):
        return {e: results[e] for e in results if e in entries}

    @staticmethod
    def _unfold_results(results):
        unfolded_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()  # will be handled by list handler through this cast
            if isinstance(value, (list, tuple)):
                nb_digits = len(str(len(value)))
                for i in range(len(value)):
                    unfolded_results['{}_{:0{}d}'.format(key, i, nb_digits)] = value[i]
            else:
                unfolded_results[key] = value
        return unfolded_results

    def on_run_end(self, results_history: dict, run_id: str):
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            for row in self.rows:
                writer.writerow(row)


class WriteBinsCsvHook(WriteCsvHook):

    def on_subject(self, results: dict, subject_name: str, run_id: str):
        # needed because not all bins are same lengths. bins_non_zero is only having full length
        non_zero_bins = results['bins_non_zero']

        bin_counts = np.zeros_like(non_zero_bins, dtype=results['bins_count'].dtype)
        bin_counts[non_zero_bins] = results['bins_count']
        results['bins_count'] = bin_counts

        bins_avg_confidence = np.zeros_like(non_zero_bins, dtype=results['bins_avg_confidence'].dtype)
        bins_avg_confidence[non_zero_bins] = results['bins_avg_confidence']
        results['bins_avg_confidence'] = bins_avg_confidence

        bins_positive_fraction = np.zeros_like(non_zero_bins, dtype=results['bins_positive_fraction'].dtype)
        bins_positive_fraction[non_zero_bins] = results['bins_positive_fraction']
        results['bins_positive_fraction'] = bins_positive_fraction

        super().on_subject(results, subject_name, run_id)


class WriteSummaryCsvHook(EvalHook):

    def __init__(self, file_path: str, entries=('min', 'max'), summary_fn=(np.min, np.max),
                 confidence_entry='probabilities') -> None:
        super().__init__()
        self.file_path = file_path
        if len(entries) != len(summary_fn):
            raise ValueError('entries and summary_fn must be of same length')
        self.entries = list(entries)
        self.summary_fn = list(summary_fn)
        self.confidence_entry = confidence_entry

    def on_run_end(self, results_history: dict, run_id: str):
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['confidence_entry'] + self.entries)
            summary = []
            for entry, fn in zip(self.entries, self.summary_fn):
                summary.append(fn(results_history[entry]))
            writer.writerow([self.confidence_entry] + summary)
