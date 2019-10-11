import time
import logging
import os
import csv

import numpy as np

import common.trainloop.context as ctx
import common.model.management as model_mgt
import common.utils.messages as msg


class TrainLoopHook:

    def on_startup(self):
        pass

    def end_startup(self, context: ctx.TrainContext):
        pass

    def on_termination(self, context: ctx.TrainContext):
        pass

    def on_epoch_start(self, context: ctx.TrainContext, epoch: int):
        pass

    def on_epoch_end(self, context: ctx.TrainContext, epoch: int):
        pass

    def on_training_start(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        pass

    def on_training_end(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        pass

    def on_training_batch_start(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                                context: ctx.TrainContext):
        pass

    def on_training_batch_end(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                              context: ctx.TrainContext):
        pass

    def on_validation_start(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        pass

    def on_validation_end(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        pass

    def on_validation_batch_start(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                                  context: ctx.TrainContext):
        pass

    def on_validation_batch_end(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                                context: ctx.TrainContext):
        pass

    def on_validation_subject_start(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                                    context: ctx.TrainContext):
        pass

    def on_validation_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                                  context: ctx.TrainContext):
        pass


class TestLoopHook:

    def on_startup(self):
        pass

    def end_startup(self, context: ctx.TestContext):
        pass

    def on_termination(self, context: ctx.TestContext):
        pass

    def on_test_start(self, task_context: ctx.TaskContext, context: ctx.TestContext):
        pass

    def on_test_end(self, task_context: ctx.TaskContext, context: ctx.TestContext):
        pass

    def on_test_batch_start(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        pass

    def on_test_batch_end(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                          context: ctx.TestContext):
        pass

    def on_test_subject_start(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                              context: ctx.TestContext):
        pass

    def on_test_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        pass


def make_compose(obj, hook_cls, hooks: list):
    """method that produces the compose of the hook_cls for all hooks in the list"""
    def _get_loop_fn(fn_name):
        def loop(*args, **kwargs):
            for hook in hooks:
                fn = getattr(hook, fn_name)
                fn(*args, **kwargs)
        return loop

    method_list = [func for func in dir(hook_cls)
                   if callable(getattr(hook_cls, func)) and not func.startswith("__")]
    for method in method_list:
        setattr(obj, method, _get_loop_fn(method))


def make_reduce_compose(obj, hook_cls, hooks: list):
    """only keeps the overridden methods not the empty ones"""
    def _get_loop_fn(fns):
        def loop(*args, **kwargs):
            for fn in fns:
                fn(*args, **kwargs)
        return loop

    method_list = [func for func in dir(hook_cls)
                   if callable(getattr(hook_cls, func)) and not func.startswith("__")]
    for method in method_list:
        hook_fns = []
        for hook in hooks:
            base_fn = getattr(hook_cls, method)
            hook_fn = getattr(hook, method)
            if hook_fn.__func__ != base_fn:
                hook_fns.append(hook_fn)
        setattr(obj, method, _get_loop_fn(hook_fns))


class ComposeTrainLoopHook(TrainLoopHook):
    def __init__(self, hooks: list) -> None:
        super().__init__()
        make_compose(self, TrainLoopHook, hooks)


class ReducedComposeTrainLoopHook(TrainLoopHook):
    def __init__(self, hooks: list) -> None:
        super().__init__()
        make_reduce_compose(self, TrainLoopHook, hooks)


class ReducedComposeTestLoopHook(TestLoopHook):
    def __init__(self, hooks: list) -> None:
        super().__init__()
        make_reduce_compose(self, TestLoopHook, hooks)


class TensorboardXHook(TrainLoopHook):

    def on_training_batch_end(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                              context: ctx.TrainContext):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        step = task_context.epoch * task_context.data.nb_batches + batch_context.batch_index
        for key, result in batch_context.metrics.items():
            context.tb.add_scalar('train/{}'.format(key), result, step)

    def on_validation_end(self, task_context: ctx.TaskContext, context: ctx.Context):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        for task in task_context.history.get_tasks():
            for metric in task_context.history.get_entries_keys(task):
                entries = task_context.history.get_entries(metric, task)
                mean = np.mean(entries)
                context.tb.add_scalar('valid/{}'.format(metric), mean, task_context.epoch)

    def on_termination(self, context: ctx.Context):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        context.tb.close()


class ConsoleLogHook(TrainLoopHook):

    def __init__(self, print_subject_results=True) -> None:
        super().__init__()
        self.train_batch_start_time = None
        self.valid_subject_start_time = None
        self.valid_start_time = None
        self.print_subject_results = print_subject_results

    def on_startup(self):
        logging.info('startup')

    def end_startup(self, context: ctx.TrainContext):
        logging.info('model: \n{}'.format(str(context.model)))
        params = sum(p.numel() for p in context.model.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(params))
        logging.info('startup finished')

    def on_termination(self, context: ctx.TrainContext):
        logging.info('training completed')

    def on_training_start(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        logging.info('training')
        self.train_batch_start_time = time.time()

    def on_training_batch_end(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                              context: ctx.TrainContext):
        if (batch_context.batch_index + 1) % context.config.log_every_nth == 0 or \
                batch_context.batch_index == task_context.data.nb_batches - 1:
            duration = time.time() - self.train_batch_start_time
            result_string = ' | '.join('{}: {:.5f}'.format(v, k) for v, k in batch_context.metrics.items())
            logging.info('[{}/{}, {}/{}, {:.3}s] {}'.format(task_context.epoch + 1, context.config.epochs,
                                                            batch_context.batch_index + 1,
                                                            task_context.data.nb_batches, duration, result_string))
            # start timing here in order to take into account the data loading
            self.train_batch_start_time = time.time()

    def on_validation_start(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        logging.info('validating')
        now = time.time()
        self.valid_subject_start_time = now
        self.valid_start_time = now

    def on_validation_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                                  context: ctx.TrainContext):
        duration = time.time() - self.valid_subject_start_time

        if self.print_subject_results:
            result_string = ' | '.join('{}: {:.5f}'.format(v, k) for v, k in subject_context.metrics.items())
            subject_name = subject_context.subject_index
            if 'subject' in subject_context.subject_data:  # in case the subject index is not the name
                subject_name = subject_context.subject_data['subject']
            logging.info('[{} {:.3}s] {}'.format(subject_name, duration, result_string))

        self.valid_subject_start_time = time.time()

    def on_validation_end(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        duration = time.time() - self.valid_start_time
        metric_means = {}
        for task in task_context.history.get_tasks():
            for metric in task_context.history.get_entries_keys(task):
                entries = task_context.history.get_entries(metric, task)
                metric_means[metric] = np.mean(entries)

        result_string = ' | '.join('{}: {:.5f}'.format(v, k) for v, k in metric_means.items())
        logging.info('[{}/{}, {:.3}s] {}'.format(task_context.epoch+1, context.config.epochs, duration, result_string))


class ConsoleTestLogHook(TestLoopHook):

    def __init__(self) -> None:
        super().__init__()
        self.test_subject_eval_time = None
        self.overall_subject_start_time = None
        self.new_subject = True
        self.test_start_time = None


    def on_startup(self):
        logging.info('startup')
        self.test_start_time = time.time()

    def end_startup(self, context: ctx.TestContext):
        logging.info('model: \n{}'.format(str(context.model)))
        logging.info('startup finished')

    def on_termination(self, context: ctx.TestContext):
        duration = time.time() - self.test_start_time
        logging.info('\ntesting completed [{:.3}s]'.format(duration))

    def on_test_start(self, task_context: ctx.TaskContext, context: ctx.TestContext):
        logging.info('testing')

    def on_test_batch_start(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        if self.new_subject:
            self.overall_subject_start_time = time.time()

    def on_test_subject_start(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                              context: ctx.TestContext):
        self.test_subject_eval_time = time.time()

    def on_test_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        now = time.time()
        subject_duration = now - self.overall_subject_start_time
        subject_eval_duration = now - self.test_subject_eval_time

        result_string = ' | '.join('{}: {:.5f}'.format(v, k) for v, k in subject_context.metrics.items())
        subject_name = subject_context.subject_index
        if 'subject' in subject_context.subject_data:  # in case the subject index is not the name
            subject_name = subject_context.subject_data['subject']
        logging.info('[{} {:.3}s ({:.3})] {}'.format(subject_name, subject_duration, subject_eval_duration, result_string))


class SaveBestModelHook(TrainLoopHook):

    def __init__(self) -> None:
        super().__init__()
        self.saved_best = None

    def on_epoch_end(self, context: ctx.TrainContext, epoch: int):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        if self.saved_best is None or context.best_score > self.saved_best:
            # delete the existing best
            model_mgt.model_service.delete_checkpoint(context.model_files.weight_checkpoint_dir, 'best')

            context.save_to_checkpoint(epoch, is_best=True)
            self.saved_best = context.best_score


class SaveNLastModelHook(TrainLoopHook):

    def __init__(self, n_last: int) -> None:
        super().__init__()
        self.n_last = n_last

    def on_epoch_end(self, context: ctx.TrainContext, epoch: int):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        to_remove = epoch - self.n_last
        if to_remove >= 0:
            model_mgt.model_service.delete_checkpoint(context.model_files.weight_checkpoint_dir, to_remove)
        context.save_to_checkpoint(epoch, is_best=False)


class WriteValidationMetricsCsvHook(TrainLoopHook):

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name
        self.subject_names = []

    def on_validation_start(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        self.subject_names.clear()

    def on_validation_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                                  context: ctx.TrainContext):
        subject_name = subject_context.subject_index
        if 'subject' in subject_context.subject_data:  # in case the subject index is not the name
            subject_name = subject_context.subject_data['subject']
        self.subject_names.append(subject_name)

    def on_validation_end(self, task_context: ctx.TaskContext, context: ctx.TrainContext):
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        epoch_dir = os.path.join(context.valid_dir, 'epoch_{}'.format(task_context.epoch))
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        file_path = os.path.join(epoch_dir, self.file_name)
        category = 'subject_metrics'

        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            sorted_keys = sorted(task_context.history.get_entries_keys(category))
            writer.writerow(['subject'] + list(sorted_keys))
            for index in range(task_context.history.get_entry_size(sorted_keys[0], category)):
                entries = task_context.history.get_entries_by_index(index, category)
                row = [self.subject_names[index]] + [entries[k] for k in sorted_keys]
                writer.writerow(row)


class WriteTestMetricsCsvHook(TestLoopHook):

    def __init__(self, file_name) -> None:
        super().__init__()
        self.file_name = file_name
        self.subject_names = []

    def on_test_start(self, task_context: ctx.TaskContext, context: ctx.TestContext):
        self.subject_names.clear()

    def on_test_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        subject_name = subject_context.subject_index
        if 'subject' in subject_context.subject_data:  # in case the subject index is not the name
            subject_name = subject_context.subject_data['subject']
        self.subject_names.append(subject_name)

    def on_test_end(self, task_context: ctx.TaskContext, context: ctx.TestContext):
        if not isinstance(context, ctx.TorchTestContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTestContext))

        file_path = os.path.join(context.test_dir, self.file_name)
        category = 'subject_metrics'

        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            sorted_keys = sorted(task_context.history.get_entries_keys(category))
            writer.writerow(['subject'] + list(sorted_keys))
            for index in range(task_context.history.get_entry_size(sorted_keys[0], category)):
                entries = task_context.history.get_entries_by_index(index, category)
                row = [self.subject_names[index]] + [entries[k] for k in sorted_keys]
                writer.writerow(row)


