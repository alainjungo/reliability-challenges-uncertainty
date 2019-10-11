import logging

import torch
import numpy as np

import common.utils.torchhelper as th
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.data as data


class Validate:

    def __init__(self, steps: list) -> None:
        self.steps = steps
        self.score_aggregation_fn = np.mean

    def __call__(self, context: ctx.TrainContext, hook: hooks.TrainLoopHook, epoch: int):
        if not context.need_validation(epoch):
            return

        context.set_mode(is_train=False)

        task_context = context.get_task_context(epoch)
        hook.on_validation_start(task_context, context)

        for i, batch in enumerate(task_context.data.loader):
            batch_context = ctx.BatchContext(batch, i)
            hook.on_validation_batch_start(batch_context, task_context, context)
            self.validate_batch(batch_context, task_context, context, hook)
            hook.on_validation_batch_end(batch_context, task_context, context)

        score = self.score_aggregation_fn(task_context.scores)
        if context.best_score is None or score > context.best_score:
            context.best_score = score

        hook.on_validation_end(task_context, context)

    def validate_batch(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.TrainContext,
                       hook: hooks.TrainLoopHook):
        for step in self.steps:
            step(batch_context, task_context, context)
        if batch_context.metrics:
            task_context.history.add(batch_context.metrics, 'batch_metrics')

        if batch_context.score is None:
            raise ValueError('"score" must be set in BatchContext')
        task_context.scores.append(batch_context.score)


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.cpu().numpy()


class ValidateSubject(Validate):

    def __init__(self, steps: list, subject_steps: list, subject_assembler, entries: tuple = None,
                 convert_fn=tensor_to_numpy, transform_fn=th.channel_to_end) -> None:
        super().__init__(steps)
        self.subject_steps = subject_steps
        self.subject_assembler = subject_assembler
        self.entries = entries
        self.convert_fn = convert_fn
        self.transform_fn = transform_fn

    def validate_batch(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.TrainContext,
                       hook: hooks.TrainLoopHook):

        for batch_step in self.steps:
            batch_step(batch_context, task_context, context)
        if batch_context.metrics:
            task_context.history.add(batch_context.metrics, 'batch_metrics')

        to_assemble = {}
        for key, value in batch_context.output.items():
            if self.entries is None or key in self.entries:
                if self.transform_fn is not None:
                    value = self.transform_fn(value)
                if self.convert_fn:
                    value = self.convert_fn(value)
                to_assemble[key] = value

        is_last_batch = batch_context.batch_index == task_context.data.nb_batches - 1
        self.subject_assembler.add_batch(to_assemble, batch_context.input, last_batch=is_last_batch)

        if self.subject_assembler.subjects_ready:
            for subject_index in list(self.subject_assembler.subjects_ready):
                subject_data = self.subject_assembler.get_assembled_subject(subject_index)

                subject_context = ctx.SubjectContext(subject_index, subject_data)
                hook.on_validation_subject_start(subject_context, task_context, context)
                for subject_step in self.subject_steps:
                    subject_step(subject_context, task_context, context)
                if subject_context.metrics:
                    task_context.history.add(subject_context.metrics, 'subject_metrics')
                if subject_context.score is None:
                    raise ValueError('"score" must be set in SubjectContext')
                task_context.scores.append(subject_context.score)
                hook.on_validation_subject_end(subject_context, task_context, context)


class Train:

    def __init__(self, steps: list, only_validate=False) -> None:
        self.steps = steps
        self.only_validate = only_validate

    def __call__(self, context: ctx.TrainContext, build_train: data.BuildData, build_valid: data.BuildData,
                 validate: Validate, hook: hooks.TrainLoopHook = hooks.TrainLoopHook()):
        hook.on_startup()

        resume_at = context.get_resume_at()
        if resume_at is None:
            context.setup_directory()

        context.setup_logging()

        seed = context.get_seed()
        if seed is not None:
            context.do_seed(seed)

        context.load_train_and_valid_data(build_train, build_valid)

        if resume_at is None:
            logging.info('build new model')
            context.load_from_new()
        else:
            context.load_from_checkpoint(resume_at)

        hook.end_startup(context)

        first_epoch = 0
        if resume_at is not None:
            first_epoch = resume_at + 1

        for epoch in range(first_epoch, context.config.epochs):
            hook.on_epoch_start(context, epoch)
            if not self.only_validate:
                self._train(context, hook, epoch)

            validate(context, hook, epoch)
            hook.on_epoch_end(context, epoch)

        hook.on_termination(context)

    def _train(self, context: ctx.TrainContext, hook: hooks.TrainLoopHook, epoch: int):
        context.set_mode(is_train=True)

        seed = context.get_seed()
        if seed is not None and epoch != 0:  # epoch 0 already seeded with startup
            context.do_seed(seed + epoch)

        task_context = context.get_task_context(epoch)
        hook.on_training_start(task_context, context)

        for i, batch in enumerate(task_context.data.loader):
            batch_context = ctx.BatchContext(batch, i)
            hook.on_training_batch_start(batch_context, task_context, context)
            for step in self.steps:
                step(batch_context, task_context, context)
            hook.on_training_batch_end(batch_context, task_context, context)
        hook.on_training_end(task_context, context)


class Test:

    def __init__(self, steps: list, subject_steps: list = None, subject_assembler=None, entries: tuple = None,
                 convert_fn=tensor_to_numpy) -> None:
        self.steps = steps
        self.subject_steps = subject_steps
        self.subject_assembler = subject_assembler
        self.entries = entries
        self.convert_fn = convert_fn
        self.channel_to_end_fn = th.channel_to_end

    def __call__(self, context: ctx.TestContext, build_test: data.BuildData,
                 hook: hooks.TestLoopHook = hooks.TestLoopHook()):
        hook.on_startup()

        context.setup_directory()
        context.setup_logging()

        seed = context.get_seed()
        if seed is not None:
            context.do_seed(seed)

        context.load_test_data(build_test)
        context.load_from_checkpoint(context.get_test_at())

        hook.end_startup(context)

        task_context = context.get_task_context()
        hook.on_test_start(task_context, context)

        for i, batch in enumerate(task_context.data.loader):
            batch_context = ctx.BatchContext(batch, i)
            hook.on_test_batch_start(batch_context, task_context, context)
            self._test_batch(batch_context, task_context, context, hook)
            hook.on_test_batch_end(batch_context, task_context, context)

        hook.on_test_end(task_context, context)
        hook.on_termination(context)

    def _test_batch(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.TestContext,
                    hook: hooks.TestLoopHook):
        for batch_step in self.steps:
            batch_step(batch_context, task_context, context)
        if batch_context.metrics:
            task_context.history.add(batch_context.metrics, 'batch_metrics')

        if self.subject_assembler is None:
            return

        to_assemble = {}
        for key, value in batch_context.output.items():
            if self.entries is None or key in self.entries:
                value = self.channel_to_end_fn(value)
                if self.convert_fn:
                    value = self.convert_fn(value)
                to_assemble[key] = value

        is_last_batch = batch_context.batch_index == task_context.data.nb_batches - 1
        self.subject_assembler.add_batch(to_assemble, batch_context.input, last_batch=is_last_batch)

        if self.subject_assembler.subjects_ready:
            for subject_index in list(self.subject_assembler.subjects_ready):
                subject_data = self.subject_assembler.get_assembled_subject(subject_index)

                subject_context = ctx.SubjectContext(subject_index, subject_data)
                hook.on_test_subject_start(subject_context, task_context, context)
                for subject_step in self.subject_steps:
                    subject_step(subject_context, task_context, context)
                if subject_context.metrics:
                    task_context.history.add(subject_context.metrics, 'subject_metrics')
                hook.on_test_subject_end(subject_context, task_context, context)

