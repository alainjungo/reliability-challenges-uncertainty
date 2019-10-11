import os
import argparse
import logging

import SimpleITK as sitk
import numpy as np
import pymia.data.assembler as assembler
import pymia.data.conversion as conversion

import common.evalutation.eval as ev
import common.trainloop.data as data
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import rechun.dl.customsteps as customstep
import rechun.directories as dirs


def main(config_file, config_id):

    if config_file is None:
        if config_id == 'baseline':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_baseline.yaml')
        elif config_id == 'baseline_mc':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_baseline_mc.yaml')
        elif config_id == 'center':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_center.yaml')
        elif config_id == 'center_mc':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_center_mc.yaml')
        elif config_id in ('cv0', 'cv1', 'cv2', 'cv3', 'cv4'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'baseline_cv',
                                       'test_brats_baseline_cv{}.yaml'.format(config_id[-1]))
        else:
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_baseline.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    if hasattr(context.config.others, 'mc'):
        test_steps = [customstep.McPredictStep(context.config.others.mc),
                      customstep.MultiPredictionSummary()]
    else:
        test_steps = [step.SegmentationPredictStep(do_probs=True)]
    subject_steps = [step.ExtractSubjectInfoStep(), EvalSubjectStep()]

    subject_assembler = assembler.SubjectAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler, entries=('probabilities',))

    hook = hooks.ReducedComposeTestLoopHook([hooks.ConsoleTestLogHook(),
                                             hooks.WriteTestMetricsCsvHook('metrics.csv'),
                                             WriteHook()
                                             ])
    test(context, build_test, hook=hook)


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': subject_context.subject_data['labels']}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)


class WriteHook(hooks.TestLoopHook):

    def on_test_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        if not isinstance(context, ctx.TorchTestContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTestContext))

        thread.do_work(WriteHook._on_test_subject_end,
                       subject_context, task_context, context, in_background=True)

    @staticmethod
    def _on_test_subject_end(subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                             context: ctx.TestContext):
        if not isinstance(context, ctx.TorchTestContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTestContext))

        probabilities = subject_context.subject_data['probabilities']
        predictions = np.argmax(probabilities, axis=-1).astype(np.uint8)

        probabilities = probabilities[..., 1]  # foreground class

        img_probs = subject_context.subject_data['properties']  # type: conversion.ImageProperties

        probability_img = conversion.NumpySimpleITKImageBridge.convert(probabilities, img_probs)
        prediction_img = conversion.NumpySimpleITKImageBridge.convert(predictions, img_probs)

        subject = subject_context.subject_data['subject']
        sitk.WriteImage(probability_img, os.path.join(context.test_dir, '{}_probabilities.nii.gz'.format(subject)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(subject)))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS test script (default)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        parser.add_argument('-config_id', type=str, help='the id of a known config (is ignored when config_file set)')
        args = parser.parse_args()
        main(args.config_file, args.config_id)
    finally:
        logging.exception('')  # log the exception
