import os
import argparse
import logging

import SimpleITK as sitk
import numpy as np
import pymia.data.assembler as assembler
import pymia.data.conversion as conversion
import torch
import torch.nn.functional as F

import common.evalutation.eval as ev
import common.trainloop.data as data
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import rechun.directories as dirs


def main(config_file):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_auxiliary_segm.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    test_steps = [SegmentationPredictStep()]
    subject_steps = [step.ExtractSubjectInfoStep(), EvalSubjectStep()]

    subject_assembler = assembler.SubjectAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler, entries=('probabilities', 'orig_prediction'))

    hook = hooks.ReducedComposeTestLoopHook([hooks.ConsoleTestLogHook(),
                                             hooks.WriteTestMetricsCsvHook('metrics.csv'),
                                             WriteHook()
                                             ])
    test(context, build_test, hook=hook)


class SegmentationPredictStep(step.BatchStep):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, (ctx.TorchTrainContext, ctx.TorchTestContext)):
            raise ValueError(msg.get_type_error_msg(context, (ctx.TorchTrainContext, ctx.TorchTestContext)))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        batch_context.input['labels'] = batch_context.input['labels'].long().to(context.device)

        pred = batch_context.input['labels'][:, 1]
        inpt = torch.cat([batch_context.input['images'], pred.unsqueeze(1).float()], dim=1)

        logits = context.model(inpt)
        batch_context.output['logits'] = logits

        probabilities = F.softmax(logits, 1)
        batch_context.output['probabilities'] = probabilities
        # add the existing prediction to be reproduced
        batch_context.output['orig_prediction'] = pred.unsqueeze(1)


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        pred = subject_context.subject_data['labels'][..., 1]
        gt = subject_context.subject_data['labels'][..., 0]
        target = pred != gt

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': target}
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

        conficence = subject_context.subject_data['probabilities']
        confidence = conficence[..., 1]  # foreground class
        prediction = subject_context.subject_data['orig_prediction']

        img_probs = subject_context.subject_data['properties']  # type: conversion.ImageProperties

        confidence_img = conversion.NumpySimpleITKImageBridge.convert(confidence, img_probs)
        prediction_img = conversion.NumpySimpleITKImageBridge.convert(prediction, img_probs)

        subject = subject_context.subject_data['subject']
        sitk.WriteImage(confidence_img, os.path.join(context.test_dir, '{}_confidence.nii.gz'.format(subject)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(subject)))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS test script (auxiliary segm)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
