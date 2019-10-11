import os
import argparse
import logging

import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import pymia.data.assembler as assembler
import pymia.data.conversion as conversion

import common.trainloop.data as data
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import common.utils.labelhelper as lh
import rechun.directories as dirs


def main(config_file):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_aleatoric.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    if not hasattr(context.config.others, 'is_log_sigma'):
        raise ValueError('"is_log_sigma" entry missing in configuration file')
    is_log_sigma = context.config.others.is_log_sigma

    test_steps = [AleatoricPredictStep(is_log_sigma)]
    subject_steps = [step.ExtractSubjectInfoStep(), step.EvalSubjectStep()]

    subject_assembler = assembler.SubjectAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler, entries=None)

    hook = hooks.ReducedComposeTestLoopHook([hooks.ConsoleTestLogHook(),
                                             hooks.WriteTestMetricsCsvHook('metrics.csv'),
                                             WriteHook()
                                             ])
    test(context, build_test, hook=hook)


class AleatoricPredictStep(step.BatchStep):

    def __init__(self, is_log_sigma=False) -> None:
        super().__init__()
        self.is_log_sigma = is_log_sigma

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTestContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTestContext))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)

        mean_logits, sigma = context.model(batch_context.input['images'])
        batch_context.output['logits'] = mean_logits

        if self.is_log_sigma:
            sigma = sigma.exp()
        else:
            sigma = sigma.abs()
        batch_context.output['sigma'] = sigma

        probabilities = F.softmax(batch_context.output['logits'], 1)
        batch_context.output['probabilities'] = probabilities


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

        sigma = subject_context.subject_data['sigma']
        prediction = np.argmax(probabilities, axis=-1)
        sigma = sigma[lh.to_one_hot(prediction).astype(np.bool)].reshape(prediction.shape)

        probabilities = probabilities[..., 1]  # foreground class

        img_probs = subject_context.subject_data['properties']  # type: conversion.ImageProperties

        probability_img = conversion.NumpySimpleITKImageBridge.convert(probabilities, img_probs)
        prediction_img = conversion.NumpySimpleITKImageBridge.convert(predictions, img_probs)
        sigma_img = conversion.NumpySimpleITKImageBridge.convert(sigma, img_probs)

        subject = subject_context.subject_data['subject']
        sitk.WriteImage(probability_img, os.path.join(context.test_dir, '{}_probabilities.nii.gz'.format(subject)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(subject)))
        sitk.WriteImage(sigma_img, os.path.join(context.test_dir, '{}_sigma.nii.gz'.format(subject)))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS test script (aleatoric)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
