import os
import argparse
import logging

import SimpleITK as sitk
import numpy as np
import pymia.data.assembler as assembler
import pymia.data.conversion as conversion
import torch.nn.functional as F

import common.evalutation.eval as ev
import common.trainloop.data as data
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import common.model.management as mgt
import rechun.directories as dirs


def main(config_file):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'test_brats_auxiliary_feat.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    if hasattr(context.config.others, 'model_dir') and hasattr(context.config.others, 'test_at'):
        mf = mgt.ModelFiles.from_model_dir(context.config.others.model_dir)
        checkpoint_path = mgt.model_service.find_checkpoint_file(mf.weight_checkpoint_dir, context.config.others.test_at)

        model = mgt.model_service.load_model_from_parameters(mf.model_path(), with_optimizer=False)
        model.provide_features = True
        mgt.model_service.load_checkpoint(checkpoint_path, model)
        test_model = model.to(context.device)

        test_model.eval()
        for params in test_model.parameters():
            params.requires_grad = False

    test_steps = [SegmentationPredictStep(test_model)]
    subject_steps = [step.ExtractSubjectInfoStep(), EvalSubjectStep()]

    subject_assembler = assembler.SubjectAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler, entries=('probabilities', 'segm_probabilities'))

    hook = hooks.ReducedComposeTestLoopHook([hooks.ConsoleTestLogHook(),
                                             hooks.WriteTestMetricsCsvHook('metrics.csv'),
                                             WriteHook()
                                             ])
    test(context, build_test, hook=hook)


class SegmentationPredictStep(step.BatchStep):

    def __init__(self, test_model) -> None:
        super().__init__()
        self.test_model = test_model

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, (ctx.TorchTrainContext, ctx.TorchTestContext)):
            raise ValueError(msg.get_type_error_msg(context, (ctx.TorchTrainContext, ctx.TorchTestContext)))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)

        segm_logits = self.test_model(batch_context.input['images'])
        segm_probabilities = F.softmax(segm_logits, 1)
        batch_context.output['segm_probabilities'] = segm_probabilities

        logits = context.model(self.test_model.features)

        probabilities = F.softmax(logits, 1)
        batch_context.output['probabilities'] = probabilities


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['segm_probabilities']
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

        probabilities = subject_context.subject_data['segm_probabilities']
        predictions = np.argmax(probabilities, axis=-1).astype(np.uint8)

        confidence = subject_context.subject_data['probabilities']
        confidence = confidence[..., 1]  # foreground confidence

        img_probs = subject_context.subject_data['properties']  # type: conversion.ImageProperties

        confidence_img = conversion.NumpySimpleITKImageBridge.convert(confidence, img_probs)
        prediction_img = conversion.NumpySimpleITKImageBridge.convert(predictions, img_probs)

        subject = subject_context.subject_data['subject']
        sitk.WriteImage(confidence_img, os.path.join(context.test_dir, '{}_confidence.nii.gz'.format(subject)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(subject)))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS test script (auxiliary feat.)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
