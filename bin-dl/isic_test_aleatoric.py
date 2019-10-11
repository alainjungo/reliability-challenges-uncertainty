import os
import argparse
import logging

import torch.nn.functional as F
import numpy as np
import pymia.data.assembler as assembler
import SimpleITK as sitk

import common.trainloop.data as data
import common.evalutation.eval as ev
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import common.utils.labelhelper as lh
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_aleatoric.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=isic.BuildIsicDataset(),
    )

    if not hasattr(context.config.others, 'is_log_sigma'):
        raise ValueError('"is_log_sigma" entry missing in configuration file')
    is_log_sigma = context.config.others.is_log_sigma

    test_steps = [AleatoricPredictStep(is_log_sigma), PrepareSubjectStep()]
    subject_steps = [EvalSubjectStep()]

    subject_assembler = assembler.Subject2dAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler)

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


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': subject_context.subject_data['labels'].squeeze(-1)}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)


class PrepareSubjectStep(step.BatchStep):

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        batch_context.output['labels'] = batch_context.input['labels'].unsqueeze(1)  # re-add previously removed dim


class WriteHook(hooks.TestLoopHook):

    def __del__(self):
        thread.join_all()
        print('joined....')

    def on_test_subject_end(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                            context: ctx.TestContext):
        if not isinstance(context, ctx.TorchTestContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTestContext))

        thread.do_work(WriteHook._on_test_subject_end,
                       subject_context, task_context, context, in_background=True)

    @staticmethod
    def _on_test_subject_end(subject_context: ctx.SubjectContext, task_context: ctx.TaskContext,
                             context: ctx.TorchTestContext):
        probabilities = subject_context.subject_data['probabilities']
        predictions = np.argmax(probabilities, axis=-1).astype(np.uint8)

        sigma = subject_context.subject_data['sigma']
        prediction = np.argmax(probabilities, axis=-1)
        sigma = sigma[lh.to_one_hot(prediction).astype(np.bool)].reshape(prediction.shape)

        probabilities = probabilities[..., 1]  # foreground class

        id_ = subject_context.subject_index

        probability_img = sitk.GetImageFromArray(probabilities)
        prediction_img = sitk.GetImageFromArray(predictions)
        sigma_img = sitk.GetImageFromArray(sigma)

        sitk.WriteImage(probability_img, os.path.join(context.test_dir, '{}_probabilities.nii.gz'.format(id_)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(id_)))
        sitk.WriteImage(sigma_img, os.path.join(context.test_dir, '{}_sigma.nii.gz'.format(id_)))

        files = context.test_data.dataset.get_files_by_id(id_)
        label_path = os.path.abspath(files['label_paths'])
        label_out_path = os.path.join(context.test_dir, os.path.basename(label_path))
        os.symlink(label_path, label_out_path)
        image_path = os.path.abspath(files['image_paths'])
        image_out_path = os.path.join(context.test_dir, os.path.basename(image_path))
        os.symlink(image_path, image_out_path)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC test script (aleatoric)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
