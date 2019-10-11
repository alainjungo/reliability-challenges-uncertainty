import os
import argparse
import logging

import SimpleITK as sitk
import numpy as np
import pymia.data.assembler as assembler

import common.evalutation.eval as ev
import common.trainloop.data as data
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.trainloop.hooks as hooks
import common.trainloop.loops as loop
import common.utils.messages as msg
import common.utils.threadhelper as thread
import rechun.dl.customsteps as customstep
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file, config_id):

    if config_file is None:
        if config_id == 'baseline':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_baseline.yaml')
        elif config_id == 'baseline_mc':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_baseline_mc.yaml')
        elif config_id == 'center':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_center.yaml')
        elif config_id == 'center_mc':
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_center_mc.yaml')
        elif config_id in ('cv0', 'cv1', 'cv2', 'cv3', 'cv4'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'baseline_cv',
                                       'test_isic_baseline_cv{}.yaml'.format(config_id[-1]))
        else:
            config_file = os.path.join(dirs.CONFIG_DIR, 'test_isic_baseline.yaml')

    context = ctx.TorchTestContext('cuda')
    context.load_from_config(config_file)

    build_test = data.BuildData(
        build_dataset=isic.BuildIsicDataset(),
    )

    if hasattr(context.config.others, 'mc'):
        test_steps = [customstep.McPredictStep(context.config.others.mc),
                      customstep.MultiPredictionSummary()]
    else:
        test_steps = [step.SegmentationPredictStep(do_probs=True)]
    test_steps.append(PrepareSubjectStep())
    subject_steps = [EvalSubjectStep()]

    subject_assembler = assembler.Subject2dAssembler()
    test = loop.Test(test_steps, subject_steps, subject_assembler)

    hook = hooks.ReducedComposeTestLoopHook([hooks.ConsoleTestLogHook(),
                                             hooks.WriteTestMetricsCsvHook('metrics.csv'),
                                             WriteHook()
                                             ])
    test(context, build_test, hook=hook)


class PrepareSubjectStep(step.BatchStep):

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        batch_context.output['labels'] = batch_context.input['labels'].unsqueeze(1)  # re-add previously removed dim


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        subject_context.subject_data['prediction'] = prediction

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': subject_context.subject_data['labels'].squeeze(-1)}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)


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
        probabilities = probabilities[..., 1]  # foreground probabilities
        predictions = subject_context.subject_data['prediction'].astype(np.uint8)

        id_ = subject_context.subject_index

        probability_img = sitk.GetImageFromArray(probabilities)
        prediction_img = sitk.GetImageFromArray(predictions)

        sitk.WriteImage(probability_img, os.path.join(context.test_dir, '{}_probabilities.nii.gz'.format(id_)))
        sitk.WriteImage(prediction_img, os.path.join(context.test_dir, '{}_prediction.nii.gz'.format(id_)))

        files = context.test_data.dataset.get_files_by_id(id_)
        label_path = os.path.abspath(files['label_paths'])
        label_out_path = os.path.join(context.test_dir, os.path.basename(label_path))
        os.symlink(label_path, label_out_path)
        image_path = os.path.abspath(files['image_paths'])
        image_out_path = os.path.join(context.test_dir, os.path.basename(image_path))
        os.symlink(image_path, image_out_path)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC test script (default)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        parser.add_argument('-config_id', type=str, help='the id of a known config (is ignored when config_file set)')
        args = parser.parse_args()
        main(args.config_file, args.config_id)
    finally:
        logging.exception('')  # log the exception
