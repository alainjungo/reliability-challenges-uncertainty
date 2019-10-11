import os
import argparse
import logging

import pymia.data.assembler as assembler
import numpy as np

import common.trainloop.context as ctx
import common.trainloop.loops as trainloop
import common.trainloop.data as data
import common.trainloop.hooks as hooks
import common.trainloop.steps as step
import common.evalutation.eval as ev
import rechun.directories as dirs


def main(config_file: str, config_id: str):

    if config_file is None:
        if config_id == 'baseline':
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_brats_baseline.yaml')
        elif config_id == 'center':
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_brats_center.yaml')
        elif config_id in ('cv0', 'cv1', 'cv2', 'cv3', 'cv4'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'baseline_cv',
                                       'train_brats_baseline_cv{}.yaml'.format(config_id[-1]))
        elif config_id in ('ensemble0', 'ensemble1', 'ensemble2', 'ensemble3', 'ensemble4', 'ensemble5', 'ensemble6',
                           'ensemble7', 'ensemble8', 'ensemble9'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_ensemble',
                                       'train_brats_ensemble_{}.yaml'.format(config_id[-1]))
        else:
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_brats_baseline.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    build_train = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
        build_sampler=data.BuildSelectionSampler(),
    )
    build_valid = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    train_steps = [step.TrainStep(), step.EvalStep()]
    train = trainloop.Train(train_steps, only_validate=False)

    subject_assembler = assembler.SubjectAssembler()
    validate = trainloop.ValidateSubject([step.SegmentationPredictStep(do_probs=True)],
                                         [step.ExtractSubjectInfoStep(), EvalSubjectStep()],
                                         subject_assembler, entries=('probabilities',))

    hook = hooks.ComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(), hooks.SaveBestModelHook(),
                                       hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy(), ev.LogLossSklearn()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': subject_context.subject_data['labels']}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)
        subject_context.score = results['dice']


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS training script (default)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        parser.add_argument('-config_id', type=str, help='the id of a known config (is ignored when config_file set)')
        args = parser.parse_args()
        main(args.config_file, args.config_id)
    finally:
        logging.exception('')  # log the exception
