import os
import argparse
import logging

import torch.nn.functional as F
import pymia.data.assembler as assembler

import common.trainloop.loops as trainloop
import common.trainloop.data as data
import common.trainloop.hooks as hooks
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.evalutation.eval as eval
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file: str, config_id: str):

    if config_file is None:
        if config_id == 'baseline':
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_baseline.yaml')
        elif config_id == 'center':
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_center.yaml')
        elif config_id in ('cv0', 'cv1', 'cv2', 'cv3', 'cv4'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'baseline_cv',
                                       'train_isic_baseline_cv{}.yaml'.format(config_id[-1]))
        elif config_id in ('ensemble0', 'ensemble1', 'ensemble2', 'ensemble3', 'ensemble4', 'ensemble5', 'ensemble6',
                           'ensemble7', 'ensemble8', 'ensemble9'):
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_ensemble',
                                       'train_isic_ensemble_{}.yaml'.format(config_id[-1]))
        else:
            config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_baseline.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    build_train = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )
    build_valid = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )

    train_steps = [step.TrainStep(), step.EvalStep()]
    train = trainloop.Train(train_steps, only_validate=False)

    assemble = assembler.Subject2dAssembler()
    validate = trainloop.ValidateSubject([step.SegmentationPredictStep(has_labels=True), PrepareSubjectStep()],
                                         [EvalStep()], assemble, convert_fn=None)

    hook = hooks.ReducedComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(print_subject_results=False),
                                              hooks.SaveBestModelHook(), hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


class EvalStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = eval.ComposeEvaluation([eval.SmoothDice('dice'), eval.Nll()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        _, prediction = probabilities.max(-1)

        to_eval = {'prediction': prediction,
                   'target': subject_context.subject_data['labels'].squeeze(),
                   'probabilities': probabilities}
        results = {}
        self.evaluate(to_eval, results)

        subject_context.metrics.update(results)
        subject_context.score = results['dice']


class PrepareSubjectStep(step.BatchStep):

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = F.softmax(batch_context.output['logits'], 1)
        batch_context.output['probabilities'] = probabilities
        batch_context.output['labels'] = batch_context.input['labels'].unsqueeze(1)  # re-add previously removed dim


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC training script (default)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        parser.add_argument('-config_id', type=str, help='the id of a known config (is ignored when config_file set)')
        args = parser.parse_args()
        main(args.config_file, args.config_id)
    finally:
        logging.exception('')  # log the exception
