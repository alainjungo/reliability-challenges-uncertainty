import os
import argparse
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pymia.data.assembler as assembler

import common.trainloop.loops as trainloop
import common.trainloop.data as data
import common.trainloop.hooks as hooks
import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.evalutation.eval as ev
import common.utils.messages as msg
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file: str):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_auxiliary_segm.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    if not hasattr(context.config.others, 'prediction_dir'):
        raise ValueError('"others.prediction_dir" is required in the config')
    prediction_dir = context.config.others.prediction_dir

    build_train = data.BuildData(
        build_dataset=isic.BuildIsicDataset(),
        prediction_dir=prediction_dir
    )
    build_valid = data.BuildData(
        build_dataset=isic.BuildIsicDataset(),
        prediction_dir=prediction_dir
    )

    train_steps = [SpecialTrainStep(), step.EvalStep()]
    train = trainloop.Train(train_steps, only_validate=False)

    assemble = assembler.Subject2dAssembler()
    validate = trainloop.ValidateSubject([SpecialSegmentationPredictStep()],
                                         [EvalSubjectStep()], assemble, entries=('probabilities', 'labels'))

    hook = hooks.ReducedComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(print_subject_results=False),
                                              hooks.SaveBestModelHook(), hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


class SpecialSegmentationPredictStep(step.BatchStep):

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
        # subject_eval needs clean (non-modified) labels
        batch_context.output['labels'] = batch_context.input['labels']


class SpecialTrainStep(step.BatchStep):

    def __init__(self, criterion=nn.CrossEntropyLoss()) -> None:
        super().__init__()
        self.criterion = criterion

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        context.optimizer.zero_grad()

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        batch_context.input['labels'] = batch_context.input['labels'].long().to(context.device)

        prediciton = batch_context.input['labels'][:, 1, ...]
        gt = batch_context.input['labels'][:, 0, ...]

        labels = (prediciton != gt).long()
        # update with correct label for the evaluation
        batch_context.input['labels'] = labels

        inpt = torch.cat([batch_context.input['images'], prediciton.unsqueeze(1).float()], dim=1)
        logits = context.model(inpt)
        batch_context.output['logits'] = logits

        loss = self.criterion(logits, labels)
        loss.backward()
        context.optimizer.step()

        batch_context.metrics['loss'] = loss.item()


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy(), ev.LogLossSklearn()])

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
        subject_context.score = results['dice']


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC training script (auxiliary segm.)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
