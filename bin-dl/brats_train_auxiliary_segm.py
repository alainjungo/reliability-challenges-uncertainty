import os
import argparse
import logging

import pymia.data.assembler as assembler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common.trainloop.context as ctx
import common.trainloop.loops as trainloop
import common.trainloop.data as data
import common.trainloop.hooks as hooks
import common.trainloop.steps as step
import common.evalutation.eval as ev
import common.utils.messages as msg
import rechun.directories as dirs


def main(config_file: str):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'train_brats_auxiliary_segm.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    build_train = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
        build_sampler=data.BuildSelectionSampler(),
    )
    build_valid = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    train_steps = [SpecialTrainStep(), step.EvalStep()]
    train = trainloop.Train(train_steps, only_validate=False)

    subject_assembler = assembler.SubjectAssembler()
    validate = trainloop.ValidateSubject([SpecialSegmentationPredictStep()],
                                         [step.ExtractSubjectInfoStep(), EvalSubjectStep()],
                                         subject_assembler, entries=('probabilities',))

    hook = hooks.ComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(), hooks.SaveBestModelHook(),
                                       hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


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


class SpecialSegmentationPredictStep(step.BatchStep):

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


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy(),
                                              # ev2.EntropyEval(entropy_threshold=0.7, with_ratios=False),
                                              ev.LogLossSklearn()])

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
        parser = argparse.ArgumentParser(description='BraTS training script (auxiliary segm.)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
