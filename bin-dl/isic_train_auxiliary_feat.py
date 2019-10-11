import os
import argparse
import logging

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
import common.model.management as mgt
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file: str):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_auxiliary_feat.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    if hasattr(context.config.others, 'model_dir') and hasattr(context.config.others, 'test_at'):
        mf = mgt.ModelFiles.from_model_dir(context.config.others.model_dir)
        checkpoint_path = mgt.model_service.find_checkpoint_file(mf.weight_checkpoint_dir,
                                                                 context.config.others.test_at)

        model = mgt.model_service.load_model_from_parameters(mf.model_path(), with_optimizer=False)
        model.provide_features = True
        mgt.model_service.load_checkpoint(checkpoint_path, model)
        test_model = model.to(context.device)

        test_model.eval()
        for params in test_model.parameters():
            params.requires_grad = False

    build_train = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )
    build_valid = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )

    train_steps = [SpecialTrainStep(test_model), step.EvalStep()]
    train = trainloop.Train(train_steps, only_validate=False)

    assemble = assembler.Subject2dAssembler()
    validate = trainloop.ValidateSubject([SpecialSegmentationPredictStep(test_model), PrepareSubjectStep()],
                                         [EvalSubjectStep()], assemble, entries=('probabilities', 'net_predictions',
                                                                                 'labels'))

    hook = hooks.ReducedComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(print_subject_results=False),
                                              hooks.SaveBestModelHook(), hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


class SpecialSegmentationPredictStep(step.BatchStep):

    def __init__(self, test_model) -> None:
        super().__init__()
        self.test_model = test_model

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, (ctx.TorchTrainContext, ctx.TorchTestContext)):
            raise ValueError(msg.get_type_error_msg(context, (ctx.TorchTrainContext, ctx.TorchTestContext)))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)

        net1_logits = self.test_model(batch_context.input['images'])
        net_prediction = net1_logits.argmax(dim=1, keepdim=True)
        batch_context.output['net_predictions'] = net_prediction

        logits = context.model(self.test_model.features)
        probabilities = F.softmax(logits, 1)
        batch_context.output['probabilities'] = probabilities


class SpecialTrainStep(step.BatchStep):

    def __init__(self, test_model, criterion=nn.CrossEntropyLoss()) -> None:
        super().__init__()
        self.test_model = test_model
        self.criterion = criterion

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        context.optimizer.zero_grad()

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        gt = batch_context.input['labels'].long().to(context.device)

        net1_logits = self.test_model(batch_context.input['images'])
        net_prediction = net1_logits.argmax(dim=1)

        batch_context.input['labels'] = (net_prediction != gt).long()

        logits = context.model(self.test_model.features)
        batch_context.output['logits'] = logits

        loss = self.criterion(logits, batch_context.input['labels'])
        loss.backward()
        context.optimizer.step()

        batch_context.metrics['loss'] = loss.item()


class EvalSubjectStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = ev.ComposeEvaluation([ev.DiceNumpy(), ev.LogLossSklearn()])

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        net_predictions = subject_context.subject_data['net_predictions']

        target = net_predictions.squeeze(-1) != subject_context.subject_data['labels'].squeeze(-1)

        prediction = np.argmax(probabilities, axis=-1)

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': target}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)
        subject_context.score = results['dice']


class PrepareSubjectStep(step.BatchStep):

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        batch_context.output['labels'] = batch_context.input['labels'].unsqueeze(1)  # re-add previously removed dim


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC training script (auxiliary feat.)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
