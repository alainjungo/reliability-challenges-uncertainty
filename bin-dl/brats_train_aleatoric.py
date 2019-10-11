import argparse
import logging
import os

import pymia.data.assembler as assembler
import torch.nn.functional as F

import common.trainloop.context as ctx
import common.trainloop.loops as trainloop
import common.trainloop.data as data
import common.trainloop.hooks as hooks
import common.trainloop.steps as step
import common.utils.messages as msg
import common.loss as loss
import rechun.directories as dirs


def main(config_file: str):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'train_brats_aleatoric.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    build_train = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
        build_sampler=data.BuildSelectionSampler(),
    )
    build_valid = data.BuildData(
        build_dataset=data.BuildParametrizableDataset(),
    )

    if not hasattr(context.config.others, 'is_log_sigma'):
        raise ValueError('"is_log_sigma" entry missing in configuration file')
    is_log_sigma = context.config.others.is_log_sigma

    train_steps = [TrainStepWithEval(loss.AleatoricLoss(is_log_sigma))]
    train = trainloop.Train(train_steps, only_validate=False)

    subject_assembler = assembler.SubjectAssembler()
    validate = trainloop.ValidateSubject([AleatoricPredictStep(is_log_sigma=is_log_sigma)],
                                         [step.ExtractSubjectInfoStep(), step.EvalSubjectStep()],
                                         subject_assembler, entries=None)

    hook = hooks.ComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(),
                                       hooks.SaveBestModelHook(),
                                       hooks.SaveNLastModelHook(3)])
    train(context, build_train, build_valid, validate, hook=hook)


class TrainStepWithEval(step.BatchStep):

    def __init__(self, criterion) -> None:
        super().__init__()
        self.criterion = criterion

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        context.optimizer.zero_grad()

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        batch_context.input['labels'] = batch_context.input['labels'].long().to(context.device)

        mean_logits, sigma = context.model(batch_context.input['images'])
        batch_context.output['logits'] = mean_logits

        loss = self.criterion(mean_logits, sigma, batch_context.input['labels'])
        loss.backward()
        context.optimizer.step()

        batch_context.metrics['loss'] = loss.item()


class AleatoricPredictStep(step.BatchStep):

    def __init__(self, is_log_sigma=False) -> None:
        super().__init__()
        self.is_log_sigma = is_log_sigma

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)

        mean_logits, sigma = context.model(batch_context.input['images'])
        batch_context.output['logits'] = mean_logits

        if self.is_log_sigma:
            sigma = sigma.exp()
        batch_context.output['sigma'] = sigma

        probabilities = F.softmax(batch_context.output['logits'], 1)
        batch_context.output['probabilities'] = probabilities


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='BraTS training script (aleatoric)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
