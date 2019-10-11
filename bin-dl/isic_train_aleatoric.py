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
import common.utils.messages as msg
import common.loss as loss
import rechun.dl.customdatasets as isic
import rechun.directories as dirs


def main(config_file: str):

    if config_file is None:
        config_file = os.path.join(dirs.CONFIG_DIR, 'train_isic_aleatoric.yaml')

    context = ctx.TorchTrainContext('cuda')
    context.load_from_config(config_file)

    build_train = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )
    build_valid = data.BuildData(
        build_dataset=isic.BuildIsicDataset()
    )

    if not hasattr(context.config.others, 'is_log_sigma'):
        raise ValueError('"is_log_sigma" entry missing in configuration file')
    is_log_sigma = context.config.others.is_log_sigma

    train_steps = [TrainStepWithEval(loss.AleatoricLoss(is_log_sigma))]
    train = trainloop.Train(train_steps, only_validate=False)

    assemble = assembler.Subject2dAssembler()
    validate = trainloop.ValidateSubject([AleatoricPredictStep(is_log_sigma=is_log_sigma)],
                                         [EvalStep()], assemble, convert_fn=None)

    hook = hooks.ReducedComposeTrainLoopHook([hooks.TensorboardXHook(), hooks.ConsoleLogHook(print_subject_results=False),
                                              hooks.SaveBestModelHook(), hooks.SaveNLastModelHook(3)])
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

        batch_context.output['labels'] = batch_context.input['labels'].unsqueeze(1).to(context.device)  # re-add previously removed dim


class EvalStep(step.SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = eval.ComposeEvaluation([eval.SmoothDice('dice')])

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


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='ISIC training script (aleatoric)')
        parser.add_argument('-config_file', type=str, help='the json file name containing the train configuration')
        args = parser.parse_args()
        main(args.config_file)
    finally:
        logging.exception('')  # log the exception
