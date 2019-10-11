import abc

import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import common.evalutation.eval as eval
import common.trainloop.context as ctx
import common.trainloop.factory as factory
import common.utils.messages as msg
import common.utils.torchhelper as th


class BatchStep(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        pass


class TrainStep(BatchStep):

    def __init__(self, criterion=nn.CrossEntropyLoss()) -> None:
        super().__init__()
        self.criterion = criterion

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, ctx.TorchTrainContext):
            raise ValueError(msg.get_type_error_msg(context, ctx.TorchTrainContext))

        context.optimizer.zero_grad()

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        batch_context.input['labels'] = batch_context.input['labels'].long().to(context.device)

        logits = context.model(batch_context.input['images'])
        batch_context.output['logits'] = logits

        loss = self.criterion(logits, batch_context.input['labels'])
        loss.backward()
        context.optimizer.step()

        batch_context.metrics['loss'] = loss.item()


class EvalStep(BatchStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = eval.SmoothDice('dice')

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = F.softmax(batch_context.output['logits'], 1)
        probabilities = th.channel_to_end(probabilities).contiguous()
        _, prediction = probabilities.max(-1)

        batch_context.output['probabilities'] = probabilities
        batch_context.output['prediction'] = prediction

        to_eval = {'prediction': prediction,
                   'probabilities': probabilities,
                   'target': batch_context.input['labels']
                   }
        results = {}
        self.evaluate(to_eval, results)
        batch_context.metrics.update(results)
        batch_context.score = results['dice']


class SegmentationPredictStep(BatchStep):

    def __init__(self, has_labels=False, do_probs=False) -> None:
        super().__init__()
        self.has_labels = has_labels
        self.do_probs = do_probs

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, (ctx.TorchTrainContext, ctx.TorchTestContext)):
            raise ValueError(msg.get_type_error_msg(context, (ctx.TorchTrainContext, ctx.TorchTestContext)))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)
        if self.has_labels:
            batch_context.input['labels'] = batch_context.input['labels'].long().to(context.device)

        logits = context.model(batch_context.input['images'])
        batch_context.output['logits'] = logits

        if self.do_probs:
            probabilities = F.softmax(logits, 1)
            batch_context.output['probabilities'] = probabilities


class SubjectStep(abc.ABC):

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        pass


class ExtractSubjectInfoStep(SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.direct_extractor = None
        self.direct_transform = None

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if self.direct_extractor is None:
            self.direct_extractor = factory.get_extractor(task_context.data_config.direct_extractor)
            self.direct_transform = factory.get_transform(task_context.data_config.direct_transform)

        extracted = task_context.data.dataset.direct_extract(self.direct_extractor, subject_context.subject_index,
                                                             transform=self.direct_transform)

        for key, value in extracted.items():
            subject_context.subject_data[key] = extracted[key]


class EvalSubjectStep(SubjectStep):

    def __init__(self) -> None:
        super().__init__()
        self.evaluate = eval.DiceNumpy()

    def __call__(self, subject_context: ctx.SubjectContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        probabilities = subject_context.subject_data['probabilities']
        prediction = np.argmax(probabilities, axis=-1)

        to_eval = {'prediction': prediction, 'probabilities': probabilities,
                   'target': subject_context.subject_data['labels']}
        results = {}
        self.evaluate(to_eval, results)
        subject_context.metrics.update(results)
        subject_context.score = results['dice']

