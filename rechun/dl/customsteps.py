import torch
import torch.nn.functional as F

import common.trainloop.steps as step
import common.trainloop.context as ctx
import common.utils.messages as msg
import common.utils.torchhelper as th


class McPredictStep(step.BatchStep):

    def __init__(self, mc_steps) -> None:
        super().__init__()
        self.mc_steps = mc_steps

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:
        if not isinstance(context, (ctx.TorchTrainContext, ctx.TorchTestContext)):
            raise ValueError(msg.get_type_error_msg(context, (ctx.TorchTrainContext, ctx.TorchTestContext)))

        batch_context.input['images'] = batch_context.input['images'].float().to(context.device)

        # weight scaling part, just for comparison
        ws_logits = context.model(batch_context.input['images'])
        ws_probabilities = F.softmax(ws_logits, 1)
        batch_context.output['ws_probabilities'] = ws_probabilities

        th.set_dropout_mode(context.model, is_train=True)

        # mc part
        mc_probabilities = []
        for i in range(self.mc_steps):
            logits = context.model(batch_context.input['images'])
            probs = F.softmax(logits, 1)
            mc_probabilities.append(probs)
        mc_probabilities = torch.stack(mc_probabilities)
        batch_context.output['multi_probabilities'] = mc_probabilities

        # reset to eval for next batch
        th.set_dropout_mode(context.model, is_train=False)


class MultiPredictionSummary(step.BatchStep):

    def __init__(self, do_mi=False, do_var=False, remove_multi_probs=True) -> None:
        super().__init__()
        self.do_mi = do_mi
        self.do_var = do_var
        self.remove_multi_probs = remove_multi_probs

    def __call__(self, batch_context: ctx.BatchContext, task_context: ctx.TaskContext, context: ctx.Context) -> None:

        if self.remove_multi_probs:
            multi_probabilities = batch_context.output.pop('multi_probabilities')
        else:
            multi_probabilities = batch_context.output['multi_probabilities']

        probabilities = multi_probabilities.mean(dim=0)
        batch_context.output['probabilities'] = probabilities

        entropy = th.entropy(probabilities, dim=1, keepdim=True)
        batch_context.output['entropy'] = entropy

        if self.do_mi:
            expected_entropy = th.entropy(multi_probabilities, dim=2, keepdim=True).mean(dim=0)
            mutual_info = entropy - expected_entropy
            batch_context.output['mutual_info'] = mutual_info

        if self.do_var:
            # as done by the bayesian segnet -> not sure if best solution
            variance = multi_probabilities.var(dim=0).mean(dim=1, keepdim=True)
            batch_context.output['variance'] = variance
