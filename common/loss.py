import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F


class AleatoricLoss(nn.Module):

    def __init__(self, is_log_sigma, nb_samples=10):
        super().__init__()
        self.is_log_sigma = is_log_sigma
        self.nb_samples = nb_samples

    def forward(self, logits, sigma, target):
        if self.is_log_sigma:
            distribution = d.Normal(logits, torch.exp(sigma))
        else:
            distribution = d.Normal(logits, sigma)

        x_hat = distribution.rsample((self.nb_samples,))

        mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
        log_probs = mc_expectation.log()
        loss = F.nll_loss(log_probs, target)

        return loss

