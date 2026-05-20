# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn



def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    reference_chosen_logits: torch.Tensor,
    reference_rejected_logits: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_inputs (ChosenRejectedOutputs): Policy log-probs and logits required for the calculation.
        reference_inputs (ChosenRejectedOutputs): Reference log-probs and logits required for the calculation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
            - losses: The DPO loss for each example in the batch.
            - chosen_rewards: Rewards for the chosen responses.
            - rejected_rewards: Rewards for the rejected responses.

    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    chosen_rewards = (
        beta
        * (policy_chosen_logps - reference_chosen_logps).detach()
    )
    rejected_rewards = (
        beta
        * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def build_dpo_loss(job_config: JobConfig):
    loss_fn = dpo_loss
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


