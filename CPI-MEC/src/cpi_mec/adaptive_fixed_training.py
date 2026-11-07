"""Train the unchanged CPI-MEC backbone with fixed auxiliary weights.

This module only implements shared initialization and fixed-weight backbone
training. Candidate generation and parameter selection live in search modules.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_VERSION = "1.1.0"
SCRIPT_RELEASE = "fixed_training_epoch_sync_20260729"


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash a model state dictionary."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state to CPU."""
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def reset_backbone(backbone, seed_everything, seed: int) -> None:
    """Reset the relational backbone."""
    seed_everything(seed)
    with torch.no_grad():
        torch.nn.init.normal_(backbone.entity_a.weight, std=0.05)
        torch.nn.init.normal_(backbone.entity_b.weight, std=0.05)
        backbone.relation_scale.fill_(1.0)
        backbone.relation_bias.zero_()


def build_seed_states(
    context,
    base,
    seed_offsets: tuple[int, ...],
    joint_seed: int,
) -> dict[int, dict[str, Any]]:
    """Build shared seed initializations."""
    from training import candidate_scorer, seed_everything

    states = {}
    for seed_offset in seed_offsets:
        seed = joint_seed + seed_offset * 100003
        reset_backbone(context.models.backbone, seed_everything, seed)
        base.refresh_frozen_features(context)
        seed_everything(seed + 1)
        scorer = candidate_scorer(context, learnable_coefficient=True)
        backbone_state = cpu_state(context.models.backbone)
        scorer_state = cpu_state(scorer)
        states[seed_offset] = {
            "seed": seed,
            "backbone": backbone_state,
            "scorer": scorer_state,
            "standardizer": copy.deepcopy(context.standardizer),
            "backbone_sha256": state_sha256(backbone_state),
            "scorer_sha256": state_sha256(scorer_state),
        }
    return states


def train_fixed_weight(
    base,
    context,
    initial_backbone: dict[str, torch.Tensor],
    initial_scorer: dict[str, torch.Tensor],
    train_rows: np.ndarray,
    graphban_weight: float,
    boltz_weight: float,
    epochs: int,
    steps_per_epoch: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Train all epochs with fixed weights."""
    from training import candidate_scorer, relation_sampler, seed_everything

    seed_everything(seed)
    context.models.backbone.load_state_dict(initial_backbone)
    for parameter in context.models.backbone.parameters():
        parameter.requires_grad_(True)
    scorer = candidate_scorer(context, learnable_coefficient=True)
    scorer.load_state_dict(initial_scorer)
    optimizer = torch.optim.Adam(
        [
            {
                "params": context.models.backbone.parameters(),
                "weight_decay": context.settings.backbone_weight_decay,
            },
            {
                "params": scorer.parameters(),
                "weight_decay": context.settings.head_weight_decay,
            },
        ],
        lr=context.settings.backbone_learning_rate,
    )
    sparse_name = context.models.architecture.sparse_features[0]
    sparse_sampler = relation_sampler(
        sparse_name,
        context.matrix_values[sparse_name],
    )
    cpi_supervision = base.prepare_cpi_supervision(
        context,
        train_rows,
    )
    history = []
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        names = (
            "total",
            "main",
            "ranking",
            "pointwise",
            "primary",
            "graphban",
            "boltz",
        )
        totals = torch.zeros(
            len(names),
            dtype=torch.float32,
            device=context.device,
        )
        context.models.backbone.train()
        scorer.train()
        for _ in range(steps_per_epoch):
            main, ranking, pointwise = base.cpi_loss(
                context,
                scorer,
                cpi_supervision,
            )
            primary, graphban, boltz = base.relation_losses(
                context,
                rng,
                sparse_sampler,
            )
            total = (
                main
                + primary
                + graphban_weight * graphban
                + boltz_weight * boltz
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            scorer.project_primary_coefficient()
            totals += torch.stack(
                (
                    total,
                    main,
                    ranking,
                    pointwise,
                    primary,
                    graphban,
                    boltz,
                )
            ).detach()

        values = dict(
            zip(
                names,
                (totals / steps_per_epoch).cpu().numpy(),
            )
        )
        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{name}_loss": float(value)
                    for name, value in values.items()
                },
                "graphban_weighted_contribution": (
                    graphban_weight
                    * float(values["graphban"])
                ),
                "boltz_weighted_contribution": (
                    boltz_weight
                    * float(values["boltz"])
                ),
            }
        )

    for parameter in context.models.backbone.parameters():
        parameter.requires_grad_(False)
    context.backbone_effective_weights = {
        context.models.architecture.auxiliary_feature: graphban_weight,
        context.models.architecture.sparse_features[0]: boltz_weight,
    }
    return pd.DataFrame(history), {
        "selected_epoch": epochs,
        **history[-1],
    }
