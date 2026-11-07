"""Fit independent CPI-MEC residual heads in one small-GPU model batch.

Anchor candidates sharing one frozen backbone train along a model dimension.
Every head keeps independent parameters while using the same initialization,
sampled batches, and optimizer settings as the serial implementation.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_VERSION = "1.2.0"
SCRIPT_RELEASE = "single_upload_schedule_no_epoch_sync_20260729"


class BatchedResidualScorer(nn.Module):
    """Evaluate independent heads."""

    def __init__(
        self,
        template,
        n_models: int,
        learnable_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        linear = template.shared[0]
        norm = template.shared[2]
        residual = template.residual_head
        self.coefficient_bounds = template.coefficient_bounds
        self.linear_weight = nn.Parameter(
            linear.weight.detach().repeat(n_models, 1, 1)
        )
        self.linear_bias = nn.Parameter(
            linear.bias.detach().repeat(n_models, 1)
        )
        self.norm_weight = nn.Parameter(
            norm.weight.detach().repeat(n_models, 1)
        )
        self.norm_bias = nn.Parameter(
            norm.bias.detach().repeat(n_models, 1)
        )
        self.residual_weight = nn.Parameter(
            residual.weight.detach().squeeze(0).repeat(n_models, 1)
        )
        self.residual_bias = nn.Parameter(
            residual.bias.detach().repeat(n_models)
        )
        self.binary_bias = nn.Parameter(
            template.binary_bias.detach().repeat(n_models)
        )
        self.binary_scale_raw = nn.Parameter(
            template.binary_scale_raw.detach().repeat(n_models)
        )
        self.primary_coefficient_raw = nn.Parameter(
            template.primary_coefficient_raw.detach().repeat(n_models)
        )
        self.register_buffer("learnable_mask", learnable_mask.float())

    def hidden(self, features: torch.Tensor) -> torch.Tensor:
        """Compute model-specific hidden states."""
        if features.ndim == 2:
            hidden = torch.einsum(
                "bf,mhf->mbh",
                features,
                self.linear_weight,
            )
        else:
            hidden = torch.einsum(
                "mbf,mhf->mbh",
                features,
                self.linear_weight,
            )
        hidden = F.gelu(hidden + self.linear_bias[:, None, :])
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = hidden.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (hidden - mean) * torch.rsqrt(variance + 1e-5)
        return (
            normalized * self.norm_weight[:, None, :]
            + self.norm_bias[:, None, :]
        )

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        """Return model-by-row residuals."""
        return (
            torch.einsum(
                "mbh,mh->mb",
                self.hidden(features),
                self.residual_weight,
            )
            + self.residual_bias[:, None]
        )

    def primary_coefficient(self) -> torch.Tensor:
        """Return bounded primary coefficients."""
        lower, upper = self.coefficient_bounds
        return torch.clamp(
            self.primary_coefficient_raw,
            min=lower,
            max=upper,
        )

    def binary_logits(
        self,
        features: torch.Tensor,
        base_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute model-specific binary logits."""
        scale = F.softplus(self.binary_scale_raw) + 1e-4
        base = (
            base_signal[None, :]
            if base_signal.ndim == 1
            else base_signal
        )
        return (
            self.binary_bias[:, None]
            + scale[:, None] * base
            + self.residual(features)
        )

    def final_score(
        self,
        primary_score: torch.Tensor,
        residual: torch.Tensor,
        correction_scale: float,
        direction: str,
    ) -> torch.Tensor:
        """Combine final scores for every model."""
        sign = 1.0 if direction == "higher_is_better" else -1.0
        primary = (
            primary_score[None, :]
            if primary_score.ndim == 1
            else primary_score
        )
        return (
            self.primary_coefficient()[:, None]
            * primary
            + sign * correction_scale * residual
        )

    @torch.no_grad()
    def project(self) -> None:
        """Project or freeze coefficients."""
        lower, upper = self.coefficient_bounds
        self.primary_coefficient_raw.clamp_(lower, upper)
        self.primary_coefficient_raw.mul_(self.learnable_mask)
        self.primary_coefficient_raw.add_(1.0 - self.learnable_mask)


class BatchedLossWeights(nn.Module):
    """Learn auxiliary weights per head."""

    def __init__(
        self,
        n_models: int,
        initial: dict[str, float],
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        super().__init__()
        self.names = ("pointwise", "positive_relation")
        if set(initial) != set(self.names):
            raise ValueError(
                "Adaptive head weights must define pointwise and "
                "positive_relation"
            )
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        values = [-math.log(float(initial[name])) for name in self.names]
        self.log_variance = nn.Parameter(
            torch.tensor(values, dtype=torch.float32).repeat(n_models, 1)
        )
        self.project()

    def weights(self) -> torch.Tensor:
        """Return model-by-loss weights."""
        return torch.exp(-self.log_variance)

    def objective(
        self,
        pointwise: torch.Tensor,
        positive_relation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adaptive objectives."""
        losses = torch.stack((pointwise, positive_relation), dim=1)
        return (
            self.weights() * losses + self.log_variance
        ).sum(dim=1)

    @torch.no_grad()
    def project(self) -> None:
        """Project effective weights."""
        self.log_variance.clamp_(
            -math.log(self.upper_bound),
            -math.log(self.lower_bound),
        )


def weighted_pair_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    direction: str,
    temperature: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted pair loss."""
    wrong_order = (
        positive - negative
        if direction == "lower_is_better"
        else negative - positive
    )
    losses = F.softplus(wrong_order / temperature)
    return (
        losses * weight[None, :]
    ).sum(dim=1) / weight.sum()


def _standard_scorer(context, batched: BatchedResidualScorer, index: int):
    """Extract one standard scorer."""
    from training import candidate_scorer

    scorer = candidate_scorer(
        context,
        bool(batched.learnable_mask[index].item()),
    )
    with torch.no_grad():
        scorer.shared[0].weight.copy_(batched.linear_weight[index])
        scorer.shared[0].bias.copy_(batched.linear_bias[index])
        scorer.shared[2].weight.copy_(batched.norm_weight[index])
        scorer.shared[2].bias.copy_(batched.norm_bias[index])
        scorer.residual_head.weight.copy_(
            batched.residual_weight[index][None, :]
        )
        scorer.residual_head.bias.copy_(
            batched.residual_bias[index][None]
        )
        scorer.binary_bias.copy_(batched.binary_bias[index])
        scorer.binary_scale_raw.copy_(batched.binary_scale_raw[index])
        scorer.primary_coefficient_raw.copy_(
            batched.primary_coefficient_raw[index]
        )
    return scorer


def head_schedule(
    context,
    seed: int,
    epochs: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generate and upload all head samples."""
    settings = context.settings
    rng = np.random.default_rng(seed)
    anchor_size = min(
        settings.anchor_batch_size,
        len(context.anchor_features),
    )
    anchor = np.empty((epochs, anchor_size), dtype=np.int64)
    source = {
        name: np.empty(
            (
                epochs,
                min(settings.positive_batch_size, len(prepared.weight)),
            ),
            dtype=np.int64,
        )
        for name, prepared in context.prepared_positive.items()
    }
    for epoch in range(epochs):
        anchor[epoch] = rng.integers(
            0,
            len(context.anchor_features),
            size=anchor_size,
        )
        for name, prepared in context.prepared_positive.items():
            source[name][epoch] = rng.integers(
                0,
                len(prepared.weight),
                size=source[name].shape[1],
            )
    device = context.device
    return (
        torch.as_tensor(anchor, device=device),
        {
            name: torch.as_tensor(values, device=device)
            for name, values in source.items()
        },
    )


def fit_candidate_batch(
    context,
    train_rows: np.ndarray,
    anchor_weights: Iterable[float],
    learnable_coefficient: bool,
    seed: int,
    epochs: int,
) -> list[Any]:
    """Fit one anchor batch."""
    from training import (
        TrainingResult,
        candidate_scorer,
        seed_everything,
        within_entity_pairs,
    )

    anchors = tuple(float(value) for value in anchor_weights)
    if not anchors or any(value < 0 for value in anchors):
        raise ValueError("anchor_weights must be non-empty and non-negative")
    rows = np.asarray(train_rows, dtype=np.int64)
    if (
        rows.ndim != 1
        or len(rows) == 0
        or len(np.unique(rows)) != len(rows)
    ):
        raise ValueError("train_rows must be non-empty and unique")

    settings = context.settings
    seed_everything(seed)
    labels = context.labels.iloc[rows].copy().reset_index(drop=True)
    features = torch.as_tensor(
        context.label_features[rows],
        dtype=torch.float32,
        device=context.device,
    )
    y = torch.as_tensor(
        labels["label"].to_numpy(np.float32),
        dtype=torch.float32,
        device=context.device,
    )
    raw = torch.as_tensor(
        context.label_raw[rows],
        dtype=torch.float32,
        device=context.device,
    )
    base = torch.as_tensor(
        context.label_base[rows],
        dtype=torch.float32,
        device=context.device,
    )
    pair_positive, pair_negative, pair_weight = within_entity_pairs(labels)
    pair_positive_t = torch.as_tensor(
        pair_positive,
        dtype=torch.long,
        device=context.device,
    )
    pair_negative_t = torch.as_tensor(
        pair_negative,
        dtype=torch.long,
        device=context.device,
    )
    pair_weight_t = torch.as_tensor(
        pair_weight,
        dtype=torch.float32,
        device=context.device,
    )

    template = candidate_scorer(context, learnable_coefficient)
    learnable_mask = torch.full(
        (len(anchors),),
        float(learnable_coefficient),
        dtype=torch.float32,
        device=context.device,
    )
    scorer = BatchedResidualScorer(
        template,
        len(anchors),
        learnable_mask,
    ).to(context.device)
    del template
    scorer.primary_coefficient_raw.register_hook(
        lambda gradient: gradient * learnable_mask
    )

    balancer = None
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": list(scorer.parameters()),
            "weight_decay": settings.head_weight_decay,
        }
    ]
    adaptive = settings.adaptive_head
    if adaptive is not None:
        balancer = BatchedLossWeights(
            len(anchors),
            adaptive["initial_weights"],
            adaptive["lower_bound"],
            adaptive["upper_bound"],
        ).to(context.device)
        parameter_groups.append(
            {
                "params": list(balancer.parameters()),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.Adam(
        parameter_groups,
        lr=settings.head_learning_rate,
    )
    anchor_weights_t = torch.as_tensor(
        anchors,
        dtype=torch.float32,
        device=context.device,
    )
    anchor_features_all = torch.as_tensor(
        context.anchor_features,
        dtype=torch.float32,
        device=context.device,
    )
    prepared_gpu = [
        (
            torch.as_tensor(
                prepared.positive_features,
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                prepared.negative_features,
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                prepared.positive_raw,
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                prepared.negative_raw,
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                prepared.weight,
                dtype=torch.float32,
                device=context.device,
            ),
        )
        for prepared in context.prepared_positive.values()
    ]
    anchor_schedule, source_schedule = head_schedule(
        context,
        seed,
        epochs,
    )
    history_gpu = torch.empty(
        (epochs, len(anchors), 8),
        dtype=torch.float32,
        device=context.device,
    )

    for epoch in range(1, epochs + 1):
        scorer.train()
        residual = scorer.residual(features)
        final = scorer.final_score(
            raw,
            residual,
            context.correction_scale,
            context.models.architecture.primary_direction,
        )
        ranking = weighted_pair_loss(
            final[:, pair_positive_t],
            final[:, pair_negative_t],
            context.models.architecture.primary_direction,
            settings.ranking_temperature,
            pair_weight_t,
        )
        binary_scale = F.softplus(scorer.binary_scale_raw) + 1e-4
        pointwise = F.binary_cross_entropy_with_logits(
            (
                scorer.binary_bias[:, None]
                + binary_scale[:, None] * base[None, :]
                + residual
            ),
            y[None, :].expand(len(anchors), -1),
            reduction="none",
        ).mean(dim=1)

        anchor_loss = scorer.residual(
            anchor_features_all[anchor_schedule[epoch - 1]]
        ).square().mean(dim=1)

        source_losses = []
        for (name, prepared), gpu in zip(
            context.prepared_positive.items(),
            prepared_gpu,
        ):
            index_t = source_schedule[name][epoch - 1]
            positive_x = gpu[0][index_t]
            negative_x = gpu[1][index_t]
            positive_raw = gpu[2][index_t]
            negative_raw = gpu[3][index_t]
            positive_score = scorer.final_score(
                positive_raw,
                scorer.residual(positive_x),
                context.correction_scale,
                context.models.architecture.primary_direction,
            )
            negative_score = scorer.final_score(
                negative_raw,
                scorer.residual(negative_x),
                context.correction_scale,
                context.models.architecture.primary_direction,
            )
            source_losses.append(
                weighted_pair_loss(
                    positive_score,
                    negative_score,
                    context.models.architecture.primary_direction,
                    settings.positive_temperature,
                    gpu[4][index_t],
                )
            )
        positive_relation = torch.stack(source_losses).mean(dim=0)
        auxiliary = (
            balancer.objective(pointwise, positive_relation)
            if balancer is not None
            else (
                settings.pointwise_weight * pointwise
                + settings.positive_weight * positive_relation
            )
        )
        total = (
            settings.ranking_weight * ranking
            + anchor_weights_t * anchor_loss
            + auxiliary
        )
        optimizer.zero_grad(set_to_none=True)
        total.sum().backward()
        optimizer.step()
        if balancer is not None:
            balancer.project()
        scorer.project()

        effective = (
            balancer.weights().detach()
            if balancer is not None
            else torch.stack(
                (
                    torch.full_like(total, settings.pointwise_weight),
                    torch.full_like(total, settings.positive_weight),
                ),
                dim=1,
            )
        )
        history_gpu[epoch - 1] = torch.stack(
            (
                total.detach(),
                ranking.detach(),
                pointwise.detach(),
                anchor_loss.detach(),
                positive_relation.detach(),
                effective[:, 0],
                effective[:, 1],
                scorer.primary_coefficient().detach(),
            ),
            dim=1,
        )

    final_effective = (
        balancer.weights().detach()
        if balancer is not None
        else torch.stack(
            (
                torch.full_like(total, settings.pointwise_weight),
                torch.full_like(total, settings.positive_weight),
            ),
            dim=1,
        )
    )
    history_values = history_gpu.cpu().numpy()
    final_effective_values = final_effective.cpu().numpy()
    results = []
    for index, anchor in enumerate(anchors):
        history = pd.DataFrame(
            history_values[:, index, :],
            columns=(
                "total_loss",
                "binary_ranking_loss",
                "binary_pointwise_loss",
                "residual_anchor_loss",
                "positive_relation_loss",
                "pointwise_effective_weight",
                "positive_relation_effective_weight",
                "primary_coefficient",
            ),
        )
        history.insert(0, "epoch", np.arange(1, epochs + 1))
        history["coefficient_learnable"] = learnable_coefficient
        results.append(
            TrainingResult(
                _standard_scorer(context, scorer, index),
                rows,
                learnable_coefficient,
                anchor,
                {
                    "pointwise": float(final_effective_values[index, 0]),
                    "positive_relation": float(
                        final_effective_values[index, 1]
                    ),
                },
                history,
            )
        )
    del (
        optimizer,
        scorer,
        balancer,
        anchor_features_all,
        prepared_gpu,
        history_gpu,
    )
    return results
