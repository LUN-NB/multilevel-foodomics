"""Run one continuous trial entirely along the GPU model dimension.

Backbones remain on GPU while standardized features, heads,
and validation predictions are completed for the whole trial batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from adaptive_fast_backbones import (
    BackboneDeviceBatchResult,
    BackboneDeviceData,
    BackboneJob,
    PaddedSupervision,
    fit_fixed_weight_batch,
    load_scorer_states,
)
from adaptive_fast_heads import (
    BatchedLossWeights,
    BatchedResidualScorer,
)


SCRIPT_VERSION = "1.0.1"
SCRIPT_RELEASE = "no_grad_feature_refresh_4gb_20260729"


@dataclass
class TrialFeatureBatch:
    """Store model-specific standardized features."""

    mean: torch.Tensor
    scale: torch.Tensor
    labels: torch.Tensor
    anchors: torch.Tensor
    positive: dict[str, tuple[torch.Tensor, ...]]


@dataclass
class TrialBatchResult:
    """Store one completed trial."""

    backbone_histories: list[pd.DataFrame]
    head_histories: list[pd.DataFrame]
    predictions: list[pd.DataFrame]
    primary_coefficients: np.ndarray
    effective_weights: np.ndarray
    backbone_seconds: float
    feature_refresh_seconds: float
    head_seconds: float
    evaluation_seconds: float
    transfer_seconds: float
    peak_cuda_memory_mb: float


def synchronize(device: torch.device) -> None:
    """Synchronize a timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def raw_pair_features(
    context,
    backbone,
    data: BackboneDeviceData,
    row: torch.Tensor,
    column: torch.Tensor,
) -> torch.Tensor:
    """Build raw batched features."""
    architecture = context.models.architecture
    primary_name = architecture.primary_feature
    auxiliary_name = architecture.auxiliary_feature
    interaction = backbone.interaction(row, column)
    primary = data.matrices[primary_name][row, column]
    auxiliary = data.matrices[auxiliary_name][row, column]
    if primary.ndim == 1:
        primary = primary[None, :].expand(backbone.n_models, -1)
        auxiliary = auxiliary[None, :].expand(backbone.n_models, -1)
    columns = [
        auxiliary - primary,
        (primary - 0.5) * (auxiliary - 0.5),
        backbone.decode_interaction(interaction, primary_name),
        backbone.decode_interaction(interaction, auxiliary_name),
        interaction.sum(dim=2),
    ]
    for name in architecture.sparse_features:
        value = data.matrices[name][row, column]
        observed = data.observed[name][row, column].float()
        if value.ndim == 1:
            value = value[None, :].expand(backbone.n_models, -1)
            observed = observed[None, :].expand(backbone.n_models, -1)
        columns.extend((torch.nan_to_num(value, nan=0.0), observed))
    return torch.cat((torch.stack(columns, dim=2), interaction), dim=2)


@torch.no_grad()
def build_features(
    context,
    backbone,
    data: BackboneDeviceData,
    row: np.ndarray,
    column: np.ndarray,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    chunk_size: int = 131072,
) -> torch.Tensor:
    """Build batched features in chunks."""
    parts = []
    for start in range(0, len(row), chunk_size):
        stop = min(start + chunk_size, len(row))
        row_t = torch.tensor(
            row[start:stop],
            dtype=torch.long,
            device=context.device,
        )
        column_t = torch.tensor(
            column[start:stop],
            dtype=torch.long,
            device=context.device,
        )
        values = raw_pair_features(context, backbone, data, row_t, column_t)
        if mean is not None and scale is not None:
            values = (
                (values.double() - mean[:, None, :])
                / scale[:, None, :]
            ).float()
        parts.append(values)
    return torch.cat(parts, dim=1)


@torch.no_grad()
def refresh_trial_features(
    context,
    backbone,
    data: BackboneDeviceData,
) -> TrialFeatureBatch:
    """Refresh all head features without graphs."""
    backbone.eval()
    coordinates = getattr(context, "_adaptive_feature_coordinates")
    sample = build_features(
        context,
        backbone,
        data,
        coordinates[0],
        coordinates[1],
    ).double()
    mean = sample.mean(dim=1)
    scale = sample.std(dim=1, correction=0)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    del sample

    labels = context.labels
    label_features = build_features(
        context,
        backbone,
        data,
        labels["entity_a_index"].to_numpy(np.int64),
        labels["entity_b_index"].to_numpy(np.int64),
        mean,
        scale,
    )
    anchor_features = build_features(
        context,
        backbone,
        data,
        context.anchor_row,
        context.anchor_column,
        mean,
        scale,
    )
    positive = {}
    for name, comparison in context.positive_comparisons.items():
        positive[name] = (
            build_features(
                context,
                backbone,
                data,
                comparison.positive_row,
                comparison.positive_column,
                mean,
                scale,
            ),
            build_features(
                context,
                backbone,
                data,
                comparison.positive_row,
                comparison.negative_column,
                mean,
                scale,
            ),
            torch.as_tensor(
                context.raw_primary[
                    comparison.positive_row,
                    comparison.positive_column,
                ],
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                context.raw_primary[
                    comparison.positive_row,
                    comparison.negative_column,
                ],
                dtype=torch.float32,
                device=context.device,
            ),
            torch.as_tensor(
                comparison.weight,
                dtype=torch.float32,
                device=context.device,
            ),
        )
    return TrialFeatureBatch(
        mean,
        scale,
        label_features,
        anchor_features,
        positive,
    )


def mixed_head_schedule(
    context,
    seeds: list[int],
    epochs: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Upload every head schedule once."""
    settings = context.settings
    anchor_size = min(settings.anchor_batch_size, len(context.anchor_row))
    anchor = np.empty(
        (len(seeds), epochs, anchor_size),
        dtype=np.int64,
    )
    source = {
        name: np.empty(
            (
                len(seeds),
                epochs,
                min(settings.positive_batch_size, len(comparison.weight)),
            ),
            dtype=np.int64,
        )
        for name, comparison in context.positive_comparisons.items()
    }
    for model, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            anchor[model, epoch] = rng.integers(
                0,
                len(context.anchor_row),
                size=anchor_size,
            )
            for name, comparison in context.positive_comparisons.items():
                source[name][model, epoch] = rng.integers(
                    0,
                    len(comparison.weight),
                    size=source[name].shape[2],
                )
    return (
        torch.as_tensor(anchor, device=context.device),
        {
            name: torch.as_tensor(values, device=context.device)
            for name, values in source.items()
        },
    )


def gather_model_rows(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather feature rows for every model."""
    expanded = index[..., None].expand(-1, -1, values.shape[2])
    return torch.gather(values, 1, expanded)


def model_pair_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    direction: str,
    temperature: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute independently normalized pair loss."""
    wrong_order = (
        positive - negative
        if direction == "lower_is_better"
        else negative - positive
    )
    return (
        F.softplus(wrong_order / temperature) * weight
    ).sum(dim=1) / weight.sum(dim=1)


def initial_head_states(context, seeds: list[int]) -> list[dict[str, torch.Tensor]]:
    """Build formal per-seed head states."""
    from training import candidate_scorer, seed_everything

    states = []
    for seed in seeds:
        seed_everything(seed)
        scorer = candidate_scorer(context, learnable_coefficient=True)
        states.append(
            {
                name: value.detach().clone()
                for name, value in scorer.state_dict().items()
            }
        )
        del scorer
    return states


def fit_trial_heads(
    context,
    features: TrialFeatureBatch,
    supervision: PaddedSupervision,
    anchors: list[float],
    seeds: list[int],
    epochs: int,
) -> tuple[
    BatchedResidualScorer,
    torch.Tensor,
    torch.Tensor,
]:
    """Fit all trial heads together."""
    from training import candidate_scorer

    states = initial_head_states(context, seeds)
    template = candidate_scorer(context, learnable_coefficient=True)
    scorer = BatchedResidualScorer(
        template,
        len(seeds),
        torch.ones(len(seeds), device=context.device),
    ).to(context.device)
    load_scorer_states(scorer, states)
    del template, states

    settings = context.settings
    balancer = None
    groups: list[dict[str, Any]] = [
        {
            "params": list(scorer.parameters()),
            "weight_decay": settings.head_weight_decay,
        }
    ]
    adaptive = settings.adaptive_head
    if adaptive is not None:
        balancer = BatchedLossWeights(
            len(seeds),
            adaptive["initial_weights"],
            adaptive["lower_bound"],
            adaptive["upper_bound"],
        ).to(context.device)
        groups.append(
            {"params": list(balancer.parameters()), "weight_decay": 0.0}
        )
    optimizer = torch.optim.Adam(groups, lr=settings.head_learning_rate)
    anchor_weight = torch.as_tensor(
        anchors,
        dtype=torch.float32,
        device=context.device,
    )
    anchor_schedule, source_schedule = mixed_head_schedule(
        context,
        seeds,
        epochs,
    )
    train_features = gather_model_rows(
        features.labels,
        supervision.relation_row,
    )
    history = torch.empty(
        (epochs, len(seeds), 8),
        dtype=torch.float32,
        device=context.device,
    )
    direction = context.models.architecture.primary_direction

    for epoch in range(epochs):
        scorer.train()
        residual = scorer.residual(train_features)
        final = scorer.final_score(
            supervision.raw,
            residual,
            context.correction_scale,
            direction,
        )
        positive = torch.gather(final, 1, supervision.pair_positive)
        negative = torch.gather(final, 1, supervision.pair_negative)
        ranking = model_pair_loss(
            positive,
            negative,
            direction,
            settings.ranking_temperature,
            supervision.pair_weight,
        )
        logits = scorer.binary_logits(train_features, supervision.base)
        binary = F.binary_cross_entropy_with_logits(
            logits,
            supervision.label,
            reduction="none",
        )
        pointwise = (
            binary * supervision.label_mask
        ).sum(dim=1) / supervision.label_mask.sum(dim=1)
        anchor_x = gather_model_rows(
            features.anchors,
            anchor_schedule[:, epoch, :],
        )
        anchor_loss = scorer.residual(anchor_x).square().mean(dim=1)

        source_losses = []
        for name, values in features.positive.items():
            index = source_schedule[name][:, epoch, :]
            positive_x = gather_model_rows(values[0], index)
            negative_x = gather_model_rows(values[1], index)
            raw_index = index
            positive_raw = torch.gather(
                values[2][None, :].expand(len(seeds), -1),
                1,
                raw_index,
            )
            negative_raw = torch.gather(
                values[3][None, :].expand(len(seeds), -1),
                1,
                raw_index,
            )
            weight = torch.gather(
                values[4][None, :].expand(len(seeds), -1),
                1,
                raw_index,
            )
            positive_score = scorer.final_score(
                positive_raw,
                scorer.residual(positive_x),
                context.correction_scale,
                direction,
            )
            negative_score = scorer.final_score(
                negative_raw,
                scorer.residual(negative_x),
                context.correction_scale,
                direction,
            )
            source_losses.append(
                model_pair_loss(
                    positive_score,
                    negative_score,
                    direction,
                    settings.positive_temperature,
                    weight,
                )
            )
        positive_relation = torch.stack(source_losses).mean(dim=0)
        effective = (
            balancer.weights()
            if balancer is not None
            else torch.stack(
                (
                    torch.full_like(ranking, settings.pointwise_weight),
                    torch.full_like(ranking, settings.positive_weight),
                ),
                dim=1,
            )
        )
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
            + anchor_weight * anchor_loss
            + auxiliary
        )
        optimizer.zero_grad(set_to_none=True)
        total.sum().backward()
        optimizer.step()
        if balancer is not None:
            balancer.project()
        scorer.project()
        history[epoch] = torch.stack(
            (
                total.detach(),
                ranking.detach(),
                pointwise.detach(),
                anchor_loss.detach(),
                positive_relation.detach(),
                effective[:, 0].detach(),
                effective[:, 1].detach(),
                scorer.primary_coefficient().detach(),
            ),
            dim=1,
        )
    final_effective = (
        balancer.weights().detach()
        if balancer is not None
        else effective.detach()
    )
    del optimizer, balancer, train_features
    return scorer, history, final_effective


@torch.no_grad()
def predict_trial_heads(
    context,
    scorer: BatchedResidualScorer,
    features: TrialFeatureBatch,
    validation_rows: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate padded validation predictions."""
    width = max(len(rows) for rows in validation_rows)
    index = np.zeros((len(validation_rows), width), dtype=np.int64)
    mask = np.zeros((len(validation_rows), width), dtype=bool)
    for model, rows in enumerate(validation_rows):
        index[model, : len(rows)] = rows
        mask[model, : len(rows)] = True
    index_t = torch.as_tensor(index, device=context.device)
    validation = gather_model_rows(features.labels, index_t)
    base = torch.as_tensor(
        context.label_base,
        dtype=torch.float32,
        device=context.device,
    )[index_t]
    raw = torch.as_tensor(
        context.label_raw,
        dtype=torch.float32,
        device=context.device,
    )[index_t]
    scorer.eval()
    residual = scorer.residual(validation)
    probability = torch.sigmoid(scorer.binary_logits(validation, base))
    final = scorer.final_score(
        raw,
        residual,
        context.correction_scale,
        context.models.architecture.primary_direction,
    )
    return (
        torch.stack((probability, residual, final), dim=2),
        torch.as_tensor(mask, device=context.device),
    )


def history_frames(
    values: np.ndarray,
    names: tuple[str, ...],
) -> list[pd.DataFrame]:
    """Convert model histories to frames."""
    frames = []
    for model in range(values.shape[1]):
        frame = pd.DataFrame(values[:, model, :], columns=names)
        frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
        frames.append(frame)
    return frames


def run_trial_batch(
    context,
    data: BackboneDeviceData,
    initial_backbones: list[dict[str, torch.Tensor]],
    initial_scorers: list[dict[str, torch.Tensor]],
    standardizers: list[Any],
    jobs: list[BackboneJob],
    backbone_seeds: list[int],
    head_seeds: list[int],
    anchors: list[float],
    validation_rows: list[np.ndarray],
    backbone_epochs: int,
    steps_per_epoch: int,
    head_epochs: int,
    supervision: PaddedSupervision,
    schedules: dict[int, tuple[torch.Tensor, ...]],
) -> TrialBatchResult:
    """Execute one full GPU trial."""
    device = context.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    backbone_result = fit_fixed_weight_batch(
        context,
        data,
        initial_backbones,
        initial_scorers,
        standardizers,
        jobs,
        backbone_epochs,
        steps_per_epoch,
        backbone_seeds,
        supervision,
        schedules,
        retain_device=True,
    )
    assert isinstance(backbone_result, BackboneDeviceBatchResult)
    backbone_seconds = backbone_result.elapsed_seconds

    synchronize(device)
    started = time.perf_counter()
    features = refresh_trial_features(
        context,
        backbone_result.backbone,
        data,
    )
    synchronize(device)
    feature_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scorer, head_history, effective = fit_trial_heads(
        context,
        features,
        supervision,
        anchors,
        head_seeds,
        head_epochs,
    )
    synchronize(device)
    head_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction, prediction_mask = predict_trial_heads(
        context,
        scorer,
        features,
        validation_rows,
    )
    synchronize(device)
    evaluation_seconds = time.perf_counter() - started

    started = time.perf_counter()
    backbone_history_cpu = backbone_result.history.cpu().numpy()
    head_history_cpu = head_history.cpu().numpy()
    prediction_cpu = prediction.cpu().numpy()
    mask_cpu = prediction_mask.cpu().numpy()
    coefficient_cpu = scorer.primary_coefficient().detach().cpu().numpy()
    effective_cpu = effective.cpu().numpy()
    synchronize(device)
    transfer_seconds = time.perf_counter() - started

    backbone_names = (
        "train_total_loss",
        "train_main_loss",
        "train_ranking_loss",
        "train_pointwise_loss",
        "train_primary_loss",
        "train_graphban_loss",
        "train_boltz_loss",
    )
    backbone_histories = history_frames(
        backbone_history_cpu,
        backbone_names,
    )
    for frame, job in zip(backbone_histories, jobs):
        frame["graphban_weighted_contribution"] = (
            job.graphban_weight * frame["train_graphban_loss"]
        )
        frame["boltz_weighted_contribution"] = (
            job.boltz_weight * frame["train_boltz_loss"]
        )

    head_names = (
        "total_loss",
        "binary_ranking_loss",
        "binary_pointwise_loss",
        "residual_anchor_loss",
        "positive_relation_loss",
        "pointwise_effective_weight",
        "positive_relation_effective_weight",
        "primary_coefficient",
    )
    head_histories = history_frames(head_history_cpu, head_names)
    for frame in head_histories:
        frame["coefficient_learnable"] = True

    predictions = []
    for model, rows in enumerate(validation_rows):
        selected = context.labels.iloc[rows]
        values = prediction_cpu[model, mask_cpu[model]]
        predictions.append(
            pd.DataFrame(
                {
                    "relation_row": selected["relation_row"].to_numpy(),
                    "entity_a": selected["entity_a"].to_numpy(),
                    "entity_b": selected["entity_b"].to_numpy(),
                    "label": selected["label"].to_numpy(),
                    "probability": values[:, 0],
                    "residual": values[:, 1],
                    "final_score": values[:, 2],
                }
            )
        )

    peak = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    del (
        scorer,
        features,
        backbone_result,
        head_history,
        prediction,
        prediction_mask,
        effective,
    )
    return TrialBatchResult(
        backbone_histories,
        head_histories,
        predictions,
        coefficient_cpu,
        effective_cpu,
        backbone_seconds,
        feature_seconds,
        head_seconds,
        evaluation_seconds,
        transfer_seconds,
        peak,
    )
