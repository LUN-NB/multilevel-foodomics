"""Train independent CPI-MEC backbones along one model dimension.

Coarse search batches different G/B candidates within one fold and seed.
Continuous TPE batches every fold and seed inside one trial. Every model keeps
its own parameters, Adam state, CPI supervision, and loss weights.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_fast_heads import BatchedResidualScorer


SCRIPT_VERSION = "1.2.1"
SCRIPT_RELEASE = "unique_seed_schedule_no_model_duplication_20260729"


@dataclass(frozen=True)
class BackboneDeviceData:
    """Store resident relation data."""

    matrices: dict[str, torch.Tensor]
    observed: dict[str, torch.Tensor]
    sparse_flat_observed: np.ndarray | None
    n_entity_a: int
    n_entity_b: int


@dataclass(frozen=True)
class BackboneJob:
    """Describe one independent backbone fit."""

    train_rows: np.ndarray
    graphban_weight: float
    boltz_weight: float


@dataclass
class BackboneBatchResult:
    """Store one member of a model batch."""

    state: dict[str, torch.Tensor]
    history: pd.DataFrame
    summary: dict[str, float | int | str]


@dataclass
class BackboneDeviceBatchResult:
    """Retain a GPU batch through a trial."""

    backbone: "BatchedRelationalBackbone"
    history: torch.Tensor
    elapsed_seconds: float
    peak_cuda_memory_mb: float


@dataclass(frozen=True)
class PaddedSupervision:
    """Store model-padded CPI supervision."""

    row: torch.Tensor
    column: torch.Tensor
    relation_row: torch.Tensor
    label: torch.Tensor
    raw: torch.Tensor
    base: torch.Tensor
    label_mask: torch.Tensor
    pair_positive: torch.Tensor
    pair_negative: torch.Tensor
    pair_weight: torch.Tensor


class BatchedRelationalBackbone(nn.Module):
    """Represent independent backbones."""

    def __init__(
        self,
        template,
        initial_states: list[dict[str, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.embedding_dim = template.embedding_dim
        self.relation_names = template.relation_names
        self.relation_index = template.relation_index
        self.decoder_outputs = template.decoder_outputs
        self.entity_a_weight = nn.Parameter(
            torch.stack(
                [state["entity_a.weight"] for state in initial_states]
            )
        )
        self.entity_b_weight = nn.Parameter(
            torch.stack(
                [state["entity_b.weight"] for state in initial_states]
            )
        )
        self.relation_scale = nn.Parameter(
            torch.stack(
                [state["relation_scale"] for state in initial_states]
            )
        )
        self.relation_bias = nn.Parameter(
            torch.stack(
                [state["relation_bias"] for state in initial_states]
            )
        )

    @property
    def n_models(self) -> int:
        """Return the number of parallel models."""
        return int(self.entity_a_weight.shape[0])

    def _gather(
        self,
        weight: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather shared or per-model rows."""
        if index.ndim == 1:
            return weight[:, index, :]
        expanded = index[..., None].expand(-1, -1, self.embedding_dim)
        return torch.gather(weight, 1, expanded)

    def interaction(
        self,
        entity_a: torch.Tensor,
        entity_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return model-pair interactions."""
        return (
            self._gather(self.entity_a_weight, entity_a)
            * self._gather(self.entity_b_weight, entity_b)
            / math.sqrt(self.embedding_dim)
        )

    def decode_interaction(
        self,
        interaction: torch.Tensor,
        relation: str,
    ) -> torch.Tensor:
        """Decode from cached interactions."""
        relation_index = self.relation_index[relation]
        raw = (
            torch.einsum(
                "mbd,md->mb",
                interaction,
                self.relation_scale[:, relation_index, :],
            )
            + self.relation_bias[:, relation_index, None]
        )
        if self.decoder_outputs[relation] == "sigmoid":
            return torch.sigmoid(raw)
        return raw

    def state_at(self, index: int) -> dict[str, torch.Tensor]:
        """Extract one standard backbone state."""
        return {
            "entity_a.weight": self.entity_a_weight[index].detach().cpu().clone(),
            "entity_b.weight": self.entity_b_weight[index].detach().cpu().clone(),
            "relation_scale": self.relation_scale[index].detach().cpu().clone(),
            "relation_bias": self.relation_bias[index].detach().cpu().clone(),
        }


def prepare_backbone_device_data(context) -> BackboneDeviceData:
    """Move relation matrices once."""
    architecture = context.models.architecture
    matrices = {
        name: torch.tensor(
            context.matrix_values[name],
            dtype=torch.float32,
            device=context.device,
        )
        for name in architecture.backbone_relations
    }
    observed = {
        name: torch.tensor(
            context.matrix_observed[name],
            dtype=torch.bool,
            device=context.device,
        )
        for name in architecture.sparse_features
    }
    sparse = architecture.sparse_features[0]
    finite = np.isfinite(context.matrix_values[sparse])
    flat = None if finite.all() else np.flatnonzero(finite.ravel())
    n_entity_a, n_entity_b = context.matrix_values[
        architecture.primary_feature
    ].shape
    return BackboneDeviceData(
        matrices,
        observed,
        flat,
        n_entity_a,
        n_entity_b,
    )


def _pad(
    parts: list[np.ndarray],
    dtype: Any,
    fill: float | int = 0,
) -> np.ndarray:
    """Pad per-model one-dimensional arrays."""
    width = max(len(part) for part in parts)
    output = np.full((len(parts), width), fill, dtype=dtype)
    for index, part in enumerate(parts):
        output[index, : len(part)] = part
    return output


def prepare_supervision(
    context,
    jobs: list[BackboneJob],
) -> PaddedSupervision:
    """Prepare labels and CID pairs."""
    from training import within_entity_pairs

    row_parts = []
    column_parts = []
    label_parts = []
    raw_parts = []
    base_parts = []
    mask_parts = []
    positive_parts = []
    negative_parts = []
    weight_parts = []
    for job in jobs:
        rows = np.asarray(job.train_rows, dtype=np.int64)
        labels = context.labels.iloc[rows].copy().reset_index(drop=True)
        row_parts.append(labels["entity_a_index"].to_numpy(np.int64))
        column_parts.append(labels["entity_b_index"].to_numpy(np.int64))
        label_parts.append(labels["label"].to_numpy(np.float32))
        raw_parts.append(context.label_raw[rows].astype(np.float32))
        base_parts.append(context.label_base[rows].astype(np.float32))
        mask_parts.append(np.ones(len(rows), dtype=np.float32))
        positive, negative, weight = within_entity_pairs(labels)
        positive_parts.append(positive.astype(np.int64))
        negative_parts.append(negative.astype(np.int64))
        weight_parts.append(weight.astype(np.float32))

    device = context.device
    return PaddedSupervision(
        torch.as_tensor(_pad(row_parts, np.int64), device=device),
        torch.as_tensor(_pad(column_parts, np.int64), device=device),
        torch.as_tensor(
            _pad(
                [
                    np.asarray(job.train_rows, dtype=np.int64)
                    for job in jobs
                ],
                np.int64,
            ),
            device=device,
        ),
        torch.as_tensor(
            _pad(label_parts, np.float32),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            _pad(raw_parts, np.float32),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            _pad(base_parts, np.float32),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            _pad(mask_parts, np.float32),
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(_pad(positive_parts, np.int64), device=device),
        torch.as_tensor(_pad(negative_parts, np.int64), device=device),
        torch.as_tensor(
            _pad(weight_parts, np.float32),
            dtype=torch.float32,
            device=device,
        ),
    )


def standardized_pair_features(
    context,
    backbone: BatchedRelationalBackbone,
    data: BackboneDeviceData,
    row: torch.Tensor,
    column: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Build model-specific features once."""
    architecture = context.models.architecture
    primary_name = architecture.primary_feature
    auxiliary_name = architecture.auxiliary_feature
    interaction = backbone.interaction(row, column)
    primary = data.matrices[primary_name][row, column]
    auxiliary = data.matrices[auxiliary_name][row, column]
    columns = [
        auxiliary - primary,
        (primary - 0.5) * (auxiliary - 0.5),
        backbone.decode_interaction(interaction, primary_name),
        backbone.decode_interaction(interaction, auxiliary_name),
        interaction.sum(dim=2),
    ]
    for name in architecture.sparse_features:
        value = data.matrices[name][row, column]
        columns.extend(
            (
                torch.nan_to_num(value, nan=0.0),
                data.observed[name][row, column].float(),
            )
        )
    features = torch.cat(
        (torch.stack(columns, dim=2), interaction),
        dim=2,
    )
    if mean.ndim == 1:
        mean = mean[None, :]
        scale = scale[None, :]
    return (features - mean[:, None, :]) / scale[:, None, :]


@torch.no_grad()
def load_scorer_states(
    scorer: BatchedResidualScorer,
    states: list[dict[str, torch.Tensor]],
) -> None:
    """Load per-model head initial states."""
    mappings = {
        "linear_weight": "shared.0.weight",
        "linear_bias": "shared.0.bias",
        "norm_weight": "shared.2.weight",
        "norm_bias": "shared.2.bias",
        "residual_weight": "residual_head.weight",
        "residual_bias": "residual_head.bias",
        "binary_bias": "binary_bias",
        "binary_scale_raw": "binary_scale_raw",
        "primary_coefficient_raw": "primary_coefficient_raw",
    }
    for target_name, source_name in mappings.items():
        values = [state[source_name] for state in states]
        if target_name == "residual_weight":
            values = [value.squeeze(0) for value in values]
        if target_name == "residual_bias":
            values = [value.squeeze(0) for value in values]
        getattr(scorer, target_name).copy_(torch.stack(values))


def cpi_losses(
    context,
    backbone: BatchedRelationalBackbone,
    scorer: BatchedResidualScorer,
    data: BackboneDeviceData,
    supervision: PaddedSupervision,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-model CPI losses."""
    features = standardized_pair_features(
        context,
        backbone,
        data,
        supervision.row,
        supervision.column,
        mean,
        scale,
    )
    residual = scorer.residual(features)
    final = scorer.final_score(
        supervision.raw,
        residual,
        context.correction_scale,
        context.models.architecture.primary_direction,
    )
    positive = torch.gather(final, 1, supervision.pair_positive)
    negative = torch.gather(final, 1, supervision.pair_negative)
    wrong_order = (
        positive - negative
        if context.models.architecture.primary_direction == "lower_is_better"
        else negative - positive
    )
    pair_loss = F.softplus(
        wrong_order / context.settings.ranking_temperature
    )
    normalized_weight = supervision.pair_weight / (
        supervision.pair_weight.sum(dim=1, keepdim=True)
    )
    ranking = (pair_loss * normalized_weight).sum(dim=1)
    logits = (
        scorer.binary_bias[:, None]
        + (F.softplus(scorer.binary_scale_raw) + 1e-4)[:, None]
        * supervision.base
        + residual
    )
    binary = F.binary_cross_entropy_with_logits(
        logits,
        supervision.label,
        reduction="none",
    )
    pointwise = (
        binary * supervision.label_mask
    ).sum(dim=1) / supervision.label_mask.sum(dim=1)
    main = ranking + context.settings.pointwise_weight * pointwise
    return main, ranking, pointwise


def full_schedule(
    context,
    data: BackboneDeviceData,
    seed: int,
    epochs: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate and upload all epochs once."""
    settings = context.settings
    dense_row = np.empty(
        (epochs, steps, settings.dense_batch_size),
        dtype=np.int32,
    )
    dense_column = np.empty_like(dense_row)
    sparse_size = settings.sparse_batch_size
    if data.sparse_flat_observed is not None:
        sparse_size = min(sparse_size, len(data.sparse_flat_observed))
    sparse_row = np.empty((epochs, steps, sparse_size), dtype=np.int32)
    sparse_column = np.empty_like(sparse_row)
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch + 1)
        for step in range(steps):
            dense_row[epoch, step] = rng.integers(
                0,
                data.n_entity_a,
                size=settings.dense_batch_size,
                dtype=np.int32,
            )
            dense_column[epoch, step] = rng.integers(
                0,
                data.n_entity_b,
                size=settings.dense_batch_size,
                dtype=np.int32,
            )
            if data.sparse_flat_observed is None:
                sparse_row[epoch, step] = rng.integers(
                    0,
                    data.n_entity_a,
                    size=sparse_size,
                    dtype=np.int32,
                )
                sparse_column[epoch, step] = rng.integers(
                    0,
                    data.n_entity_b,
                    size=sparse_size,
                    dtype=np.int32,
                )
            else:
                flat = data.sparse_flat_observed[rng.integers(
                    0,
                    len(data.sparse_flat_observed),
                    size=sparse_size,
                )]
                (
                    sparse_row[epoch, step],
                    sparse_column[epoch, step],
                ) = np.divmod(flat, data.n_entity_b)
    return tuple(
        torch.as_tensor(values, device=context.device)
        for values in (
            dense_row,
            dense_column,
            sparse_row,
            sparse_column,
        )
    )


def mixed_full_schedule(
    context,
    data: BackboneDeviceData,
    seeds: list[int],
    epochs: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack full schedules for every model."""
    schedules = [
        full_schedule(context, data, seed, epochs, steps)
        for seed in seeds
    ]
    return tuple(
        torch.stack([schedule[index] for schedule in schedules])
        for index in range(4)
    )


def schedule_step(
    schedules: dict[int, tuple[torch.Tensor, ...]],
    seeds: list[int],
    epoch: int,
    step: int,
) -> tuple[torch.Tensor, ...]:
    """Map unique-seed schedule slices to models."""
    return tuple(
        torch.stack(
            [
                schedules[seed][coordinate][epoch, step]
                for seed in seeds
            ]
        ).long()
        for coordinate in range(4)
    )


def relation_losses(
    context,
    backbone: BatchedRelationalBackbone,
    data: BackboneDeviceData,
    dense_row: torch.Tensor,
    dense_column: torch.Tensor,
    sparse_row: torch.Tensor,
    sparse_column: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute U/G/B per model."""
    architecture = context.models.architecture
    primary_name = architecture.primary_feature
    auxiliary_name = architecture.auxiliary_feature
    sparse_name = architecture.sparse_features[0]

    dense_interaction = backbone.interaction(dense_row, dense_column)
    primary_prediction = backbone.decode_interaction(
        dense_interaction,
        primary_name,
    )
    graphban_prediction = backbone.decode_interaction(
        dense_interaction,
        auxiliary_name,
    )
    primary_target = data.matrices[primary_name][dense_row, dense_column]
    graphban_target = data.matrices[auxiliary_name][
        dense_row,
        dense_column,
    ]
    if dense_row.ndim == 1:
        primary_target = primary_target[None, :]
        graphban_target = graphban_target[None, :]
    primary = (primary_prediction - primary_target).square().mean(dim=1)
    graphban = (graphban_prediction - graphban_target).square().mean(dim=1)

    sparse_interaction = backbone.interaction(sparse_row, sparse_column)
    sparse_prediction = backbone.decode_interaction(
        sparse_interaction,
        sparse_name,
    )
    sparse_target = data.matrices[sparse_name][sparse_row, sparse_column]
    if sparse_row.ndim == 1:
        sparse_target = sparse_target[None, :]
    boltz = (sparse_prediction - sparse_target).square().mean(dim=1)
    return primary, graphban, boltz


def fit_fixed_weight_batch(
    context,
    data: BackboneDeviceData,
    initial_backbone: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    initial_scorer: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    standardizer: Any | list[Any],
    jobs: Iterable[BackboneJob],
    epochs: int,
    steps_per_epoch: int,
    seed: int | list[int],
    supervision: PaddedSupervision | None = None,
    schedules: dict[int, tuple[torch.Tensor, ...]] | None = None,
    retain_device: bool = False,
) -> list[BackboneBatchResult] | BackboneDeviceBatchResult:
    """Train fixed-weight backbones in parallel."""
    from training import candidate_scorer, seed_everything

    jobs = list(jobs)
    if not jobs:
        return []
    initial_backbones = (
        initial_backbone
        if isinstance(initial_backbone, list)
        else [initial_backbone for _ in jobs]
    )
    initial_scorers = (
        initial_scorer
        if isinstance(initial_scorer, list)
        else [initial_scorer for _ in jobs]
    )
    standardizers = (
        standardizer
        if isinstance(standardizer, list)
        else [standardizer for _ in jobs]
    )
    seeds = seed if isinstance(seed, list) else [seed for _ in jobs]
    if not all(
        len(values) == len(jobs)
        for values in (
            initial_backbones,
            initial_scorers,
            standardizers,
            seeds,
        )
    ):
        raise ValueError("Per-model initialization lengths must match jobs")
    seed_everything(seeds[0])
    device = context.device
    backbone = BatchedRelationalBackbone(
        context.models.backbone,
        initial_backbones,
    ).to(device)
    template = candidate_scorer(context, learnable_coefficient=True)
    template.load_state_dict(initial_scorers[0])
    learnable_mask = torch.ones(len(jobs), device=device)
    scorer = BatchedResidualScorer(
        template,
        len(jobs),
        learnable_mask,
    ).to(device)
    load_scorer_states(scorer, initial_scorers)
    del template
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(backbone.parameters()),
                "weight_decay": context.settings.backbone_weight_decay,
            },
            {
                "params": list(scorer.parameters()),
                "weight_decay": context.settings.head_weight_decay,
            },
        ],
        lr=context.settings.backbone_learning_rate,
    )
    supervision = supervision or prepare_supervision(context, jobs)
    mean = torch.as_tensor(
        np.stack([value.mean for value in standardizers]),
        dtype=torch.float32,
        device=device,
    )
    scale = torch.as_tensor(
        np.stack([value.scale for value in standardizers]),
        dtype=torch.float32,
        device=device,
    )
    graphban_weight = torch.as_tensor(
        [job.graphban_weight for job in jobs],
        dtype=torch.float32,
        device=device,
    )
    boltz_weight = torch.as_tensor(
        [job.boltz_weight for job in jobs],
        dtype=torch.float32,
        device=device,
    )
    schedule_by_seed = schedules or {
        value: full_schedule(
            context,
            data,
            value,
            epochs,
            steps_per_epoch,
        )
        for value in set(seeds)
    }
    history_gpu = torch.empty(
        (epochs, len(jobs), 7),
        dtype=torch.float32,
        device=device,
    )
    report_every = max(1, epochs // 6)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        totals = torch.zeros(
            (len(jobs), 7),
            dtype=torch.float32,
            device=device,
        )
        backbone.train()
        scorer.train()
        for step in range(steps_per_epoch):
            main, ranking, pointwise = cpi_losses(
                context,
                backbone,
                scorer,
                data,
                supervision,
                mean,
                scale,
            )
            current_schedule = schedule_step(
                schedule_by_seed,
                seeds,
                epoch - 1,
                step,
            )
            primary, graphban, boltz = relation_losses(
                context,
                backbone,
                data,
                *current_schedule,
            )
            total = (
                main
                + primary
                + graphban_weight * graphban
                + boltz_weight * boltz
            )
            optimizer.zero_grad(set_to_none=True)
            total.sum().backward()
            optimizer.step()
            scorer.project()
            totals += torch.stack(
                (
                    total,
                    main,
                    ranking,
                    pointwise,
                    primary,
                    graphban,
                    boltz,
                ),
                dim=1,
            ).detach()

        history_gpu[epoch - 1] = totals / steps_per_epoch
        if epoch == 1 or epoch == epochs or epoch % report_every == 0:
            print(
                f"PROGRESS fast-backbone epoch={epoch}/{epochs} "
                f"models={len(jobs)}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    if retain_device:
        del optimizer, scorer, supervision
        return BackboneDeviceBatchResult(
            backbone,
            history_gpu,
            elapsed,
            peak_memory,
        )

    values = history_gpu.cpu().numpy()
    results = []
    for index, job in enumerate(jobs):
        history = pd.DataFrame(
            {
                "epoch": np.arange(1, epochs + 1),
                "train_total_loss": values[:, index, 0],
                "train_main_loss": values[:, index, 1],
                "train_ranking_loss": values[:, index, 2],
                "train_pointwise_loss": values[:, index, 3],
                "train_primary_loss": values[:, index, 4],
                "train_graphban_loss": values[:, index, 5],
                "train_boltz_loss": values[:, index, 6],
                "graphban_weighted_contribution": (
                    job.graphban_weight * values[:, index, 5]
                ),
                "boltz_weighted_contribution": (
                    job.boltz_weight * values[:, index, 6]
                ),
            }
        )
        results.append(
            BackboneBatchResult(
                backbone.state_at(index),
                history,
                {
                    "selected_epoch": epochs,
                    **history.iloc[-1].to_dict(),
                    "backbone_execution": "fast_model_batch",
                    "backbone_model_batch_size": len(jobs),
                    "batch_elapsed_seconds": elapsed,
                    "seconds_per_backbone": elapsed / len(jobs),
                    "peak_cuda_memory_mb": peak_memory,
                },
            )
        )
    del optimizer, scorer, backbone, supervision, history_gpu
    return results
