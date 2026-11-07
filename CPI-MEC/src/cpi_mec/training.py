"""Train one declared relational candidate without selecting it.

The module owns optimization, sampling, and prediction for a supplied training
row set. Cross-validation orchestration and candidate selection remain outside
this file so held-out rows cannot influence fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import load_config
from models import (
    ModelBundle,
    ResidualScorer,
    UncertaintyLossBalancer,
    binary_supervision_loss,
    build_models,
    build_pair_features,
    dense_reconstruction_loss,
    pairwise_ranking_loss,
    residual_anchor_loss,
)
from relations import PairRelation, RelationBundle, load_relations
from transforms import FittedStandardizer, fill_row_means, fit_standardizer, median_row_scale, stable_logit


@dataclass(frozen=True)
class TrainingSettings:
    """Store fixed training parameters and relation mappings."""

    binary_input: str
    binary_label: str
    binary_base_transform: str
    positive_input: str
    positive_group: str
    positive_confidence: str | None
    missing_confidence: float
    dense_batch_size: int
    sparse_batch_size: int
    steps_per_epoch: int
    backbone_epochs: int
    backbone_learning_rate: float
    backbone_sparse_weight: float
    backbone_weight_decay: float
    feature_sample_size: int
    anchor_sample_size: int
    anchor_train_fraction: float
    anchor_batch_size: int
    positive_weight: float
    unobserved_pair_ratio: int
    positive_batch_size: int
    positive_temperature: float
    source_seed_offsets: dict[str, int]
    head_epochs: int
    head_learning_rate: float
    ranking_temperature: float
    ranking_weight: float
    pointwise_weight: float
    residual_anchor_weight: float
    head_weight_decay: float
    adaptive_backbone: dict[str, Any] | None
    adaptive_head: dict[str, Any] | None
    seed: int


@dataclass(frozen=True)
class RelationSampler:
    """Store finite coordinates for one backbone relation."""

    name: str
    values: np.ndarray
    flat_observed: np.ndarray | None

    def sample(self, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample finite relation values."""
        n_entity_a, n_entity_b = self.values.shape
        if self.flat_observed is None:
            row = rng.integers(0, n_entity_a, size=size, dtype=np.int64)
            column = rng.integers(0, n_entity_b, size=size, dtype=np.int64)
        else:
            flat = self.flat_observed[
                rng.integers(0, len(self.flat_observed), size=min(size, len(self.flat_observed)))
            ]
            row, column = np.divmod(flat, n_entity_b)
        return row, column, self.values[row, column]


@dataclass(frozen=True)
class PositiveComparisons:
    """Store positive-only pairs and same-entity comparators."""

    name: str
    positive_row: np.ndarray
    positive_column: np.ndarray
    negative_column: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class PreparedPositiveFeatures:
    """Cache frozen features for one positive relation type."""

    positive_features: np.ndarray
    negative_features: np.ndarray
    positive_raw: np.ndarray
    negative_raw: np.ndarray
    weight: np.ndarray


@dataclass
class TrainingContext:
    """Store frozen context shared by candidate fits."""

    config: dict[str, Any]
    models: ModelBundle
    relations: RelationBundle
    settings: TrainingSettings
    device: torch.device
    matrix_values: dict[str, np.ndarray]
    matrix_observed: dict[str, np.ndarray]
    raw_primary: np.ndarray
    correction_scale: float
    labels: pd.DataFrame
    positive_comparisons: dict[str, PositiveComparisons]
    standardizer: FittedStandardizer
    anchor_row: np.ndarray
    anchor_column: np.ndarray
    label_features: np.ndarray
    label_raw: np.ndarray
    label_base: np.ndarray
    anchor_features: np.ndarray
    prepared_positive: dict[str, PreparedPositiveFeatures]
    domain_features: np.ndarray | None
    backbone_history: pd.DataFrame
    backbone_effective_weights: dict[str, float]


@dataclass
class TrainingResult:
    """Store one fitted candidate."""

    scorer: ResidualScorer
    train_rows: np.ndarray
    learnable_coefficient: bool
    residual_anchor_weight: float
    effective_loss_weights: dict[str, float]
    history: pd.DataFrame

    @property
    def primary_coefficient(self) -> float:
        """Return the final bounded primary coefficient."""
        return float(self.scorer.primary_coefficient().detach().cpu())


def seed_everything(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch RNG states."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    """Resolve an automatic or explicit device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def adaptive_spec(
    declaration: dict[str, Any],
    section: str,
) -> dict[str, Any] | None:
    """Read one optional adaptive-weight declaration."""
    value = declaration.get(section)
    if value is None or not bool(value.get("enabled", False)):
        return None
    required = {"lower_bound", "upper_bound", "initial_weights"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"adaptive_weights.{section} is missing {sorted(missing)}")
    initial = {str(name): float(weight) for name, weight in value["initial_weights"].items()}
    lower = float(value["lower_bound"])
    upper = float(value["upper_bound"])
    if not initial or lower <= 0 or upper <= lower or any(weight <= 0 for weight in initial.values()):
        raise ValueError(f"adaptive_weights.{section} contains invalid positive-weight settings")
    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "initial_weights": initial,
    }


def build_loss_balancer(
    specification: dict[str, Any] | None,
    expected_names: tuple[str, ...],
    device: torch.device,
) -> UncertaintyLossBalancer | None:
    """Build and validate an auxiliary-loss balancer."""
    if specification is None:
        return None
    initial = specification["initial_weights"]
    if set(initial) != set(expected_names):
        raise ValueError(
            "Adaptive loss names do not match the active objective: "
            f"expected {sorted(expected_names)}, got {sorted(initial)}"
        )
    return UncertaintyLossBalancer(
        initial,
        specification["lower_bound"],
        specification["upper_bound"],
    ).to(device)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash the complete configuration content."""
    return hashlib.sha256(
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def parse_settings(config: dict[str, Any]) -> TrainingSettings:
    """Read training parameters and validate semantic mappings."""
    declaration = config.get("training")
    if not isinstance(declaration, dict):
        raise ValueError("Configuration must declare training")
    binary = declaration["binary_supervision"]
    positive = declaration["positive_only_supervision"]
    model = config["model"]
    selection = config["selection"]
    adaptive = config.get("adaptive_weights", {})
    if not isinstance(adaptive, dict):
        raise ValueError("adaptive_weights must be a mapping when supplied")
    settings = TrainingSettings(
        binary_input=binary["input"],
        binary_label=binary["label"],
        binary_base_transform=binary["base_transform"],
        positive_input=positive["input"],
        positive_group=positive["group"],
        positive_confidence=positive.get("confidence"),
        missing_confidence=float(positive.get("missing_confidence", 1.0)),
        dense_batch_size=int(model["dense_batch_size"]),
        sparse_batch_size=int(model["sparse_batch_size"]),
        steps_per_epoch=int(model["steps_per_epoch"]),
        backbone_epochs=int(model["dense_pretrain_epochs"]),
        backbone_learning_rate=float(model["pretrain_learning_rate"]),
        backbone_sparse_weight=float(model["pretrain_sparse_weight"]),
        backbone_weight_decay=float(model["pretrain_weight_decay"]),
        feature_sample_size=int(model["feature_sample_size"]),
        anchor_sample_size=int(model["anchor_sample_size"]),
        anchor_train_fraction=float(model["anchor_train_fraction"]),
        anchor_batch_size=int(model["anchor_batch_size"]),
        positive_weight=float(model["positive_relation_weight"]),
        unobserved_pair_ratio=int(model["unobserved_pair_ratio"]),
        positive_batch_size=int(model["positive_relation_batch_size"]),
        positive_temperature=float(model["relation_ranking_temperature"]),
        source_seed_offsets={
            str(source): int(offset)
            for source, offset in positive["source_seed_offsets"].items()
        },
        head_epochs=int(model["head_epochs"]),
        head_learning_rate=float(model["head_learning_rate"]),
        ranking_temperature=float(model["ranking_temperature"]),
        ranking_weight=float(model["ranking_weight"]),
        pointwise_weight=float(model["pointwise_weight"]),
        residual_anchor_weight=float(selection["residual_anchor_weight"]),
        head_weight_decay=float(model.get("head_weight_decay", 0.0)),
        adaptive_backbone=adaptive_spec(adaptive, "backbone"),
        adaptive_head=adaptive_spec(adaptive, "head"),
        seed=int(model["seed"]),
    )
    positive_numbers = (
        settings.dense_batch_size,
        settings.sparse_batch_size,
        settings.steps_per_epoch,
        settings.backbone_epochs,
        settings.feature_sample_size,
        settings.anchor_sample_size,
        settings.anchor_batch_size,
        settings.unobserved_pair_ratio,
        settings.positive_batch_size,
        settings.head_epochs,
    )
    if min(positive_numbers) <= 0:
        raise ValueError("Training counts and batch sizes must be positive")
    if min(
        settings.backbone_learning_rate,
        settings.backbone_sparse_weight,
        settings.head_learning_rate,
        settings.positive_temperature,
        settings.ranking_temperature,
        settings.positive_weight,
    ) <= 0:
        raise ValueError("Learning rates, temperatures, and positive-relation weight must be positive")
    if min(settings.backbone_weight_decay, settings.head_weight_decay) < 0:
        raise ValueError("Weight decay must be non-negative")
    if not 0 < settings.anchor_train_fraction < 1:
        raise ValueError("Anchor train fraction is invalid")
    if min(settings.ranking_weight, settings.pointwise_weight, settings.residual_anchor_weight) < 0:
        raise ValueError("Loss weights must be non-negative")
    return settings


def relation_sampler(name: str, values: np.ndarray) -> RelationSampler:
    """Build a complete or sparse relation sampler."""
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"Backbone relation {name} has no finite values")
    flat = None if finite.all() else np.flatnonzero(finite.ravel())
    return RelationSampler(name, values, flat)


def tensor(values: np.ndarray, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Build a tensor on the selected device."""
    return torch.tensor(values, device=device, dtype=dtype)


def pair_values(
    context: TrainingContext,
    row: np.ndarray,
    column: np.ndarray,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Gather all model relation values and masks."""
    names = context.models.architecture.backbone_relations
    values = {name: tensor(context.matrix_values[name][row, column], context.device) for name in names}
    observed = {
        name: tensor(context.matrix_observed[name][row, column], context.device, torch.bool)
        for name in context.models.architecture.sparse_features
    }
    return values, observed


@torch.no_grad()
def raw_pair_features(
    context: TrainingContext,
    row: np.ndarray,
    column: np.ndarray,
    batch_size: int = 131072,
) -> np.ndarray:
    """Build raw pair features in chunks."""
    context.models.backbone.eval()
    output = np.empty((len(row), len(context.models.feature_names)), dtype=np.float32)
    for start in range(0, len(row), batch_size):
        stop = min(start + batch_size, len(row))
        row_part, column_part = row[start:stop], column[start:stop]
        values, observed = pair_values(context, row_part, column_part)
        features = build_pair_features(
            context.models.backbone,
            context.models.architecture,
            tensor(row_part, context.device, torch.long),
            tensor(column_part, context.device, torch.long),
            values,
            observed,
        )
        output[start:stop] = features.cpu().numpy()
    return output


def scaled_pair_features(context: TrainingContext, row: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Build and standardize pair features."""
    return context.standardizer.transform(raw_pair_features(context, row, column))


def pretrain_backbone(
    context: TrainingContext,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> pd.DataFrame:
    """Pretrain the shared backbone on declared relations."""
    settings = context.settings
    epochs = settings.backbone_epochs if epochs is None else epochs
    steps = settings.steps_per_epoch if steps_per_epoch is None else steps_per_epoch
    rng = np.random.default_rng(settings.seed)
    architecture = context.models.architecture
    primary = architecture.primary_feature
    auxiliary = architecture.auxiliary_feature
    sparse = architecture.sparse_features[0]
    sparse_sampler = relation_sampler(sparse, context.matrix_values[sparse])
    balancer = build_loss_balancer(
        settings.adaptive_backbone,
        (auxiliary, sparse),
        context.device,
    )
    with torch.no_grad():
        context.models.backbone.relation_bias[
            context.models.backbone.relation_index[primary]
        ].fill_(0.5)
        context.models.backbone.relation_bias[
            context.models.backbone.relation_index[auxiliary]
        ].fill_(0.5)
        sparse_mean = float(np.nanmean(context.matrix_values[sparse]))
        context.models.backbone.relation_bias[
            context.models.backbone.relation_index[sparse]
        ].fill_(float(np.log(np.clip(sparse_mean, 1e-5, 1 - 1e-5) / np.clip(1 - sparse_mean, 1e-5, 1))))
    parameters: list[dict[str, Any]] = [
        {
            "params": list(context.models.backbone.parameters()),
            "weight_decay": settings.backbone_weight_decay,
        }
    ]
    if balancer is not None:
        parameters.append({"params": list(balancer.parameters()), "weight_decay": 0.0})
    optimizer = torch.optim.Adam(
        parameters,
        lr=settings.backbone_learning_rate,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        totals = {"total": 0.0, primary: 0.0, auxiliary: 0.0, sparse: 0.0}
        context.models.backbone.train()
        for _ in range(steps):
            row = rng.integers(
                0,
                context.matrix_values[primary].shape[0],
                size=settings.dense_batch_size,
                dtype=np.int64,
            )
            column = rng.integers(
                0,
                context.matrix_values[primary].shape[1],
                size=settings.dense_batch_size,
                dtype=np.int64,
            )
            row_t = tensor(row, context.device, torch.long)
            column_t = tensor(column, context.device, torch.long)
            primary_loss = dense_reconstruction_loss(
                context.models.backbone.decode(row_t, column_t, primary),
                tensor(context.matrix_values[primary][row, column], context.device),
            )
            auxiliary_loss = dense_reconstruction_loss(
                context.models.backbone.decode(row_t, column_t, auxiliary),
                tensor(context.matrix_values[auxiliary][row, column], context.device),
            )
            sparse_row, sparse_column, sparse_target = sparse_sampler.sample(
                settings.sparse_batch_size, rng
            )
            sparse_loss = dense_reconstruction_loss(
                context.models.backbone.decode(
                    tensor(sparse_row, context.device, torch.long),
                    tensor(sparse_column, context.device, torch.long),
                    sparse,
                ),
                tensor(sparse_target, context.device),
            )
            auxiliary_losses = {
                auxiliary: auxiliary_loss,
                sparse: sparse_loss,
            }
            total = (
                primary_loss + balancer.objective(auxiliary_losses)
                if balancer is not None
                else primary_loss + auxiliary_loss + settings.backbone_sparse_weight * sparse_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if balancer is not None:
                balancer.project()
            totals["total"] += float(total.detach().cpu())
            totals[primary] += float(primary_loss.detach().cpu())
            totals[auxiliary] += float(auxiliary_loss.detach().cpu())
            totals[sparse] += float(sparse_loss.detach().cpu())
        record = {
            "epoch": epoch,
            **{f"{name}_loss": value / steps for name, value in totals.items()},
        }
        effective = (
            balancer.detached_weights()
            if balancer is not None
            else {auxiliary: 1.0, sparse: settings.backbone_sparse_weight}
        )
        record.update({f"{name}_effective_weight": weight for name, weight in effective.items()})
        history.append(record)
    context.backbone_effective_weights = (
        balancer.detached_weights()
        if balancer is not None
        else {auxiliary: 1.0, sparse: settings.backbone_sparse_weight}
    )
    for parameter in context.models.backbone.parameters():
        parameter.requires_grad_(False)
    return pd.DataFrame(history)


def sample_domain(
    n_entity_a: int,
    n_entity_b: int,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample coordinates from the full relation domain."""
    return (
        rng.integers(0, n_entity_a, size=size, dtype=np.int64),
        rng.integers(0, n_entity_b, size=size, dtype=np.int64),
    )


def sample_anchor_domain(
    n_entity_a: int,
    n_entity_b: int,
    size: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample legacy anchor training coordinates without replacement."""
    n_total = n_entity_a * n_entity_b
    flat = rng.choice(n_total, size=min(size, n_total), replace=False)
    order = rng.permutation(len(flat))
    audit_size = max(1, int(round((1.0 - train_fraction) * len(flat))))
    train = flat[order[audit_size:]]
    return (train // n_entity_b).astype(np.int64), (train % n_entity_b).astype(np.int64)


def map_pair_frame(
    relation: PairRelation,
    row_index: dict[str, int],
    column_index: dict[str, int],
) -> pd.DataFrame:
    """Map entity identifiers to domain coordinates."""
    frame = relation.frame.copy()
    frame["entity_a_index"] = frame["entity_a"].map(row_index).astype(int)
    frame["entity_b_index"] = frame["entity_b"].map(column_index).astype(int)
    return frame


def prepare_positive_comparisons(
    frame: pd.DataFrame,
    settings: TrainingSettings,
    n_entity_b: int,
) -> dict[str, PositiveComparisons]:
    """Build same-entity unobserved comparators per positive relation type."""
    comparisons: dict[str, PositiveComparisons] = {}
    for group, seed_offset in settings.source_seed_offsets.items():
        part = frame.loc[frame[settings.positive_group].astype(str) == group]
        if part.empty:
            raise ValueError(f"Positive-only supervision has no rows for source {group}")
        rng = np.random.default_rng(settings.seed + seed_offset)
        positive_lookup = {
            (int(row), int(column))
            for row, column in zip(part["entity_a_index"], part["entity_b_index"])
        }
        positive_row = np.repeat(part["entity_a_index"].to_numpy(np.int64), settings.unobserved_pair_ratio)
        positive_column = np.repeat(part["entity_b_index"].to_numpy(np.int64), settings.unobserved_pair_ratio)
        negative_column = rng.integers(0, n_entity_b, size=len(positive_row), dtype=np.int64)
        invalid = np.asarray(
            [(int(row), int(column)) in positive_lookup for row, column in zip(positive_row, negative_column)]
        )
        while invalid.any():
            negative_column[invalid] = rng.integers(0, n_entity_b, size=int(invalid.sum()), dtype=np.int64)
            invalid = np.asarray(
                [(int(row), int(column)) in positive_lookup for row, column in zip(positive_row, negative_column)]
            )
        if settings.positive_confidence is None:
            confidence = np.ones(len(part), dtype=np.float32)
        else:
            confidence = pd.to_numeric(part[settings.positive_confidence]).fillna(settings.missing_confidence)
            confidence = confidence.to_numpy(np.float32)
        confidence = np.clip(confidence, 1e-6, None)
        confidence /= confidence.mean()
        repeated_confidence = np.repeat(confidence, settings.unobserved_pair_ratio).astype(np.float64)
        weight = np.zeros(len(positive_row), dtype=np.float64)
        for entity_a in np.unique(positive_row):
            mask = positive_row == entity_a
            weight[mask] = repeated_confidence[mask] / repeated_confidence[mask].sum()
        weight = (weight / len(np.unique(positive_row))).astype(np.float32)
        comparisons[str(group)] = PositiveComparisons(
            str(group), positive_row, positive_column, negative_column, weight
        )
    if not comparisons:
        raise ValueError("Positive-only supervision contains no relation groups")
    return comparisons


def prepare_context(
    config_path: str | Path,
    domain_name: str | None = None,
    device_name: str = "auto",
    backbone_checkpoint: str | Path | None = None,
    backbone_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    feature_sample_size: int | None = None,
    anchor_sample_size: int | None = None,
) -> TrainingContext:
    """Load data, pretrain the backbone, and freeze shared context."""
    config = load_config(config_path)
    settings = parse_settings(config)
    seed_everything(settings.seed)
    models = build_models(config_path, domain_name)
    relations = getattr(models.transforms, "relations", None)
    if relations is None:
        relations = load_relations(config_path, domain_name)
    device = choose_device(device_name)
    models.backbone.to(device)
    models.scorer.to(device)
    matrix_values = {
        name: matrix.values.to_numpy(np.float32, copy=False)
        for name, matrix in models.transforms.matrices.items()
    }
    matrix_observed = {
        name: matrix.observed.to_numpy(bool, copy=False)
        for name, matrix in models.transforms.matrices.items()
    }
    raw_relation = relations.matrices[models.architecture.primary_input].values
    raw_primary = fill_row_means(raw_relation).to_numpy(np.float32)
    row_index = {value: index for index, value in enumerate(relations.domain.entity_a)}
    column_index = {value: index for index, value in enumerate(relations.domain.entity_b)}

    binary = map_pair_frame(relations.pair_tables[settings.binary_input], row_index, column_index)
    binary["label"] = pd.to_numeric(binary[settings.binary_label]).astype(int)
    binary.insert(0, "relation_row", np.arange(len(binary), dtype=np.int64))
    positive = map_pair_frame(relations.pair_tables[settings.positive_input], row_index, column_index)
    comparisons = prepare_positive_comparisons(positive, settings, len(column_index))

    placeholder = FittedStandardizer(
        np.zeros(len(models.feature_names), dtype=np.float64),
        np.ones(len(models.feature_names), dtype=np.float64),
    )
    context = TrainingContext(
        config=config,
        models=models,
        relations=relations,
        settings=settings,
        device=device,
        matrix_values=matrix_values,
        matrix_observed=matrix_observed,
        raw_primary=raw_primary,
        correction_scale=median_row_scale(raw_primary),
        labels=binary,
        positive_comparisons=comparisons,
        standardizer=placeholder,
        anchor_row=np.empty(0, dtype=np.int64),
        anchor_column=np.empty(0, dtype=np.int64),
        label_features=np.empty((0, len(models.feature_names)), dtype=np.float32),
        label_raw=np.empty(0, dtype=np.float32),
        label_base=np.empty(0, dtype=np.float32),
        anchor_features=np.empty((0, len(models.feature_names)), dtype=np.float32),
        prepared_positive={},
        domain_features=None,
        backbone_history=pd.DataFrame(),
        backbone_effective_weights={},
    )
    checkpoint = Path(backbone_checkpoint) if backbone_checkpoint is not None else None
    expected_backbone_epochs = (
        settings.backbone_epochs if backbone_epochs is None else int(backbone_epochs)
    )
    expected_steps_per_epoch = (
        settings.steps_per_epoch if steps_per_epoch is None else int(steps_per_epoch)
    )
    if checkpoint is not None and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=context.device)
        if payload["config_fingerprint"] != config_fingerprint(config):
            raise ValueError("Backbone checkpoint was created from a different configuration")
        if (
            int(payload["backbone_epochs"]) != expected_backbone_epochs
            or int(payload["steps_per_epoch"]) != expected_steps_per_epoch
        ):
            raise ValueError("Backbone checkpoint was created with a different training budget")
        context.models.backbone.load_state_dict(payload["backbone_state"])
        context.backbone_effective_weights = {
            str(name): float(weight)
            for name, weight in payload["backbone_effective_weights"].items()
        }
        context.backbone_history = pd.DataFrame(payload["backbone_history"])
        for parameter in context.models.backbone.parameters():
            parameter.requires_grad_(False)
    else:
        context.backbone_history = pretrain_backbone(
            context,
            backbone_epochs,
            steps_per_epoch,
        )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config_fingerprint": config_fingerprint(config),
                    "backbone_epochs": expected_backbone_epochs,
                    "steps_per_epoch": expected_steps_per_epoch,
                    "backbone_state": context.models.backbone.state_dict(),
                    "backbone_effective_weights": context.backbone_effective_weights,
                    "backbone_history": context.backbone_history.to_dict(orient="list"),
                },
                checkpoint,
            )
    rng = np.random.default_rng(settings.seed + 101)
    feature_size = settings.feature_sample_size if feature_sample_size is None else feature_sample_size
    feature_row, feature_column = sample_domain(len(row_index), len(column_index), feature_size, rng)
    context.standardizer = fit_standardizer(raw_pair_features(context, feature_row, feature_column))
    anchor_size = settings.anchor_sample_size if anchor_sample_size is None else anchor_sample_size
    context.anchor_row, context.anchor_column = sample_anchor_domain(
        len(row_index),
        len(column_index),
        anchor_size,
        settings.anchor_train_fraction,
        np.random.default_rng(settings.seed + 151),
    )
    label_row = context.labels["entity_a_index"].to_numpy(np.int64)
    label_column = context.labels["entity_b_index"].to_numpy(np.int64)
    context.label_features = scaled_pair_features(context, label_row, label_column)
    context.label_raw = context.raw_primary[label_row, label_column]
    base_matrix = context.matrix_values[settings.binary_base_transform]
    context.label_base = stable_logit(base_matrix[label_row, label_column]).astype(
        np.float32,
        copy=False,
    )
    context.anchor_features = scaled_pair_features(
        context,
        context.anchor_row,
        context.anchor_column,
    )
    context.prepared_positive = {
        name: PreparedPositiveFeatures(
            positive_features=scaled_pair_features(
                context,
                comparison.positive_row,
                comparison.positive_column,
            ),
            negative_features=scaled_pair_features(
                context,
                comparison.positive_row,
                comparison.negative_column,
            ),
            positive_raw=context.raw_primary[
                comparison.positive_row,
                comparison.positive_column,
            ],
            negative_raw=context.raw_primary[
                comparison.positive_row,
                comparison.negative_column,
            ],
            weight=comparison.weight,
        )
        for name, comparison in context.positive_comparisons.items()
    }
    return context


def context_fingerprint(context: TrainingContext) -> str:
    """Hash frozen context to prevent mixed-backbone resumes."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            context.config,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    for name, value in sorted(context.models.backbone.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(np.asarray(context.standardizer.mean).tobytes())
    digest.update(np.asarray(context.standardizer.scale).tobytes())
    digest.update(context.anchor_row.tobytes())
    digest.update(context.anchor_column.tobytes())
    digest.update(context.labels["entity_a"].astype(str).str.cat(sep="\0").encode("utf-8"))
    digest.update(context.labels["entity_b"].astype(str).str.cat(sep="\0").encode("utf-8"))
    digest.update(context.labels["label"].to_numpy(np.int8).tobytes())
    digest.update(
        json.dumps(context.backbone_effective_weights, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()


def cache_domain_features(
    context: TrainingContext,
    row_chunk_size: int = 8,
) -> np.ndarray:
    """Cache frozen full-domain features for all repeats."""
    if context.domain_features is not None:
        return context.domain_features
    n_rows, n_columns = context.raw_primary.shape
    output = np.empty(
        (n_rows, n_columns, len(context.models.feature_names)),
        dtype=np.float32,
    )
    columns = np.arange(n_columns, dtype=np.int64)
    for start in range(0, n_rows, row_chunk_size):
        stop = min(start + row_chunk_size, n_rows)
        row = np.repeat(np.arange(start, stop, dtype=np.int64), n_columns)
        column = np.tile(columns, stop - start)
        output[start:stop] = scaled_pair_features(context, row, column).reshape(
            stop - start,
            n_columns,
            -1,
        )
    context.domain_features = output
    return output


def within_entity_pairs(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build within-entity label pairs with equal entity weight."""
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    informative = []
    for _, part in labels.groupby("entity_a_index", sort=False):
        positive = np.flatnonzero(part["label"].to_numpy() == 1)
        negative = np.flatnonzero(part["label"].to_numpy() == 0)
        if len(positive) and len(negative):
            informative.append((part.index.to_numpy()[positive], part.index.to_numpy()[negative]))
    if not informative:
        raise ValueError("Training rows contain no entity with both binary classes")
    entity_weight = 1.0 / len(informative)
    for positive, negative in informative:
        pos = np.repeat(positive, len(negative))
        neg = np.tile(negative, len(positive))
        positive_parts.append(pos)
        negative_parts.append(neg)
        weight_parts.append(np.full(len(pos), entity_weight / len(pos), dtype=np.float32))
    return np.concatenate(positive_parts), np.concatenate(negative_parts), np.concatenate(weight_parts)


def candidate_scorer(context: TrainingContext, learnable_coefficient: bool) -> ResidualScorer:
    """Build a fresh candidate head to prevent state reuse across folds."""
    template = context.models.scorer
    scorer = ResidualScorer(
        len(context.models.feature_names),
        int(context.config["model"]["head_hidden_dim"]),
        template.coefficient_bounds,
        learnable_coefficient,
    )
    return scorer.to(context.device)


def final_score(
    context: TrainingContext,
    scorer: ResidualScorer,
    raw: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Combine final scores with the formal raw-scale formula."""
    return scorer.final_score(
        raw,
        residual,
        context.correction_scale,
        context.models.architecture.primary_direction,
    )


def fit_candidate(
    context: TrainingContext,
    train_rows: np.ndarray,
    learnable_coefficient: bool,
    residual_anchor_weight: float | None = None,
    seed: int | None = None,
    epochs: int | None = None,
) -> TrainingResult:
    """Fit one candidate using only the supplied binary rows."""
    settings = context.settings
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if train_rows.ndim != 1 or len(train_rows) == 0 or len(np.unique(train_rows)) != len(train_rows):
        raise ValueError("train_rows must be a non-empty unique one-dimensional array")
    if train_rows.min() < 0 or train_rows.max() >= len(context.labels):
        raise ValueError("train_rows contains an out-of-range relation row")
    anchor_weight = settings.residual_anchor_weight if residual_anchor_weight is None else float(residual_anchor_weight)
    if anchor_weight < 0:
        raise ValueError("residual_anchor_weight must be non-negative")
    fit_seed = settings.seed if seed is None else int(seed)
    seed_everything(fit_seed)
    rng = np.random.default_rng(fit_seed)
    scorer = candidate_scorer(context, learnable_coefficient)
    balancer = build_loss_balancer(
        settings.adaptive_head,
        ("pointwise", "positive_relation"),
        context.device,
    )
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter for parameter in scorer.parameters() if parameter.requires_grad
            ],
            "weight_decay": settings.head_weight_decay,
        }
    ]
    if balancer is not None:
        parameter_groups.append({"params": list(balancer.parameters()), "weight_decay": 0.0})
    optimizer = torch.optim.Adam(
        parameter_groups,
        lr=settings.head_learning_rate,
    )

    labels = context.labels.iloc[train_rows].copy().reset_index(drop=True)
    features = tensor(context.label_features[train_rows], context.device)
    y = tensor(labels["label"].to_numpy(np.float32), context.device)
    raw = tensor(context.label_raw[train_rows], context.device)
    base = tensor(context.label_base[train_rows], context.device)
    pair_positive, pair_negative, pair_weight = within_entity_pairs(labels)
    pair_positive_t = tensor(pair_positive, context.device, torch.long)
    pair_negative_t = tensor(pair_negative, context.device, torch.long)
    pair_weight_t = tensor(pair_weight, context.device)
    n_epochs = settings.head_epochs if epochs is None else int(epochs)
    history: list[dict[str, float | int | bool]] = []
    for epoch in range(1, n_epochs + 1):
        scorer.train()
        residual = scorer.residual(features)
        final = final_score(context, scorer, raw, residual)
        ranking = pairwise_ranking_loss(
            final[pair_positive_t],
            final[pair_negative_t],
            context.models.architecture.primary_direction,
            settings.ranking_temperature,
            pair_weight_t,
        )
        pointwise = binary_supervision_loss(scorer.binary_logits(features, base), y)
        anchor_index = rng.integers(
            0,
            len(context.anchor_features),
            size=min(settings.anchor_batch_size, len(context.anchor_features)),
        )
        anchor_tensor = tensor(context.anchor_features[anchor_index], context.device)
        anchor = residual_anchor_loss(scorer.residual(anchor_tensor))

        positive_losses = []
        for prepared in context.prepared_positive.values():
            index = rng.integers(
                0,
                len(prepared.weight),
                size=min(settings.positive_batch_size, len(prepared.weight)),
            )
            pos_x = tensor(prepared.positive_features[index], context.device)
            neg_x = tensor(prepared.negative_features[index], context.device)
            pos_final = final_score(
                context,
                scorer,
                tensor(prepared.positive_raw[index], context.device),
                scorer.residual(pos_x),
            )
            neg_final = final_score(
                context,
                scorer,
                tensor(prepared.negative_raw[index], context.device),
                scorer.residual(neg_x),
            )
            positive_losses.append(
                pairwise_ranking_loss(
                    pos_final,
                    neg_final,
                    context.models.architecture.primary_direction,
                    settings.positive_temperature,
                    tensor(prepared.weight[index], context.device),
                )
            )
        positive_loss = torch.stack(positive_losses).mean()
        adaptive_losses = {
            "pointwise": pointwise,
            "positive_relation": positive_loss,
        }
        auxiliary_objective = (
            balancer.objective(adaptive_losses)
            if balancer is not None
            else settings.pointwise_weight * pointwise + settings.positive_weight * positive_loss
        )
        total = settings.ranking_weight * ranking + anchor_weight * anchor + auxiliary_objective
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if balancer is not None:
            balancer.project()
        scorer.project_primary_coefficient()
        effective = (
            balancer.detached_weights()
            if balancer is not None
            else {
                "pointwise": settings.pointwise_weight,
                "positive_relation": settings.positive_weight,
            }
        )
        history.append(
            {
                "epoch": epoch,
                "total_loss": float(total.detach().cpu()),
                "binary_ranking_loss": float(ranking.detach().cpu()),
                "binary_pointwise_loss": float(pointwise.detach().cpu()),
                "residual_anchor_loss": float(anchor.detach().cpu()),
                "positive_relation_loss": float(positive_loss.detach().cpu()),
                "pointwise_effective_weight": effective["pointwise"],
                "positive_relation_effective_weight": effective["positive_relation"],
                "primary_coefficient": float(scorer.primary_coefficient().detach().cpu()),
                "coefficient_learnable": learnable_coefficient,
            }
        )
    effective = (
        balancer.detached_weights()
        if balancer is not None
        else {
            "pointwise": settings.pointwise_weight,
            "positive_relation": settings.positive_weight,
        }
    )
    return TrainingResult(
        scorer,
        train_rows,
        learnable_coefficient,
        anchor_weight,
        effective,
        pd.DataFrame(history),
    )


@torch.no_grad()
def predict_candidate(
    context: TrainingContext,
    result: TrainingResult,
    relation_rows: np.ndarray,
) -> pd.DataFrame:
    """Predict specified binary-relation rows."""
    relation_rows = np.asarray(relation_rows, dtype=np.int64)
    selected = context.labels.iloc[relation_rows].copy()
    features = tensor(context.label_features[relation_rows], context.device)
    base = tensor(
        context.label_base[relation_rows],
        context.device,
    )
    raw = tensor(context.label_raw[relation_rows], context.device)
    result.scorer.eval()
    residual = result.scorer.residual(features)
    probability = torch.sigmoid(result.scorer.binary_logits(features, base))
    final = final_score(context, result.scorer, raw, residual)
    return pd.DataFrame(
        {
            "relation_row": selected["relation_row"].to_numpy(),
            "entity_a": selected["entity_a"].to_numpy(),
            "entity_b": selected["entity_b"].to_numpy(),
            "label": selected["label"].to_numpy(),
            "probability": probability.cpu().numpy(),
            "residual": residual.cpu().numpy(),
            "final_score": final.cpu().numpy(),
        }
    )


def main() -> None:
    """Run real-data assembly validation or a short training test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a two-epoch all-label smoke fit.")
    args = parser.parse_args()

    context = prepare_context(
        args.config,
        args.domain,
        args.device,
        backbone_epochs=1 if args.smoke_test else 1,
        steps_per_epoch=2 if args.smoke_test else 1,
        feature_sample_size=4096,
        anchor_sample_size=4096,
    )
    print(
        f"PASS  domain {context.relations.domain.name}: "
        f"{len(context.relations.domain.entity_a)} x {len(context.relations.domain.entity_b)}"
    )
    print(
        f"PASS  training relations: {len(context.models.architecture.backbone_relations)} backbone, "
        f"{len(context.positive_comparisons)} positive-only groups, {len(context.labels)} binary rows"
    )
    print(f"PASS  feature standardizer: {len(context.standardizer.mean)} features")
    print(f"PASS  raw correction scale: {context.correction_scale:.8g}")
    if args.smoke_test:
        result = fit_candidate(
            context,
            np.arange(len(context.labels)),
            learnable_coefficient=True,
            epochs=2,
        )
        predictions = predict_candidate(context, result, np.arange(len(context.labels)))
        if not np.isfinite(predictions[["probability", "residual", "final_score"]].to_numpy()).all():
            raise RuntimeError("Smoke predictions contain non-finite values")
        print(
            f"PASS  candidate smoke fit: 2 epochs, beta={result.primary_coefficient:.6f}, "
            f"predictions {len(predictions)}"
        )
    print("PASS  training module")


if __name__ == "__main__":
    main()
