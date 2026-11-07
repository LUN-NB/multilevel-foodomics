"""Shared CPI-MEC training kernels for the frozen three-parameter workflow.

This module contains only the training and prediction primitives shared by
coarse search, continuous TPE, M1, and Final. It contains no legacy strategy
comparison, Reference guard, one-SE selection, or parameter fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


SCRIPT_VERSION = "2.1.0"
SCRIPT_RELEASE = "shared_exact_best_cached_features_20260729"


@dataclass(frozen=True)
class PreparedCPISupervision:
    """Store one fold's CPI supervision."""

    row: torch.Tensor
    column: torch.Tensor
    values: dict[str, torch.Tensor]
    observed: dict[str, torch.Tensor]
    mean: torch.Tensor
    scale: torch.Tensor
    label: torch.Tensor
    base: torch.Tensor
    raw: torch.Tensor
    pair_positive: torch.Tensor
    pair_negative: torch.Tensor
    pair_weight: torch.Tensor


def build_experiment_config(
    base_path: Path,
    output_dir: Path,
) -> Path:
    """Build the fixed training config."""
    with base_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    project_root = (
        base_path.parent / config.get("path_base", ".")
    ).resolve()
    config["path_base"] = str(project_root)
    config["model"]["pretrain_sparse_weight"] = 0.25
    config["model"]["pointwise_weight"] = 0.1
    config["model"]["positive_relation_weight"] = 1.0
    config["selection"]["residual_anchor_weight"] = 0.1
    config["adaptive_weights"]["backbone"]["enabled"] = False
    config["adaptive_weights"]["head"]["enabled"] = False
    config_path = output_dir / "experiment_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return config_path


def tensor(
    values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a tensor on the selected device."""
    return torch.tensor(values, device=device, dtype=dtype)


def cpi_loss(
    context,
    scorer,
    prepared: PreparedCPISupervision,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the formal CPI loss."""
    from models import (
        binary_supervision_loss,
        build_pair_features,
        pairwise_ranking_loss,
    )
    from training import final_score

    features = build_pair_features(
        context.models.backbone,
        context.models.architecture,
        prepared.row,
        prepared.column,
        prepared.values,
        prepared.observed,
    )
    features = (features - prepared.mean) / prepared.scale
    residual = scorer.residual(features)
    score = final_score(context, scorer, prepared.raw, residual)
    ranking = pairwise_ranking_loss(
        score[prepared.pair_positive],
        score[prepared.pair_negative],
        context.models.architecture.primary_direction,
        context.settings.ranking_temperature,
        prepared.pair_weight,
    )
    pointwise = binary_supervision_loss(
        (
            scorer.binary_bias
            + scorer.positive_scale(scorer.binary_scale_raw) * prepared.base
            + residual
        ),
        prepared.label,
    )
    main = ranking + context.settings.pointwise_weight * pointwise
    return main, ranking, pointwise


def prepare_cpi_supervision(
    context,
    relation_rows: np.ndarray,
) -> PreparedCPISupervision:
    """Prepare one fold once."""
    from training import pair_values, within_entity_pairs

    labels = (
        context.labels.iloc[relation_rows]
        .copy()
        .reset_index(drop=True)
    )
    row = labels["entity_a_index"].to_numpy(np.int64)
    column = labels["entity_b_index"].to_numpy(np.int64)
    values, observed = pair_values(context, row, column)
    positive, negative, weight = within_entity_pairs(labels)
    return PreparedCPISupervision(
        tensor(row, context.device, torch.long),
        tensor(column, context.device, torch.long),
        values,
        observed,
        tensor(
            context.standardizer.mean.astype(np.float32),
            context.device,
        ),
        tensor(
            context.standardizer.scale.astype(np.float32),
            context.device,
        ),
        tensor(
            labels["label"].to_numpy(np.float32),
            context.device,
        ),
        tensor(context.label_base[relation_rows], context.device),
        tensor(context.label_raw[relation_rows], context.device),
        tensor(positive, context.device, torch.long),
        tensor(negative, context.device, torch.long),
        tensor(weight, context.device),
    )


def relation_losses(
    context,
    rng: np.random.Generator,
    sparse_sampler,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute U/G/B losses."""
    from models import dense_reconstruction_loss

    architecture = context.models.architecture
    primary = architecture.primary_feature
    auxiliary = architecture.auxiliary_feature
    sparse = architecture.sparse_features[0]
    n_row, n_column = context.matrix_values[primary].shape
    row = rng.integers(
        0,
        n_row,
        size=context.settings.dense_batch_size,
        dtype=np.int64,
    )
    column = rng.integers(
        0,
        n_column,
        size=context.settings.dense_batch_size,
        dtype=np.int64,
    )
    row_t = tensor(row, context.device, torch.long)
    column_t = tensor(column, context.device, torch.long)
    interaction = context.models.backbone.interaction(row_t, column_t)
    primary_prediction = context.models.backbone.decode_interaction(
        interaction,
        primary,
    )
    graphban_prediction = context.models.backbone.decode_interaction(
        interaction,
        auxiliary,
    )
    if (
        context.models.backbone.decoder_outputs[primary]
        == "sigmoid"
    ):
        primary_prediction = torch.sigmoid(primary_prediction)
    if (
        context.models.backbone.decoder_outputs[auxiliary]
        == "sigmoid"
    ):
        graphban_prediction = torch.sigmoid(graphban_prediction)
    primary_loss = dense_reconstruction_loss(
        primary_prediction,
        tensor(
            context.matrix_values[primary][row, column],
            context.device,
        ),
    )
    graphban_loss = dense_reconstruction_loss(
        graphban_prediction,
        tensor(
            context.matrix_values[auxiliary][row, column],
            context.device,
        ),
    )
    sparse_row, sparse_column, sparse_target = sparse_sampler.sample(
        context.settings.sparse_batch_size,
        rng,
    )
    boltz_loss = dense_reconstruction_loss(
        context.models.backbone.decode(
            tensor(sparse_row, context.device, torch.long),
            tensor(sparse_column, context.device, torch.long),
            sparse,
        ),
        tensor(sparse_target, context.device),
    )
    return primary_loss, graphban_loss, boltz_loss


def refresh_frozen_features(context) -> None:
    """Rebuild formal head features."""
    from training import (
        PreparedPositiveFeatures,
        raw_pair_features,
        scaled_pair_features,
    )
    from transforms import fit_standardizer

    coordinates = getattr(
        context,
        "_adaptive_feature_coordinates",
        None,
    )
    if coordinates is None:
        rng = np.random.default_rng(context.settings.seed + 101)
        n_row, n_column = context.raw_primary.shape
        coordinates = (
            rng.integers(
                0,
                n_row,
                size=context.settings.feature_sample_size,
                dtype=np.int64,
            ),
            rng.integers(
                0,
                n_column,
                size=context.settings.feature_sample_size,
                dtype=np.int64,
            ),
        )
        context._adaptive_feature_coordinates = coordinates
    feature_row, feature_column = coordinates
    context.standardizer = fit_standardizer(
        raw_pair_features(
            context,
            feature_row,
            feature_column,
        )
    )
    label_row = context.labels[
        "entity_a_index"
    ].to_numpy(np.int64)
    label_column = context.labels[
        "entity_b_index"
    ].to_numpy(np.int64)
    context.label_features = scaled_pair_features(
        context,
        label_row,
        label_column,
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


def evaluate_strategy(
    context,
    fitted,
    test_rows: np.ndarray,
    strategy: str,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Predict rows and compute metrics."""
    from evaluation import evaluate_predictions, ranking_score
    from training import predict_candidate

    prediction = predict_candidate(context, fitted, test_rows)
    prediction.insert(0, "strategy", strategy)
    prediction["ranking_score"] = ranking_score(
        prediction["final_score"],
        context.models.architecture.primary_direction,
    )
    metrics = evaluate_predictions(
        prediction,
        context.settings.ranking_temperature,
    )
    return prediction, {"strategy": strategy, **metrics}
