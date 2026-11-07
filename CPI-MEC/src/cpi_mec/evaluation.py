"""Refit selected candidates and evaluate untouched outer-test relations.

Each candidate uses its own inner-CV hyperparameter selected for the current
outer-training set. The module refits on that complete training set, predicts
the corresponding outer test set, and reports probability and ranking metrics.
It never selects a candidate or reads outer-test labels before fitting.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from selection import (
    COEFFICIENT_CANDIDATES,
    MaterializedSplits,
    audit_splits,
    binary_log_loss,
    load_materialized_splits,
)
from training import (
    TrainingContext,
    TrainingResult,
    context_fingerprint,
    fit_candidate,
    final_score,
    predict_candidate,
    prepare_context,
    scaled_pair_features,
    tensor,
)


TOP_FRACTIONS = (0.30, 0.50)


@dataclass(frozen=True)
class EvaluationResult:
    """Store outer-test predictions, metrics, and fit audit."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    training_history: pd.DataFrame
    domain_rows: pd.DataFrame
    domain_values: np.ndarray
    domain_all_rows: tuple[str, ...]
    domain_columns: tuple[str, ...]


@torch.no_grad()
def score_domain(
    context: TrainingContext,
    fitted: TrainingResult,
    row_indices: np.ndarray | None = None,
    column_indices: np.ndarray | None = None,
    row_chunk_size: int = 16,
) -> np.ndarray:
    """Score a rectangular domain in row chunks."""
    rows = (
        np.arange(context.raw_primary.shape[0])
        if row_indices is None
        else np.asarray(row_indices)
    )
    columns = (
        np.arange(context.raw_primary.shape[1])
        if column_indices is None
        else np.asarray(column_indices)
    )
    output = np.empty((len(rows), len(columns)), dtype=np.float32)
    fitted.scorer.eval()
    for start in range(0, len(rows), row_chunk_size):
        stop = min(start + row_chunk_size, len(rows))
        row = np.repeat(rows[start:stop], len(columns)).astype(np.int64)
        column = np.tile(columns, stop - start).astype(np.int64)
        if context.domain_features is None:
            feature_values = scaled_pair_features(context, row, column)
        else:
            feature_values = context.domain_features[
                rows[start:stop, None],
                columns[None, :],
            ].reshape(len(row), -1)
        features = tensor(feature_values, context.device)
        raw = tensor(context.raw_primary[row, column], context.device)
        residual = fitted.scorer.residual(features)
        values = final_score(context, fitted.scorer, raw, residual)
        output[start:stop] = values.cpu().numpy().reshape(
            stop - start,
            len(columns),
        )
    return output


def selected_hyperparameters(path: str | Path, repeat: int, outer_fold: int) -> pd.DataFrame:
    """Read both selected candidates for one outer fold."""
    selected = pd.read_csv(path)
    selected = selected[
        (selected["repeat"] == repeat) & (selected["outer_fold"] == outer_fold)
    ].copy()
    if set(selected["candidate"]) != set(COEFFICIENT_CANDIDATES) or len(selected) != 2:
        raise ValueError(f"repeat {repeat}, outer fold {outer_fold}: expected one selected row per candidate")
    return selected.sort_values("candidate").reset_index(drop=True)


def audit_selected_context(selected: pd.DataFrame, context: TrainingContext) -> None:
    """Verify selection and evaluation share one frozen context."""
    required = bool(
        context.config.get("selection", {}).get("require_context_fingerprint", False)
    )
    if not required:
        return
    if "context_fingerprint" not in selected:
        raise ValueError("Selected hyperparameters do not contain a context fingerprint")
    fingerprints = selected["context_fingerprint"].dropna().astype(str).unique()
    current = context_fingerprint(context)
    if len(fingerprints) != 1 or fingerprints[0] != current:
        raise ValueError(
            "Selected hyperparameters were produced by a different frozen context"
        )


def outer_rows(splits: MaterializedSplits, repeat: int, outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return complete outer-train and untouched test rows."""
    assignment = splits.outer[splits.outer["repeat"] == repeat]
    test = assignment.loc[assignment["outer_fold"] == outer_fold, "relation_row"].to_numpy(np.int64)
    train = assignment.loc[assignment["outer_fold"] != outer_fold, "relation_row"].to_numpy(np.int64)
    return train, test


def load_domain_assignments(path: str | Path) -> pd.DataFrame:
    """Read outer-fold assignments for all domain entities."""
    return pd.read_csv(
        Path(path) / "domain_assignments.csv",
        dtype={"entity_a": str, "group": str},
    )


def ranking_score(final_score: np.ndarray, direction: str) -> np.ndarray:
    """Convert final scores to higher-is-better ranking values."""
    if direction == "lower_is_better":
        return -np.asarray(final_score, dtype=np.float64)
    if direction == "higher_is_better":
        return np.asarray(final_score, dtype=np.float64)
    raise ValueError(f"Unsupported score direction: {direction}")


def within_entity_metrics(frame: pd.DataFrame, temperature: float) -> dict[str, float | int]:
    """Compute entity-equal ranking loss and AUROC."""
    losses: list[float] = []
    aurocs: list[float] = []
    for _, part in frame.groupby("entity_a", sort=False):
        positive = part.loc[part["label"] == 1, "ranking_score"].to_numpy(np.float64)
        negative = part.loc[part["label"] == 0, "ranking_score"].to_numpy(np.float64)
        if len(positive) == 0 or len(negative) == 0:
            continue
        margin = (positive[:, None] - negative[None, :]) / temperature
        losses.append(float(np.logaddexp(0.0, -margin).mean()))
        aurocs.append(float(roc_auc_score(part["label"], part["ranking_score"])))
    return {
        "pairwise_logistic_loss": float(np.mean(losses)),
        "macro_within_entity_auroc": float(np.mean(aurocs)),
        "median_within_entity_auroc": float(np.median(aurocs)),
        "n_informative_entity": len(aurocs),
    }


def top_fraction_metrics(labels: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | int]:
    """Compute fixed top-fraction metrics among test labels."""
    n_top = max(1, int(math.ceil(len(labels) * fraction)))
    predicted = np.zeros(len(labels), dtype=int)
    predicted[np.argsort(-scores, kind="stable")[:n_top]] = 1
    true_positive = int(np.sum((labels == 1) & (predicted == 1)))
    false_positive = int(np.sum((labels == 0) & (predicted == 1)))
    false_negative = int(np.sum((labels == 1) & (predicted == 0)))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    prefix = f"top{int(round(fraction * 100))}"
    return {
        f"{prefix}_n": n_top,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        f"{prefix}_mcc": float(matthews_corrcoef(labels, predicted)),
    }


def evaluate_predictions(frame: pd.DataFrame, temperature: float) -> dict[str, float | int]:
    """Summarize one candidate on one outer test fold."""
    labels = frame["label"].to_numpy(np.int64)
    probability = frame["probability"].to_numpy(np.float64)
    scores = frame["ranking_score"].to_numpy(np.float64)
    prevalence = float(labels.mean())
    positive_percentile = pd.Series(scores).rank(method="average", ascending=False).to_numpy() / len(scores)
    metrics: dict[str, float | int] = {
        "n_test": len(frame),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "positive_prevalence": prevalence,
        "probability_log_loss": binary_log_loss(labels, probability),
        "probability_auroc": float(roc_auc_score(labels, probability)),
        "probability_auprc": float(average_precision_score(labels, probability)),
        "ranking_pooled_auroc": float(roc_auc_score(labels, scores)),
        "ranking_pooled_auprc": float(average_precision_score(labels, scores)),
        "positive_median_rank_percentile": float(np.median(positive_percentile[labels == 1])),
    }
    metrics["probability_auprc_lift"] = metrics["probability_auprc"] / prevalence
    metrics["ranking_auprc_lift"] = metrics["ranking_pooled_auprc"] / prevalence
    metrics.update(within_entity_metrics(frame, temperature))
    for fraction in TOP_FRACTIONS:
        metrics.update(top_fraction_metrics(labels, scores, fraction))
    return metrics


def evaluate_outer_fold(
    context: TrainingContext,
    splits: MaterializedSplits,
    domain_assignments: pd.DataFrame,
    selected: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    epochs: int | None = None,
) -> EvaluationResult:
    """Refit and evaluate both candidates for one outer fold."""
    train_rows, test_rows = outer_rows(splits, repeat, outer_fold)
    fit_seed = context.settings.seed + repeat * 100000 + outer_fold * 10000 + 5000
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, float | int | str | bool]] = []
    coefficients: list[dict[str, float | int | str | bool]] = []
    histories: list[pd.DataFrame] = []
    domain_rows = domain_assignments[
        (domain_assignments["repeat"] == repeat)
        & (domain_assignments["outer_fold"] == outer_fold)
    ][["repeat", "outer_fold", "entity_a"]].copy()
    row_index = {entity: index for index, entity in enumerate(context.relations.domain.entity_a)}
    score_rows = domain_rows["entity_a"].map(row_index).to_numpy(np.int64)
    domain_values: np.ndarray | None = None
    direction = context.models.architecture.primary_direction

    for row in selected.itertuples(index=False):
        learnable = bool(COEFFICIENT_CANDIDATES[row.candidate])
        fitted = fit_candidate(
            context,
            train_rows,
            learnable_coefficient=learnable,
            residual_anchor_weight=float(row.residual_anchor_weight),
            seed=fit_seed,
            epochs=epochs,
        )
        prediction = predict_candidate(context, fitted, test_rows)
        prediction.insert(0, "outer_fold", outer_fold)
        prediction.insert(0, "repeat", repeat)
        prediction.insert(2, "candidate", row.candidate)
        prediction.insert(3, "coefficient_learnable", learnable)
        prediction.insert(4, "residual_anchor_weight", float(row.residual_anchor_weight))
        prediction.insert(5, "primary_coefficient", fitted.primary_coefficient)
        prediction["ranking_score"] = ranking_score(prediction["final_score"], direction)
        predictions.append(prediction)

        metric = evaluate_predictions(prediction, context.settings.ranking_temperature)
        metrics.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "candidate": row.candidate,
                "coefficient_learnable": learnable,
                "residual_anchor_weight": float(row.residual_anchor_weight),
                "primary_coefficient": fitted.primary_coefficient,
                "pointwise_effective_weight": fitted.effective_loss_weights["pointwise"],
                "positive_relation_effective_weight": fitted.effective_loss_weights[
                    "positive_relation"
                ],
                "score_direction": direction,
                **metric,
            }
        )
        coefficients.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "candidate": row.candidate,
                "coefficient_learnable": learnable,
                "residual_anchor_weight": float(row.residual_anchor_weight),
                "primary_coefficient": fitted.primary_coefficient,
                "pointwise_effective_weight": fitted.effective_loss_weights["pointwise"],
                "positive_relation_effective_weight": fitted.effective_loss_weights[
                    "positive_relation"
                ],
                **{
                    f"backbone_{name}_effective_weight": weight
                    for name, weight in context.backbone_effective_weights.items()
                },
            }
        )
        history = fitted.history.copy()
        history.insert(0, "outer_fold", outer_fold)
        history.insert(0, "repeat", repeat)
        history.insert(2, "candidate", row.candidate)
        history.insert(
            3,
            "residual_anchor_weight_selected",
            float(row.residual_anchor_weight),
        )
        histories.append(history)
        if row.candidate == "learned_beta":
            domain_values = score_domain(context, fitted, row_indices=score_rows)

    if domain_values is None:
        raise RuntimeError("learned_beta domain block was not scored")

    return EvaluationResult(
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(coefficients),
        pd.concat(histories, ignore_index=True),
        domain_rows,
        domain_values,
        context.relations.domain.entity_a,
        context.relations.domain.entity_b,
    )


def combine_results(results: list[EvaluationResult]) -> EvaluationResult:
    """Combine evaluation results from outer folds."""
    return EvaluationResult(
        pd.concat([result.predictions for result in results], ignore_index=True),
        pd.concat([result.metrics for result in results], ignore_index=True),
        pd.concat([result.coefficients for result in results], ignore_index=True),
        pd.concat([result.training_history for result in results], ignore_index=True),
        pd.concat([result.domain_rows for result in results], ignore_index=True),
        np.concatenate([result.domain_values for result in results], axis=0),
        results[0].domain_all_rows,
        results[0].domain_columns,
    )


def save_result(result: EvaluationResult, output_dir: str | Path) -> None:
    """Save the minimal outer-evaluation audit files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output / "outer_test_predictions.csv", index=False)
    result.metrics.to_csv(output / "outer_fold_metrics.csv", index=False)
    result.coefficients.to_csv(output / "outer_coefficients.csv", index=False)
    result.training_history.to_csv(output / "outer_training_history.csv", index=False)
    np.savez_compressed(
        output / "learned_beta_outer_domain_preZ.npz",
        repeat=result.domain_rows["repeat"].to_numpy(np.int16),
        outer_fold=result.domain_rows["outer_fold"].to_numpy(np.int8),
        entity_a=result.domain_rows["entity_a"].to_numpy(str),
        entity_b=np.asarray(result.domain_columns, dtype=str),
        domain_entity_a=np.asarray(result.domain_all_rows, dtype=str),
        values=result.domain_values.astype(np.float32),
    )


def main() -> None:
    """Run one fold, all outer folds, or a smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    parser.add_argument("--splits-dir", required=True, help="Directory produced by splits.py.")
    parser.add_argument("--selection-file", help="selected_hyperparameters.csv produced by selection.py.")
    parser.add_argument("--output-dir", help="Directory for formal evaluation CSV files.")
    parser.add_argument(
        "--context-checkpoint",
        help="Shared frozen-backbone checkpoint created by the selection stage.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device.")
    parser.add_argument("--repeat", type=int, help="One outer repeat to evaluate.")
    parser.add_argument("--outer-fold", type=int, help="One outer fold to evaluate.")
    parser.add_argument("--all-outer-folds", action="store_true", help="Explicitly evaluate every selected outer fold.")
    parser.add_argument(
        "--dev-head-epochs",
        type=int,
        help="DEV only: override head epochs to match the small-range selection run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Evaluate both candidates with anchor weight 0.1 and one epoch.",
    )
    args = parser.parse_args()

    if (args.repeat is None) != (args.outer_fold is None):
        parser.error("--repeat and --outer-fold must be supplied together")
    if not args.smoke_test and args.selection_file is None:
        parser.error("--selection-file is required for formal evaluation")
    if not args.smoke_test and args.output_dir is None:
        parser.error("--output-dir is required for formal evaluation")
    if not args.smoke_test and not args.all_outer_folds and args.repeat is None:
        parser.error("select one outer fold or pass --all-outer-folds")
    if args.dev_head_epochs is not None and args.dev_head_epochs <= 0:
        parser.error("--dev-head-epochs must be positive")

    context = prepare_context(
        args.config,
        args.domain,
        args.device,
        backbone_checkpoint=args.context_checkpoint,
        backbone_epochs=1 if args.smoke_test else None,
        steps_per_epoch=2 if args.smoke_test else None,
        feature_sample_size=4096 if args.smoke_test else None,
        anchor_sample_size=4096 if args.smoke_test else None,
    )
    splits = load_materialized_splits(args.splits_dir)
    audit_splits(splits, context)
    domain_assignments = load_domain_assignments(args.splits_dir)
    expected_entities = set(context.relations.domain.entity_a)
    for repeat, frame in domain_assignments.groupby("repeat", sort=True):
        if set(frame["entity_a"]) != expected_entities or frame["entity_a"].duplicated().any():
            raise ValueError(f"repeat {repeat}: domain assignments must cover every entity once")
    if args.smoke_test:
        keys = [(args.repeat, args.outer_fold)] if args.repeat is not None else [splits.outer_keys()[0]]
    elif args.all_outer_folds:
        available = pd.read_csv(args.selection_file)[["repeat", "outer_fold"]].drop_duplicates()
        keys = [tuple(map(int, value)) for value in available.itertuples(index=False, name=None)]
    else:
        keys = [(args.repeat, args.outer_fold)]

    results = []
    for repeat, outer_fold in keys:
        if args.smoke_test:
            selected = pd.DataFrame(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "candidate": list(COEFFICIENT_CANDIDATES),
                    "residual_anchor_weight": 0.1,
                    "context_fingerprint": context_fingerprint(context),
                }
            )
        else:
            selected = selected_hyperparameters(args.selection_file, repeat, outer_fold)
        audit_selected_context(selected, context)
        result = evaluate_outer_fold(
            context,
            splits,
            domain_assignments,
            selected,
            repeat,
            outer_fold,
            epochs=1 if args.smoke_test else args.dev_head_epochs,
        )
        results.append(result)
        for row in result.metrics.itertuples(index=False):
            print(
                f"PASS  repeat {repeat}, outer fold {outer_fold}, {row.candidate}: "
                f"anchor={row.residual_anchor_weight:.8g}, "
                f"beta={row.primary_coefficient:.6f}, "
                f"log_loss={row.probability_log_loss:.8g}, ranking_AUPRC={row.ranking_pooled_auprc:.8g}"
            )

    combined = combine_results(results)
    if args.smoke_test:
        if len(combined.metrics) != 2 * len(keys) or not np.isfinite(
            combined.predictions[["probability", "residual", "final_score"]].to_numpy()
        ).all():
            raise RuntimeError("Smoke evaluation produced incomplete or non-finite results")
        print(f"PASS  evaluation smoke test: {len(combined.metrics)} candidate outer-test evaluations")
    else:
        save_result(combined, args.output_dir)
        print(f"PASS  evaluation files: {Path(args.output_dir)}")
    print("PASS  evaluation module")


if __name__ == "__main__":
    main()
