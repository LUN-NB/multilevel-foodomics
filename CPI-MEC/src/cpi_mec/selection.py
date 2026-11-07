"""Select regularization inside fixed nested grouped cross-validation splits.

For each outer-training set, both coefficient candidates are tuned separately
on the same inner folds. The formal objective is the equal-weight global and
within-CID skill score, selected by its exact maximum. Outer-test labels are
never read by the selection loop.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from training import (
    TrainingContext,
    context_fingerprint,
    fit_candidate,
    predict_candidate,
    prepare_context,
)


ANCHOR_WEIGHT_PATH = np.logspace(-4, 4, 100)
COEFFICIENT_CANDIDATES = {
    "fixed_beta_1": False,
    "learned_beta": True,
}


@dataclass(frozen=True)
class MaterializedSplits:
    """Store materialized outer and inner folds."""

    outer: pd.DataFrame
    inner: pd.DataFrame

    def outer_keys(self) -> list[tuple[int, int]]:
        """Return all outer repeat-fold keys."""
        keys = self.inner[["repeat", "outer_fold"]].drop_duplicates().sort_values(
            ["repeat", "outer_fold"]
        )
        return [tuple(map(int, row)) for row in keys.itertuples(index=False, name=None)]

    def inner_folds(self, repeat: int, outer_fold: int) -> list[int]:
        """Return inner folds for one outer training set."""
        selected = self.inner[
            (self.inner["repeat"] == repeat) & (self.inner["outer_fold"] == outer_fold)
        ]
        return sorted(selected["inner_fold"].astype(int).unique().tolist())

    def inner_validation_rows(self, repeat: int, outer_fold: int, inner_fold: int) -> np.ndarray:
        """Return validation rows for one inner fold."""
        selected = self.inner[
            (self.inner["repeat"] == repeat)
            & (self.inner["outer_fold"] == outer_fold)
            & (self.inner["inner_fold"] == inner_fold)
        ]
        return selected["relation_row"].to_numpy(np.int64)

    def inner_train_rows(self, repeat: int, outer_fold: int, inner_fold: int) -> np.ndarray:
        """Return training rows for one inner fold."""
        selected = self.inner[
            (self.inner["repeat"] == repeat) & (self.inner["outer_fold"] == outer_fold)
        ]
        outer_train = selected["relation_row"].drop_duplicates().to_numpy(np.int64)
        validation = self.inner_validation_rows(repeat, outer_fold, inner_fold)
        return np.setdiff1d(outer_train, validation)


@dataclass(frozen=True)
class SelectionResult:
    """Store tuning results for outer folds."""

    fold_scores: pd.DataFrame
    candidate_summary: pd.DataFrame
    selected: pd.DataFrame


def load_materialized_splits(split_dir: str | Path) -> MaterializedSplits:
    """Read fixed folds produced by splits.py."""
    directory = Path(split_dir)
    outer = pd.read_csv(directory / "outer_assignments.csv")
    inner = pd.read_csv(directory / "inner_assignments.csv")
    return MaterializedSplits(outer, inner)


def audit_splits(splits: MaterializedSplits, context: TrainingContext) -> None:
    """Match folds to training rows and audit leakage."""
    identity = context.labels[["relation_row", "entity_a", "entity_b", "label"]].copy()
    for frame, name in ((splits.outer, "outer"), (splits.inner, "inner")):
        merged = frame.merge(identity, on="relation_row", suffixes=("_split", "_data"), validate="many_to_one")
        for column in ("entity_a", "entity_b", "label"):
            if not merged[f"{column}_split"].astype(str).equals(merged[f"{column}_data"].astype(str)):
                raise ValueError(f"{name} assignments do not match binary relation column {column}")

    all_rows = set(identity["relation_row"].astype(int))
    for repeat, outer_fold in splits.outer_keys():
        inner = splits.inner[
            (splits.inner["repeat"] == repeat) & (splits.inner["outer_fold"] == outer_fold)
        ]
        if inner.groupby("relation_row")["inner_fold"].nunique().max() != 1:
            raise ValueError(f"repeat {repeat}, outer fold {outer_fold}: a row crosses inner folds")
        outer_test = set(
            splits.outer.loc[
                (splits.outer["repeat"] == repeat) & (splits.outer["outer_fold"] == outer_fold),
                "relation_row",
            ].astype(int)
        )
        outer_train = set(inner["relation_row"].astype(int))
        if outer_test & outer_train or outer_test | outer_train != all_rows:
            raise ValueError(f"repeat {repeat}, outer fold {outer_fold}: outer/inner row leakage")


def binary_log_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    """Compute row-unweighted binary Log-loss."""
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def cid_equal_pairwise_logistic_loss(
    frame: pd.DataFrame,
    temperature: float,
) -> float:
    """Compute pairwise-within-CID and equal-across-CID logistic loss."""
    losses = []
    for _, part in frame.groupby("entity_a", sort=False):
        positive = part.loc[
            part["label"] == 1,
            "ranking_score",
        ].to_numpy(np.float64)
        negative = part.loc[
            part["label"] == 0,
            "ranking_score",
        ].to_numpy(np.float64)
        if len(positive) and len(negative):
            margin = (positive[:, None] - negative[None, :]) / temperature
            losses.append(float(np.logaddexp(0.0, -margin).mean()))
    return float(np.mean(losses))


def pooled_selection_metrics(
    frame: pd.DataFrame,
    temperature: float,
) -> pd.Series:
    """Compute pooled-OOF G, W, and J."""
    prevalence = float(frame["label"].mean())
    auprc = float(
        average_precision_score(frame["label"], frame["ranking_score"])
    )
    pairwise = cid_equal_pairwise_logistic_loss(frame, temperature)
    global_skill = (auprc - prevalence) / (1.0 - prevalence)
    within_skill = 1.0 - pairwise / np.log(2.0)
    return pd.Series(
        {
            "pooled_oof_pairwise_logistic_loss": pairwise,
            "pooled_oof_auprc": auprc,
            "pooled_oof_positive_prevalence": prevalence,
            "pooled_oof_global_skill": global_skill,
            "pooled_oof_within_skill": within_skill,
            "pooled_oof_selection_score_j": (
                0.5 * global_skill + 0.5 * within_skill
            ),
        }
    )


def summarize_candidates(
    fold_scores: pd.DataFrame,
    objective: str,
    rule: str,
) -> pd.DataFrame:
    """Summarize inner metrics and apply the declared rule."""
    keys = [
        "repeat",
        "outer_fold",
        "candidate",
        "coefficient_learnable",
        "residual_anchor_weight",
        "context_fingerprint",
    ]
    summary = (
        fold_scores.groupby(keys, sort=False)
        .agg(
            n_inner_folds=("inner_fold", "nunique"),
            mean_pairwise_logistic_loss=(
                "pairwise_logistic_loss",
                "mean",
            ),
            sd_pairwise_logistic_loss=(
                "pairwise_logistic_loss",
                "std",
            ),
            pooled_oof_pairwise_logistic_loss=(
                "pooled_oof_pairwise_logistic_loss",
                "first",
            ),
            mean_log_loss=("log_loss", "mean"),
            sd_log_loss=("log_loss", "std"),
            mean_pooled_auprc=("pooled_auprc", "mean"),
            sd_pooled_auprc=("pooled_auprc", "std"),
            pooled_oof_auprc=("pooled_oof_auprc", "first"),
            pooled_oof_positive_prevalence=(
                "pooled_oof_positive_prevalence",
                "first",
            ),
            pooled_oof_global_skill=("pooled_oof_global_skill", "first"),
            pooled_oof_within_skill=("pooled_oof_within_skill", "first"),
            pooled_oof_selection_score_j=(
                "pooled_oof_selection_score_j",
                "first",
            ),
            mean_pointwise_effective_weight=("pointwise_effective_weight", "mean"),
            mean_positive_relation_effective_weight=(
                "positive_relation_effective_weight",
                "mean",
            ),
        )
        .reset_index()
    )
    summary["objective_best"] = False
    summary["selected"] = False
    summary["selection_objective"] = objective
    summary["selection_rule"] = rule

    family_keys = ["repeat", "outer_fold", "candidate"]
    for _, index in summary.groupby(family_keys, sort=False).groups.items():
        family = summary.loc[index]
        if objective == "global_within_equal_skill":
            best_index = family.sort_values(
                [
                    "pooled_oof_selection_score_j",
                    "residual_anchor_weight",
                ],
                ascending=[False, True],
                kind="stable",
            ).index[0]
        elif objective == "pairwise_logistic_loss":
            best_index = family.sort_values(
                [
                    "pooled_oof_pairwise_logistic_loss",
                    "mean_pairwise_logistic_loss",
                    "pooled_oof_auprc",
                    "residual_anchor_weight",
                ],
                ascending=[True, True, False, True],
                kind="stable",
            ).index[0]
        elif objective == "binary_log_loss":
            best_index = family["mean_log_loss"].idxmin()
        elif objective == "ranking_pooled_auprc":
            best_index = family.sort_values(
                [
                    "pooled_oof_auprc",
                    "mean_log_loss",
                    "residual_anchor_weight",
                ],
                ascending=[False, True, True],
                kind="stable",
            ).index[0]
        else:
            raise ValueError(f"Unsupported selection objective: {objective}")
        if rule != "exact_best":
            raise ValueError("The formal selection rule must be exact_best")
        summary.loc[best_index, "objective_best"] = True
        summary.loc[best_index, "selected"] = True
    return summary.sort_values(
        ["repeat", "outer_fold", "candidate", "residual_anchor_weight"]
    ).reset_index(drop=True)


def select_outer_fold(
    context: TrainingContext,
    splits: MaterializedSplits,
    repeat: int,
    outer_fold: int,
    anchor_weight_path: np.ndarray = ANCHOR_WEIGHT_PATH,
    epochs: int | None = None,
    show_progress: bool = False,
) -> SelectionResult:
    """Tune both candidates inside one outer-training set."""
    inner_folds = splits.inner_folds(repeat, outer_fold)
    anchor_weights = np.asarray(anchor_weight_path, dtype=float)
    total_fits = len(COEFFICIENT_CANDIDATES) * len(anchor_weights) * len(inner_folds)
    completed_fits = 0
    started = time.monotonic()
    selection = context.config.get("selection", {})
    objective = str(
        selection.get("objective", "global_within_equal_skill")
    )
    rule = str(selection.get("rule", "exact_best"))
    fingerprint = context_fingerprint(context)
    if show_progress:
        head_epochs = context.settings.head_epochs if epochs is None else epochs
        print(
            f"PLAN  repeat {repeat}, outer fold {outer_fold}: "
            f"{len(COEFFICIENT_CANDIDATES)} candidates x {len(anchor_weights)} anchor weights x "
            f"{len(inner_folds)} inner folds = {total_fits} fits; "
            f"{head_epochs} head epochs per fit",
            flush=True,
        )
    records: list[dict[str, float | int | bool | str]] = []
    prediction_records: list[pd.DataFrame] = []
    for candidate, learnable in COEFFICIENT_CANDIDATES.items():
        if show_progress:
            print(f"STAGE candidate {candidate} started", flush=True)
        for anchor_index, anchor_weight in enumerate(anchor_weights, start=1):
            for inner_fold in inner_folds:
                train_rows = splits.inner_train_rows(repeat, outer_fold, inner_fold)
                validation_rows = splits.inner_validation_rows(repeat, outer_fold, inner_fold)
                fit_seed = context.settings.seed + repeat * 10000 + outer_fold * 100 + inner_fold
                fitted = fit_candidate(
                    context,
                    train_rows,
                    learnable_coefficient=learnable,
                    residual_anchor_weight=anchor_weight,
                    seed=fit_seed,
                    epochs=epochs,
                )
                prediction = predict_candidate(context, fitted, validation_rows)
                log_loss = binary_log_loss(prediction["label"], prediction["probability"])
                ranking_score = (
                    -prediction["final_score"]
                    if context.models.architecture.primary_direction == "lower_is_better"
                    else prediction["final_score"]
                )
                pairwise_logistic_loss = cid_equal_pairwise_logistic_loss(
                    pd.DataFrame(
                        {
                            "entity_a": prediction["entity_a"],
                            "label": prediction["label"],
                            "ranking_score": ranking_score,
                        }
                    ),
                    context.settings.ranking_temperature,
                )
                pooled_auprc = float(
                    average_precision_score(prediction["label"], ranking_score)
                )
                prediction_records.append(
                    pd.DataFrame(
                        {
                            "candidate": candidate,
                            "residual_anchor_weight": float(anchor_weight),
                            "entity_a": prediction["entity_a"].to_numpy(),
                            "label": prediction["label"].to_numpy(np.int8),
                            "ranking_score": np.asarray(ranking_score, dtype=np.float64),
                        }
                    )
                )
                completed_fits += 1
                if show_progress:
                    elapsed = time.monotonic() - started
                    eta = elapsed / completed_fits * (total_fits - completed_fits)
                    print(
                        f"FIT   {completed_fits:04d}/{total_fits} "
                        f"({100.0 * completed_fits / total_fits:5.1f}%) | "
                        f"{candidate} | anchor {anchor_index:03d}/{len(anchor_weights)}="
                        f"{anchor_weight:.6g} | "
                        f"inner {inner_fold} | "
                        f"pairwise_loss={pairwise_logistic_loss:.8g} | "
                        f"elapsed={elapsed / 60:.1f}m | ETA={eta / 60:.1f}m",
                        flush=True,
                    )
                records.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "candidate": candidate,
                        "coefficient_learnable": learnable,
                        "residual_anchor_weight": float(anchor_weight),
                        "n_train": len(train_rows),
                        "n_validation": len(validation_rows),
                        "pairwise_logistic_loss": pairwise_logistic_loss,
                        "log_loss": log_loss,
                        "pooled_auprc": pooled_auprc,
                        "primary_coefficient": fitted.primary_coefficient,
                        "pointwise_effective_weight": fitted.effective_loss_weights[
                            "pointwise"
                        ],
                        "positive_relation_effective_weight": fitted.effective_loss_weights[
                            "positive_relation"
                        ],
                        "context_fingerprint": fingerprint,
                    }
                )
        if show_progress:
            print(f"STAGE candidate {candidate} completed", flush=True)
    fold_scores = pd.DataFrame(records)
    prediction_table = pd.concat(prediction_records, ignore_index=True)
    pooled = (
        prediction_table.groupby(["candidate", "residual_anchor_weight"], sort=False)
        .apply(
            pooled_selection_metrics,
            temperature=context.settings.ranking_temperature,
        )
        .reset_index()
    )
    fold_scores = fold_scores.merge(
        pooled,
        on=["candidate", "residual_anchor_weight"],
        how="left",
        validate="many_to_one",
    )
    summary = summarize_candidates(fold_scores, objective, rule)
    selected = summary.loc[summary["selected"]].reset_index(drop=True)
    return SelectionResult(fold_scores, summary, selected)


def combine_results(results: list[SelectionResult]) -> SelectionResult:
    """Combine results from multiple outer folds."""
    return SelectionResult(
        pd.concat([result.fold_scores for result in results], ignore_index=True),
        pd.concat([result.candidate_summary for result in results], ignore_index=True),
        pd.concat([result.selected for result in results], ignore_index=True),
    )


def save_result(result: SelectionResult, output_dir: str | Path) -> None:
    """Save the minimal tuning audit files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.fold_scores.to_csv(
        output / "inner_fold_selection_metrics.csv",
        index=False,
    )
    result.candidate_summary.to_csv(output / "candidate_summary.csv", index=False)
    result.selected.to_csv(output / "selected_hyperparameters.csv", index=False)


def main() -> None:
    """Run one outer fold, all folds, or a smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--domain", help="Configured domain name; defaults to the first domain.")
    parser.add_argument("--splits-dir", required=True, help="Directory produced by splits.py.")
    parser.add_argument("--output-dir", help="Directory for formal selection CSV files.")
    parser.add_argument(
        "--context-checkpoint",
        help="Shared frozen-backbone checkpoint; created when absent and reused when present.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device.")
    parser.add_argument("--repeat", type=int, help="One outer repeat to tune.")
    parser.add_argument("--outer-fold", type=int, help="One outer fold to tune.")
    parser.add_argument("--all-outer-folds", action="store_true", help="Explicitly tune every outer fold.")
    parser.add_argument(
        "--dev-anchor-count",
        type=int,
        help="DEV only: sample this many anchor weights and include the legacy value 0.1.",
    )
    parser.add_argument(
        "--dev-head-epochs",
        type=int,
        help="DEV only: override head epochs for the small-range regression check.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one anchor weight and one head epoch.",
    )
    args = parser.parse_args()

    if (args.repeat is None) != (args.outer_fold is None):
        parser.error("--repeat and --outer-fold must be supplied together")
    if not args.all_outer_folds and args.repeat is None and not args.smoke_test:
        parser.error("select one outer fold or pass --all-outer-folds")
    if args.output_dir is None and not args.smoke_test:
        parser.error("--output-dir is required for formal selection")
    if args.dev_head_epochs is not None and args.dev_head_epochs <= 0:
        parser.error("--dev-head-epochs must be positive")

    print("CONTEXT loading data and pretraining shared backbone...", flush=True)
    context_started = time.monotonic()
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
    print(
        f"CONTEXT ready: device={context.device}, labels={len(context.labels)}, "
        f"backbone_epochs={len(context.backbone_history)}, "
        f"elapsed={(time.monotonic() - context_started) / 60:.1f}m",
        flush=True,
    )
    splits = load_materialized_splits(args.splits_dir)
    audit_splits(splits, context)
    keys = splits.outer_keys() if args.all_outer_folds else [(args.repeat, args.outer_fold)]
    if args.smoke_test and args.repeat is None:
        keys = [splits.outer_keys()[0]]
    if args.smoke_test:
        path = np.asarray([0.1])
    elif args.dev_anchor_count is not None:
        if not 2 <= args.dev_anchor_count <= len(ANCHOR_WEIGHT_PATH):
            parser.error("--dev-anchor-count must be within 2..100")
        indices = np.linspace(
            0, len(ANCHOR_WEIGHT_PATH) - 1, args.dev_anchor_count
        ).round().astype(int)
        path = np.unique(np.append(ANCHOR_WEIGHT_PATH[indices], 0.1))
    else:
        path = ANCHOR_WEIGHT_PATH
    epochs = 1 if args.smoke_test else args.dev_head_epochs

    results = []
    for repeat, outer_fold in keys:
        result = select_outer_fold(
            context,
            splits,
            repeat,
            outer_fold,
            path,
            epochs,
            show_progress=True,
        )
        results.append(result)
        selected = result.selected[
            [
                "candidate",
                "residual_anchor_weight",
                "mean_log_loss",
                "pooled_oof_auprc",
            ]
        ]
        for row in selected.itertuples(index=False):
            print(
                f"PASS  repeat {repeat}, outer fold {outer_fold}, {row.candidate}: "
                f"anchor={row.residual_anchor_weight:.8g}, "
                f"log_loss={row.mean_log_loss:.8g}, AUPRC={row.pooled_oof_auprc:.8g}"
            )

    combined = combine_results(results)
    if args.smoke_test:
        expected = sum(
            len(splits.inner_folds(repeat, outer_fold)) * len(COEFFICIENT_CANDIDATES)
            for repeat, outer_fold in keys
        )
        if len(combined.fold_scores) != expected or len(combined.selected) != 2 * len(keys):
            raise RuntimeError("Smoke selection produced an unexpected number of rows")
        print(f"PASS  selection smoke test: {len(combined.fold_scores)} inner-fold fits")
    else:
        save_result(combined, args.output_dir)
        print(f"PASS  selection files: {Path(args.output_dir)}")
    print("PASS  selection module")


if __name__ == "__main__":
    main()
