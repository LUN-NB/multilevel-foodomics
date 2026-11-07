#!/usr/bin/env python3
"""Evaluate M0, selected M1, and Final in one formal module.

This module preserves the numerical definitions of the former 03A, 03B, and
03C scripts: protein-wise physicochemical CV-R², primary CPI ranking metrics,
and all non-trajectory supplementary metrics. Formal M1 CPI performance is
read from strict outer-OOF results, while the full M1 matrix is used only for
matrix-level analyses. Final label metrics are descriptive.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


SCRIPT_VERSION = "2.0.0"
FEATURES = ["HeavyAtomCount", "MolLogP", "TPSA", "RotatableBonds"]
GLOBAL_FRACTIONS = [0.01, 0.05, 0.10, 0.30, 0.50]
CLASSIFICATION_FRACTIONS = [0.10, 0.30, 0.50]
WITHIN_FRACTIONS = [0.01, 0.05, 0.10, 0.30]
SUCCESS_K = [1, 5, 15, 100]
BEDROC_ALPHA = 20.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse unified evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="Method name and preZ matrix. Repeat for M0, M1 and Final.",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--m1-oof-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--higher-better",
        action="append",
        default=[],
        metavar="METHOD",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--global-top-fraction", type=float, default=0.05)
    parser.add_argument("--target-top-k", type=int, default=100)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser.parse_args(argv)


def parse_matrices(items: list[str]) -> dict[str, Path]:
    """Split METHOD=PATH arguments."""
    matrices: dict[str, Path] = {}
    for item in items:
        method, path = item.split("=", maxsplit=1)
        matrices[method] = Path(path).resolve()
    return matrices


def common_matrix_domain(
    matrices: dict[str, Path],
) -> tuple[list[str], list[str]]:
    """Get the common domain in first-matrix order."""
    cid_lists = [
        pd.read_csv(path, usecols=["cid"], dtype={"cid": str})["cid"].tolist()
        for path in matrices.values()
    ]
    gene_lists = [
        [
            str(column)
            for column in pd.read_csv(path, nrows=0).columns
            if column != "cid"
        ]
        for path in matrices.values()
    ]
    shared_cids = set.intersection(*(set(values) for values in cid_lists))
    shared_genes = set.intersection(*(set(values) for values in gene_lists))
    return (
        [cid for cid in cid_lists[0] if cid in shared_cids],
        [gene for gene in gene_lists[0] if gene in shared_genes],
    )


def load_matrix(
    path: Path,
    common_cids: list[str],
    common_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load common domain and impute row means."""
    matrix = pd.read_csv(path)
    matrix["cid"] = matrix["cid"].astype(str)
    matrix = matrix.set_index("cid").loc[common_cids, common_genes]
    cids = matrix.index.to_numpy(dtype=str)
    genes = matrix.columns.to_numpy(dtype=str)
    scores = matrix.to_numpy(dtype=np.float64)
    missing = np.isnan(scores)
    row_means = np.nanmean(scores, axis=1)
    missing_rows, missing_columns = np.where(missing)
    scores[missing_rows, missing_columns] = row_means[missing_rows]
    return cids, genes, scores, missing


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    """Generate stable one-based ascending ranks."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


def map_labels(
    labels: pd.DataFrame,
    cids: np.ndarray,
    genes: np.ndarray,
) -> pd.DataFrame:
    """Map labels to matrix coordinates."""
    cid_index = pd.Series(np.arange(len(cids)), index=cids)
    gene_index = pd.Series(np.arange(len(genes)), index=genes)
    mapped = labels.copy()
    mapped["row_index"] = mapped["cid"].map(cid_index)
    mapped["column_index"] = mapped["gene"].map(gene_index)
    return mapped


# ---------------------------------------------------------------------------
# 03A Protein-wise physicochemical bias
# ---------------------------------------------------------------------------


def make_group_folds(
    groups: np.ndarray,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build approximately balanced grouped folds."""
    group_names, group_counts = np.unique(groups, return_counts=True)
    shuffled = rng.permutation(len(group_names))
    order = shuffled[np.argsort(-group_counts[shuffled], kind="stable")]
    fold_sizes = np.zeros(n_folds, dtype=int)
    group_to_fold: dict[str, int] = {}
    for index in order:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[group_names[index]] = fold
        fold_sizes[fold] += group_counts[index]
    return np.fromiter((group_to_fold[group] for group in groups), dtype=int)


def build_fold_table(
    cids: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Generate all repeated folds once."""
    records = []
    for repeat in range(1, n_repeats + 1):
        folds = make_group_folds(
            groups,
            n_folds,
            np.random.default_rng(seed + repeat - 1),
        )
        records.append(
            pd.DataFrame(
                {
                    "repeat": repeat,
                    "cid": cids,
                    "group_id": groups,
                    "fold": folds + 1,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def standardize_train_apply(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training-fold parameters."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def multioutput_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """Generate OOF predictions for all proteins."""
    predictions = np.empty_like(y, dtype=np.float64)
    for fold in range(1, n_folds + 1):
        test = folds == fold
        train = ~test
        x_train, x_test = standardize_train_apply(x[train], x[test])
        x_train = np.column_stack((np.ones(train.sum()), x_train))
        x_test = np.column_stack((np.ones(test.sum()), x_test))
        coefficients = np.linalg.lstsq(x_train, y[train], rcond=None)[0]
        predictions[test] = x_test @ coefficients
    return predictions


def r2_by_gene(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Compute gene-wise OOF R², retaining negatives."""
    residual_ss = np.square(observed - predicted).sum(axis=0)
    total_ss = np.square(observed - observed.mean(axis=0)).sum(axis=0)
    return 1.0 - residual_ss / total_ss


def load_physchem_matrix(
    path: Path,
    descriptors: pd.DataFrame,
    common_cids: list[str],
    common_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Align physicochemical input and impute rows."""
    matrix = pd.read_csv(path)
    matrix["cid"] = matrix["cid"].astype(str)
    matrix = matrix.set_index("cid").loc[common_cids, common_genes].reset_index()
    aligned = descriptors.merge(matrix, on="cid", how="inner", validate="one_to_one")
    x = aligned[FEATURES].to_numpy(dtype=np.float64)
    y = aligned[common_genes].to_numpy(dtype=np.float64)
    missing = np.isnan(y)
    row_means = np.nanmean(y, axis=1)
    missing_rows, missing_columns = np.where(missing)
    y[missing_rows, missing_columns] = row_means[missing_rows]
    return (
        aligned["cid"].to_numpy(),
        aligned["group_id"].astype(str).to_numpy(),
        common_genes,
        x,
        y,
        missing,
    )


def evaluate_physchem_method(
    method: str,
    matrix_path: Path,
    descriptors: pd.DataFrame,
    fold_table: pd.DataFrame,
    n_folds: int,
    common_cids: list[str],
    common_genes: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate protein-wise physicochemical bias."""
    cids, _, genes, x, y, missing = load_physchem_matrix(
        matrix_path,
        descriptors,
        common_cids,
        common_genes,
    )
    records = []
    for repeat, repeat_folds in fold_table.groupby("repeat", sort=True):
        folds = repeat_folds.set_index("cid").loc[cids, "fold"].to_numpy(dtype=int)
        predicted = multioutput_oof_predictions(x, y, folds, n_folds)
        records.append(
            pd.DataFrame(
                {
                    "method": method,
                    "repeat": repeat,
                    "gene": genes,
                    "cv_r2": r2_by_gene(y, predicted),
                }
            )
        )
    audit = {
        "method": method,
        "n_cid": int(len(cids)),
        "n_gene": int(len(genes)),
        "n_pairs": int(y.size),
        "n_missing": int(missing.sum()),
        "missing_fraction": float(missing.mean()),
        "missing_value_policy": "compound_row_mean",
    }
    return pd.concat(records, ignore_index=True), audit


def summarize_physchem(
    gene_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize physicochemical bias."""
    repeat_summary = (
        gene_metrics.groupby(["method", "repeat"], as_index=False)["cv_r2"]
        .agg(
            median_protein_cv_r2="median",
            q25_protein_cv_r2=lambda values: values.quantile(0.25),
            q75_protein_cv_r2=lambda values: values.quantile(0.75),
            p90_protein_cv_r2=lambda values: values.quantile(0.90),
        )
    )
    method_summary = (
        repeat_summary.groupby("method", as_index=False)
        .agg(
            median_protein_cv_r2=("median_protein_cv_r2", "median"),
            repeat_q25=("median_protein_cv_r2", lambda values: values.quantile(0.25)),
            repeat_q75=("median_protein_cv_r2", lambda values: values.quantile(0.75)),
            mean_protein_q25=("q25_protein_cv_r2", "mean"),
            mean_protein_q75=("q75_protein_cv_r2", "mean"),
            mean_protein_p90=("p90_protein_cv_r2", "mean"),
        )
        .sort_values("median_protein_cv_r2")
    )
    return repeat_summary, method_summary


# ---------------------------------------------------------------------------
# 03B Primary CPI ranking metrics
# ---------------------------------------------------------------------------


def global_top_mask(scores: np.ndarray, top_fraction: float) -> np.ndarray:
    """Find the full-matrix top region."""
    flat = scores.ravel()
    top_n = int(np.ceil(flat.size * top_fraction))
    top_indices = np.argpartition(flat, top_n - 1)[:top_n]
    mask = np.zeros(flat.size, dtype=bool)
    mask[top_indices] = True
    return mask


def per_cid_metrics(
    mapped: pd.DataFrame,
    scores: np.ndarray,
    missing: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Compute within-CID AUROC and Top-k retrieval."""
    records = []
    for cid, group in mapped.groupby("cid", sort=False):
        row = int(group["row_index"].iloc[0])
        labels = group["Y"].to_numpy(dtype=int)
        columns = group["column_index"].to_numpy(dtype=int)
        labelled_scores = -scores[row, columns]
        within_auc = (
            roc_auc_score(labels, labelled_scores)
            if np.unique(labels).size == 2
            else np.nan
        )
        positive_columns = group.loc[
            group["Y"] == 1,
            "column_index",
        ].to_numpy(dtype=int)
        if positive_columns.size:
            top_columns = np.argpartition(scores[row], top_k - 1)[:top_k]
            success = bool(np.isin(positive_columns, top_columns).any())
            ranks = ordinal_ranks(scores[row])
            best_positive_rank = int(ranks[positive_columns].min())
        else:
            success = None
            best_positive_rank = None
        records.append(
            {
                "cid": cid,
                "n_positive": int((labels == 1).sum()),
                "n_negative": int((labels == 0).sum()),
                "n_matrix_missing": int(missing[row].sum()),
                "matrix_missing_fraction": float(missing[row].mean()),
                "within_compound_auroc": within_auc,
                f"success_top{top_k}": success,
                "best_positive_rank": best_positive_rank,
            }
        )
    return pd.DataFrame(records)


def evaluate_cpi_method(
    method: str,
    matrix_path: Path,
    labels: pd.DataFrame,
    top_fraction: float,
    top_k: int,
    common_cids: list[str],
    common_genes: list[str],
    higher_better: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Compute primary CPI metrics for one matrix."""
    cids, genes, raw_scores, missing = load_matrix(
        matrix_path,
        common_cids,
        common_genes,
    )
    scores = -raw_scores if higher_better else raw_scores
    mapped = map_labels(labels, cids, genes)
    rows = mapped["row_index"].to_numpy(dtype=int)
    columns = mapped["column_index"].to_numpy(dtype=int)
    y_true = mapped["Y"].to_numpy(dtype=int)
    prediction = -scores[rows, columns]
    pooled_auprc = average_precision_score(y_true, prediction)
    prevalence = float(y_true.mean())
    pairwise_losses = []
    for _, part in mapped.assign(
        prediction_score=prediction
    ).groupby("cid", sort=False):
        positive = part.loc[
            part["Y"] == 1,
            "prediction_score",
        ].to_numpy(np.float64)
        negative = part.loc[
            part["Y"] == 0,
            "prediction_score",
        ].to_numpy(np.float64)
        if len(positive) and len(negative):
            pairwise_losses.append(
                float(
                    np.logaddexp(
                        0.0,
                        -(positive[:, None] - negative[None, :]),
                    ).mean()
                )
            )
    pairwise_loss = float(np.mean(pairwise_losses))
    global_skill = (pooled_auprc - prevalence) / (1.0 - prevalence)
    within_skill = 1.0 - pairwise_loss / np.log(2.0)
    top_mask = global_top_mask(scores, top_fraction)
    labelled_flat = np.ravel_multi_index((rows, columns), scores.shape)
    positive_in_top = int(top_mask[labelled_flat][y_true == 1].sum())
    n_positive = int((y_true == 1).sum())
    recall_top = positive_in_top / n_positive
    cid_metrics = per_cid_metrics(mapped, scores, missing, top_k)
    auc_values = cid_metrics["within_compound_auroc"].dropna()
    success_values = cid_metrics.loc[
        cid_metrics["n_positive"] > 0,
        f"success_top{top_k}",
    ].astype(bool)
    summary = {
        "method": method,
        "score_direction": "higher_is_better" if higher_better else "lower_is_better",
        "pooled_auprc": pooled_auprc,
        "positive_prevalence": prevalence,
        "auprc_lift": pooled_auprc / prevalence,
        "pairwise_logistic_loss": pairwise_loss,
        "global_skill": global_skill,
        "within_skill": within_skill,
        "selection_score_j": (
            0.5 * global_skill + 0.5 * within_skill
        ),
        f"global_recall_top{top_fraction:g}": recall_top,
        f"global_ef_top{top_fraction:g}": recall_top / top_fraction,
        "positive_in_global_top": positive_in_top,
        "n_positive": n_positive,
        "macro_within_compound_auroc": float(auc_values.mean()),
        "median_within_compound_auroc": float(auc_values.median()),
        "n_cid_with_both_classes": int(len(auc_values)),
        f"success_top{top_k}": float(success_values.mean()),
        f"n_cid_success_top{top_k}": int(success_values.sum()),
        "n_cid_with_positive": int(len(success_values)),
        "n_matrix_cid": int(len(cids)),
        "n_matrix_gene": int(len(genes)),
        "n_matrix_pairs": int(scores.size),
        "n_matrix_missing": int(missing.sum()),
        "matrix_missing_fraction": float(missing.mean()),
        "n_labelled_missing": int(missing[rows, columns].sum()),
        "missing_value_policy": "compound_row_mean",
        "n_labels": int(len(mapped)),
    }
    labelled_detail = mapped[["cid", "gene", "Y"]].copy()
    labelled_detail.insert(0, "method", method)
    labelled_missing = missing[rows, columns]
    labelled_detail["raw_score"] = np.where(
        labelled_missing,
        np.nan,
        raw_scores[rows, columns],
    )
    labelled_detail["score_was_missing"] = labelled_missing
    labelled_detail["ranking_score_used"] = scores[rows, columns]
    labelled_detail["prediction_score"] = prediction
    labelled_detail[f"in_global_top{top_fraction:g}"] = top_mask[labelled_flat]
    cid_metrics.insert(0, "method", method)
    return summary, labelled_detail, cid_metrics


# ---------------------------------------------------------------------------
# 03C Supplementary metrics
# ---------------------------------------------------------------------------


def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient."""
    ordered = np.sort(values.astype(np.float64))
    n = ordered.size
    return float(
        (2 * np.dot(np.arange(1, n + 1), ordered) / ordered.sum() - n - 1) / n
    )


def grouped_oof_r2(
    x: np.ndarray,
    y: np.ndarray,
    cids: np.ndarray,
    fold_assignments: pd.DataFrame,
) -> np.ndarray:
    """Compute repeated grouped OOF R²."""
    results = []
    for _, fold_table in fold_assignments.groupby("repeat", sort=True):
        folds = fold_table.set_index("cid").loc[cids, "fold"].to_numpy(dtype=int)
        predicted = np.empty(len(y), dtype=np.float64)
        for fold in np.sort(np.unique(folds)):
            test = folds == fold
            train = ~test
            x_train, x_test = standardize_train_apply(x[train], x[test])
            x_train = np.column_stack((np.ones(train.sum()), x_train))
            x_test = np.column_stack((np.ones(test.sum()), x_test))
            coefficients = np.linalg.lstsq(x_train, y[train], rcond=None)[0]
            predicted[test] = x_test @ coefficients
        residual_ss = np.square(y - predicted).sum()
        total_ss = np.square(y - y.mean()).sum()
        results.append(1.0 - residual_ss / total_ss)
    return np.asarray(results)


def bedroc(y_true: np.ndarray, prediction: np.ndarray) -> float:
    """Compute BEDROC over explicit labels."""
    ordered = np.argsort(-prediction, kind="stable")
    ranked = [[float(prediction[index]), int(y_true[index])] for index in ordered]
    return float(CalcBEDROC(ranked, 1, BEDROC_ALPHA))


def evaluate_supplementary_method(
    method: str,
    path: Path,
    labels: pd.DataFrame,
    descriptors: pd.DataFrame,
    folds: pd.DataFrame,
    common_cids: list[str],
    common_genes: list[str],
    higher_better: bool,
) -> dict[str, object]:
    """Compute all supplementary metrics."""
    cids, genes, raw_scores, missing = load_matrix(path, common_cids, common_genes)
    scores = -raw_scores if higher_better else raw_scores
    mapped = map_labels(labels, cids, genes)
    rows = mapped["row_index"].to_numpy(dtype=int)
    columns = mapped["column_index"].to_numpy(dtype=int)
    y_true = mapped["Y"].to_numpy(dtype=int)
    flat_scores = scores.ravel()
    global_ranks = ordinal_ranks(flat_scores)
    labelled_flat = np.ravel_multi_index((rows, columns), scores.shape)
    labelled_global_rank = global_ranks[labelled_flat]
    labelled_global_fraction = labelled_global_rank / scores.size
    prediction = -scores[rows, columns]
    n_positive = int((y_true == 1).sum())

    global_record: dict[str, object] = {
        "method": method,
        "pooled_auroc": roc_auc_score(y_true, prediction),
        "pooled_auprc": average_precision_score(y_true, prediction),
        "bedroc_alpha20": bedroc(y_true, prediction),
        "positive_global_rank_mean": float(
            labelled_global_fraction[y_true == 1].mean()
        ),
        "positive_global_rank_median": float(
            np.median(labelled_global_fraction[y_true == 1])
        ),
        "positive_global_rank_q25": float(
            np.quantile(labelled_global_fraction[y_true == 1], 0.25)
        ),
        "positive_global_rank_q75": float(
            np.quantile(labelled_global_fraction[y_true == 1], 0.75)
        ),
        "positive_pair_global_reciprocal_rank_mean": float(
            np.mean(1.0 / labelled_global_rank[y_true == 1])
        ),
    }
    for fraction in GLOBAL_FRACTIONS:
        predicted_top = global_ranks <= int(np.ceil(scores.size * fraction))
        positive_in_top = int(predicted_top[labelled_flat][y_true == 1].sum())
        recall = positive_in_top / n_positive
        global_record[f"recall_top{fraction:g}"] = recall
        global_record[f"ef_top{fraction:g}"] = recall / fraction
    for fraction in CLASSIFICATION_FRACTIONS:
        predicted_label = (
            labelled_global_rank <= int(np.ceil(scores.size * fraction))
        ).astype(int)
        global_record[f"precision_top{fraction:g}"] = precision_score(
            y_true,
            predicted_label,
        )
        global_record[f"f1_top{fraction:g}"] = f1_score(y_true, predicted_label)
        global_record[f"mcc_top{fraction:g}"] = matthews_corrcoef(
            y_true,
            predicted_label,
        )

    within_ranks = np.empty(len(mapped), dtype=np.int64)
    cid_records = []
    for cid, group in mapped.groupby("cid", sort=False):
        matrix_row = int(group["row_index"].iloc[0])
        row_ranks = ordinal_ranks(scores[matrix_row])
        indices = group.index.to_numpy()
        group_columns = group["column_index"].to_numpy(dtype=int)
        within_ranks[indices] = row_ranks[group_columns]
        positive_columns = group.loc[
            group["Y"] == 1,
            "column_index",
        ].to_numpy(dtype=int)
        record: dict[str, object] = {"method": method, "cid": cid}
        if positive_columns.size:
            best_rank = int(row_ranks[positive_columns].min())
            record["best_positive_rank"] = best_rank
            for top_k in SUCCESS_K:
                record[f"success_top{top_k}"] = best_rank <= top_k
        cid_records.append(record)
    within_fraction = within_ranks / len(genes)
    within_prediction = 1.0 - within_fraction
    cid_table = pd.DataFrame(cid_records)
    within_record: dict[str, object] = {
        "method": method,
        "pooled_within_auroc": roc_auc_score(y_true, within_prediction),
        "pooled_within_auprc": average_precision_score(y_true, within_prediction),
        "positive_within_rank_mean": float(within_fraction[y_true == 1].mean()),
        "positive_within_rank_median": float(
            np.median(within_fraction[y_true == 1])
        ),
        "positive_within_rank_q25": float(
            np.quantile(within_fraction[y_true == 1], 0.25)
        ),
        "positive_within_rank_q75": float(
            np.quantile(within_fraction[y_true == 1], 0.75)
        ),
    }
    for fraction in WITHIN_FRACTIONS:
        within_record[f"recall_top{fraction:g}"] = float(
            (within_fraction[y_true == 1] <= fraction).mean()
        )
    within_record["target_retrieval_mrr"] = float(
        np.mean(1.0 / cid_table["best_positive_rank"].dropna())
    )
    for top_k in SUCCESS_K:
        within_record[f"success_top{top_k}"] = float(
            cid_table[f"success_top{top_k}"].mean()
        )

    descriptor_table = pd.DataFrame({"cid": cids}).merge(
        descriptors[["cid", "group_id", *FEATURES]],
        on="cid",
        validate="one_to_one",
    )
    x = descriptor_table[FEATURES].to_numpy(dtype=np.float64)
    compound_mean = scores.mean(axis=1)
    bias_records = []
    for feature in FEATURES:
        rho, p_value = spearmanr(descriptor_table[feature], compound_mean)
        bias_records.append(
            {
                "method": method,
                "outcome": "compound_mean_score",
                "feature": feature,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
    cv_records = [
        {
            "method": method,
            "outcome": "compound_mean_score",
            "repeat": repeat,
            "cv_r2": value,
        }
        for repeat, value in enumerate(
            grouped_oof_r2(x, compound_mean, cids, folds),
            start=1,
        )
    ]
    protein_concentration = []
    for fraction in [0.01, 0.05, 0.10, 0.30]:
        top = global_ranks.reshape(scores.shape) <= int(
            np.ceil(scores.size * fraction)
        )
        cid_share = top.mean(axis=1)
        cv_records.extend(
            {
                "method": method,
                "outcome": f"cid_share_global_top{fraction:g}",
                "repeat": repeat,
                "cv_r2": value,
            }
            for repeat, value in enumerate(
                grouped_oof_r2(x, cid_share, cids, folds),
                start=1,
            )
        )
        protein_hits = top.sum(axis=0)
        protein_concentration.append(
            {
                "method": method,
                "top_fraction": fraction,
                "coefficient_of_variation": float(
                    protein_hits.std(ddof=0) / protein_hits.mean()
                ),
                "gini": gini(protein_hits),
                "maximum": int(protein_hits.max()),
                "p95": float(np.quantile(protein_hits, 0.95)),
            }
        )

    labelled_detail = mapped[["cid", "gene", "Y"]].copy()
    labelled_detail.insert(0, "method", method)
    labelled_detail["global_rank"] = labelled_global_rank
    labelled_detail["global_rank_fraction"] = labelled_global_fraction
    labelled_detail["within_rank"] = within_ranks
    labelled_detail["within_rank_fraction"] = within_fraction
    labelled_detail["score_was_missing"] = missing[rows, columns]
    return {
        "global": pd.DataFrame([global_record]),
        "within": pd.DataFrame([within_record]),
        "cid": cid_table,
        "bias_spearman": pd.DataFrame(bias_records),
        "bias_cv_r2": pd.DataFrame(cv_records),
        "protein_concentration": pd.DataFrame(protein_concentration),
        "labelled_ranks": labelled_detail,
        "all_ranks": global_ranks,
    }


def stability_metrics(rank_map: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare downstream matrices against M0."""
    baseline = rank_map["M0"]
    n_pairs = baseline.size
    records = []
    for method, candidate in rank_map.items():
        if method == "M0":
            continue
        shift = np.abs(candidate - baseline)
        record: dict[str, object] = {
            "reference": "M0",
            "method": method,
            "spearman_rank_correlation": float(
                np.corrcoef(baseline, candidate)[0, 1]
            ),
            "mean_absolute_rank_shift": float(shift.mean()),
            "median_absolute_rank_shift": float(np.median(shift)),
            "maximum_rank_shift": int(shift.max()),
        }
        for fraction in [0.01, 0.05, 0.10, 0.30]:
            top_n = int(np.ceil(n_pairs * fraction))
            overlap = ((baseline <= top_n) & (candidate <= top_n)).sum()
            record[f"top{fraction:g}_overlap"] = overlap / top_n
        records.append(record)
    return pd.DataFrame(records)


def build_all_metrics_summary(
    cpi_summary: pd.DataFrame,
    physicochemical_summary: pd.DataFrame,
    supplementary_global: pd.DataFrame,
    supplementary_within: pd.DataFrame,
) -> pd.DataFrame:
    """Build one overview row per method."""
    global_extra = supplementary_global.drop(
        columns=["pooled_auprc"],
    ).rename(
        columns={
            column: f"supp_global_{column}"
            for column in supplementary_global.columns
            if column != "method"
        }
    )
    within_extra = supplementary_within.rename(
        columns={
            column: f"supp_within_{column}"
            for column in supplementary_within.columns
            if column != "method"
        }
    )
    return (
        cpi_summary.merge(physicochemical_summary, on="method", how="left")
        .merge(global_extra, on="method", how="left")
        .merge(within_extra, on="method", how="left")
    )


def apply_m1_oof_summary(
    matrix_summary: pd.DataFrame,
    oof_path: Path,
    top_fraction: float,
    top_k: int,
) -> pd.DataFrame:
    """Apply strict M1 OOF metrics."""
    summary = matrix_summary.copy()
    summary["performance_role"] = summary["method"].map(
        {
            "M0": "unadapted_matrix_baseline",
            "M1": "strict_nested_outer_oof",
            "Final": "full_label_fit_descriptive_only",
        }
    )
    oof = pd.read_csv(oof_path).iloc[0]
    mapping = {
        "pooled_auprc": "ranking_pooled_auprc",
        "positive_prevalence": "positive_prevalence",
        "auprc_lift": "ranking_auprc_lift",
        "pairwise_logistic_loss": "pairwise_logistic_loss",
        "global_skill": "global_skill",
        "within_skill": "within_skill",
        "selection_score_j": "selection_score_j",
        f"global_recall_top{top_fraction:g}": (
            f"global_recall_top{top_fraction:g}"
        ),
        f"global_ef_top{top_fraction:g}": (
            f"global_ef_top{top_fraction:g}"
        ),
        "positive_in_global_top": "positive_in_global_top",
        f"global_recall_top{top_k}": f"global_recall_top{top_k}",
        f"positive_in_global_top{top_k}": (
            f"positive_in_global_top{top_k}"
        ),
        "n_positive": "n_positive",
        "macro_within_compound_auroc": "macro_within_entity_auroc",
        "median_within_compound_auroc": "median_within_entity_auroc",
        "n_cid_with_both_classes": "n_informative_entity",
        f"success_top{top_k}": f"success_top{top_k}",
        f"n_cid_success_top{top_k}": f"n_cid_success_top{top_k}",
        "n_cid_with_positive": "n_cid_with_positive",
        "n_labels": "n_test",
    }
    index = summary.index[summary["method"] == "M1"][0]
    for target, source in mapping.items():
        summary.at[index, target] = oof[source]
        sd_source = f"{source}_sd"
        if sd_source in oof:
            summary.at[index, f"{target}_sd"] = oof[sd_source]
    summary.at[index, "score_direction"] = "higher_is_better"
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run all three evaluations."""
    args = parse_args(argv)
    matrices = parse_matrices(args.matrix)
    common_cids, common_genes = common_matrix_domain(matrices)
    labels = pd.read_csv(
        args.labels,
        dtype={"cid": str, "gene": str},
    )[["cid", "gene", "Y"]]
    n_labels_input = len(labels)
    labels = labels[
        labels["cid"].isin(common_cids)
        & labels["gene"].isin(common_genes)
    ].reset_index(drop=True)
    descriptors = pd.read_csv(
        args.descriptors,
        dtype={"cid": str, "group_id": str},
    )[["cid", "group_id", *FEATURES]]

    reference_path = next(iter(matrices.values()))
    reference_cids, reference_groups, _, _, _, _ = load_physchem_matrix(
        reference_path,
        descriptors,
        common_cids,
        common_genes,
    )
    fold_table = build_fold_table(
        reference_cids,
        reference_groups,
        args.folds,
        args.repeats,
        args.seed,
    )

    physicochemical_results = [
        evaluate_physchem_method(
            method,
            path,
            descriptors,
            fold_table,
            args.folds,
            common_cids,
            common_genes,
        )
        for method, path in matrices.items()
    ]
    gene_metrics = pd.concat(
        (result[0] for result in physicochemical_results),
        ignore_index=True,
    )
    matrix_audit = pd.DataFrame(
        result[1] for result in physicochemical_results
    )
    physicochemical_by_repeat, physicochemical_summary = summarize_physchem(
        gene_metrics
    )

    cpi_results = [
        evaluate_cpi_method(
            method,
            path,
            labels,
            args.global_top_fraction,
            args.target_top_k,
            common_cids,
            common_genes,
            method in args.higher_better,
        )
        for method, path in matrices.items()
    ]
    matrix_cpi_summary = pd.DataFrame(result[0] for result in cpi_results)
    cpi_summary = apply_m1_oof_summary(
        matrix_cpi_summary,
        args.m1_oof_summary,
        args.global_top_fraction,
        args.target_top_k,
    )
    labelled_scores = pd.concat(
        (result[1] for result in cpi_results),
        ignore_index=True,
    )
    labelled_scores["performance_role"] = labelled_scores["method"].map(
        {
            "M0": "unadapted_matrix_baseline",
            "M1": "complete_matrix_descriptive_not_formal_oof",
            "Final": "full_label_fit_descriptive_only",
        }
    )
    within_compound = pd.concat(
        (result[2] for result in cpi_results),
        ignore_index=True,
    )
    within_compound["performance_role"] = within_compound["method"].map(
        {
            "M0": "unadapted_matrix_baseline",
            "M1": "complete_matrix_descriptive_not_formal_oof",
            "Final": "full_label_fit_descriptive_only",
        }
    )

    supplementary_results = [
        evaluate_supplementary_method(
            method,
            path,
            labels,
            descriptors,
            fold_table,
            common_cids,
            common_genes,
            method in args.higher_better,
        )
        for method, path in matrices.items()
    ]
    supplementary_keys = [
        key
        for key in supplementary_results[0]
        if key != "all_ranks"
    ]
    supplementary_outputs = {
        key: pd.concat(
            [result[key] for result in supplementary_results],
            ignore_index=True,
        )
        for key in supplementary_keys
    }
    rank_map = {
        method: result["all_ranks"]
        for method, result in zip(
            matrices,
            supplementary_results,
            strict=True,
        )
    }
    supplementary_outputs["stability"] = stability_metrics(rank_map)
    all_metrics = build_all_metrics_summary(
        cpi_summary,
        physicochemical_summary,
        supplementary_outputs["global"],
        supplementary_outputs["within"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_tables = {
        "proteinwise_cv_r2_by_repeat.csv": gene_metrics,
        "physicochemical_bias_by_repeat.csv": physicochemical_by_repeat,
        "physicochemical_bias_summary.csv": physicochemical_summary,
        "fold_assignments.csv": fold_table,
        "matrix_audit.csv": matrix_audit,
        "cpi_ranking_metrics_summary.csv": cpi_summary,
        "cpi_ranking_metrics_matrix_descriptive.csv": matrix_cpi_summary,
        "labelled_pair_scores.csv": labelled_scores,
        "within_compound_metrics.csv": within_compound,
        "supplementary_global_metrics.csv": supplementary_outputs["global"],
        "supplementary_within_compound_metrics.csv": supplementary_outputs["within"],
        "target_retrieval_by_cid.csv": supplementary_outputs["cid"],
        "physicochemical_spearman.csv": supplementary_outputs["bias_spearman"],
        "physicochemical_composition_cv_r2.csv": supplementary_outputs["bias_cv_r2"],
        "global_top_protein_concentration.csv": supplementary_outputs[
            "protein_concentration"
        ],
        "labelled_pair_rank_details.csv": supplementary_outputs["labelled_ranks"],
        "matrix_stability_vs_M0.csv": supplementary_outputs["stability"],
        "all_metrics_summary.csv": all_metrics,
    }
    for filename, table in output_tables.items():
        table.to_csv(args.output_dir / filename, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module": Path(__file__).name,
        "module_version": SCRIPT_VERSION,
        "matrices": {
            method: str(path)
            for method, path in matrices.items()
        },
        "labels": str(args.labels.resolve()),
        "m1_oof_summary": str(args.m1_oof_summary.resolve()),
        "descriptors": str(args.descriptors.resolve()),
        "features": FEATURES,
        "physicochemical_cv": {
            "folds": args.folds,
            "repeats": args.repeats,
            "seed": args.seed,
            "grouping_variable": "group_id",
            "negative_cv_r2": "retained",
        },
        "global_top_fraction": args.global_top_fraction,
        "target_top_k": args.target_top_k,
        "score_direction": {
            method: (
                "higher_is_better"
                if method in args.higher_better
                else "lower_is_better"
            )
            for method in matrices
        },
        "comparison_domain": "intersection_of_all_matrix_cids_and_proteins",
        "n_common_cid": len(common_cids),
        "n_common_gene": len(common_genes),
        "n_labels_input": n_labels_input,
        "n_labels_in_common_domain": len(labels),
        "missing_value_policy": "compound_row_mean",
        "unlabelled_pairs": "ranked_but_not_treated_as_negative",
        "performance_policy": {
            "M0": "direct unadapted matrix baseline",
            "M1": "strict nested outer-OOF predictions",
            "Final": "descriptive full-label fit; no generalization claim",
        },
        "tie_break": "stable_matrix_row_major_order",
        "bedroc_alpha": BEDROC_ALPHA,
        "excluded": ["cluster_bias", "continuous_axis_bias", "branch_bias"],
        "outputs": list(output_tables),
    }
    (args.output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS  unified matrix metrics: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
