"""Generate the nested-CV M1 matrix and strict outer-OOF predictions.

Each outer fold uses its independently selected exact-best, refits on outer
training labels, and scores the complete domain. Averaging all outer models
creates M1. Formal performance comes only from strict outer-test predictions,
never by reading label scores back from the complete M1 ensemble matrix.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_VERSION = "3.0.0"
SCRIPT_RELEASE = "nested_fold_specific_m1_20260728"
PROTOCOL_VERSION = "3.0.0"
BASE_DEPENDENCY = "adaptive_training.py"
FIXED_DEPENDENCY = "adaptive_fixed_training.py"
PARAMETER_NAMES = (
    "anchor_weight",
    "graphban_weight",
    "boltz_weight",
)
CSV_FLOAT_FORMAT = "%.9g"


@dataclass(frozen=True)
class Parameters:
    """Store one outer fold exact-best."""

    repeat: int
    outer_fold: int
    anchor: float
    graphban: float
    boltz: float

    @property
    def identifier(self) -> str:
        """Build a stable parameter identifier."""
        values = [
            f"{value:.12g}".replace(".", "p")
            for value in (self.anchor, self.graphban, self.boltz)
        ]
        return f"a{values[0]}_g{values[1]}_b{values[2]}"

    def as_dict(self) -> dict[str, float]:
        """Return the parameter mapping."""
        return dict(
            zip(
                PARAMETER_NAMES,
                (self.anchor, self.graphban, self.boltz),
            )
        )


def utc_now() -> str:
    """Return current UTC time."""
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """Compute a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    """Load and register a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_int_values(text: str) -> tuple[int, ...]:
    """Parse ordered non-negative integers."""
    values = tuple(sorted(set(int(value) for value in text.split(","))))
    if not values or values[0] < 0:
        raise ValueError("Seed offsets must be non-empty")
    return values


def load_protocol(config_path: Path) -> tuple[int, int]:
    """Read formal repeat and fold counts."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    settings = config["cross_validation"]
    return int(settings["outer_repeats"]), int(settings["outer_folds"])


def read_selected(path: Path) -> pd.DataFrame:
    """Read fold-specific exact-best settings."""
    return pd.read_csv(path).sort_values(
        ["repeat", "outer_fold"]
    ).reset_index(drop=True)


def parameters_from_row(row: pd.Series | object) -> Parameters:
    """Restore parameters from a row."""
    return Parameters(
        int(row.repeat),
        int(row.outer_fold),
        float(row.anchor_weight),
        float(row.graphban_weight),
        float(row.boltz_weight),
    )


def frame_to_json(frame: pd.DataFrame) -> str:
    """Serialize a small table."""
    return frame.to_json(orient="split", index=False, double_precision=15)


def frame_from_json(text: str) -> pd.DataFrame:
    """Restore a small table."""
    return pd.read_json(io.StringIO(text), orient="split")


def checkpoint_name(
    parameters: Parameters,
    seed_offset: int,
) -> str:
    """Build one model checkpoint name."""
    return (
        f"repeat_{parameters.repeat:02d}_outer_{parameters.outer_fold:02d}"
        f"_seed_{seed_offset:02d}_{parameters.identifier}.npz"
    )


def save_checkpoint(
    path: Path,
    metadata: dict,
    values: np.ndarray,
    predictions: pd.DataFrame,
) -> None:
    """Atomically save one model."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        values=np.asarray(values, dtype=np.float32),
        predictions=np.asarray(frame_to_json(predictions)),
    )
    os.replace(temporary, path)


def load_checkpoint(path: Path, signature: str) -> dict:
    """Load one model checkpoint."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["run_signature"] != signature:
            raise ValueError(f"{path.name} belongs to another run")
        return {
            "metadata": metadata,
            "values": data["values"].astype(np.float32),
            "predictions": frame_from_json(str(data["predictions"])),
        }


def run_signature(
    script_path: Path,
    config_path: Path,
    splits_dir: Path,
    base_path: Path,
    fixed_path: Path,
    args: argparse.Namespace,
    seed_offsets: tuple[int, ...],
) -> tuple[str, dict]:
    """Build the stable run signature."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "hashes": {
            "script": sha256(script_path),
            "config": sha256(config_path),
            "base_dependency": sha256(base_path),
            "fixed_dependency": sha256(fixed_path),
            "outer_assignments": sha256(
                splits_dir / "outer_assignments.csv"
            ),
            "inner_assignments": sha256(
                splits_dir / "inner_assignments.csv"
            ),
        },
        "seed_offsets": seed_offsets,
        "backbone_epochs": args.backbone_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "head_epochs": args.head_epochs,
        "row_chunk_size": args.row_chunk_size,
        "global_top_fraction": args.global_top_fraction,
        "target_top_k": args.target_top_k,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return signature, payload


def matrix_frame(
    values: np.ndarray,
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    index_name: str,
) -> pd.DataFrame:
    """Restore a labelled wide matrix."""
    frame = pd.DataFrame(values, index=list(rows), columns=list(columns))
    frame.index.name = index_name
    return frame


def annotate_full_domain_ranks(
    predictions: pd.DataFrame,
    values: np.ndarray,
    domain_rows: tuple[str, ...],
    domain_columns: tuple[str, ...],
    score_direction: str,
    global_top_fraction: float,
    target_top_k: int,
) -> pd.DataFrame:
    """Add strict ranks."""
    ranked = values if score_direction == "lower_is_better" else -values
    row_index = {entity: index for index, entity in enumerate(domain_rows)}
    column_index = {
        entity: index for index, entity in enumerate(domain_columns)
    }
    rows = predictions["entity_a"].map(row_index).to_numpy(np.int64)
    columns = predictions["entity_b"].map(column_index).to_numpy(np.int64)
    flat_indices = np.ravel_multi_index((rows, columns), ranked.shape)
    n_top = max(1, int(math.ceil(ranked.size * global_top_fraction)))
    top_indices = np.argpartition(ranked.ravel(), n_top - 1)[:n_top]
    top_mask = np.zeros(ranked.size, dtype=bool)
    top_mask[top_indices] = True
    n_top_k = min(target_top_k, ranked.size)
    top_k_indices = np.argpartition(
        ranked.ravel(),
        n_top_k - 1,
    )[:n_top_k]
    top_k_mask = np.zeros(ranked.size, dtype=bool)
    top_k_mask[top_k_indices] = True

    within_rank = np.empty(len(predictions), dtype=np.int32)
    for row in np.unique(rows):
        order = np.argsort(ranked[row], kind="stable")
        inverse = np.empty(len(order), dtype=np.int32)
        inverse[order] = np.arange(1, len(order) + 1)
        selected = rows == row
        within_rank[selected] = inverse[columns[selected]]

    output = predictions.copy()
    output["matrix_score"] = values[rows, columns]
    output[f"in_global_top{global_top_fraction:g}"] = top_mask[
        flat_indices
    ]
    output[f"in_global_top{target_top_k}"] = top_k_mask[flat_indices]
    output["within_cid_rank"] = within_rank
    output[f"success_top{target_top_k}"] = (
        (output["label"] == 1)
        & (output["within_cid_rank"] <= target_top_k)
    )
    return output


def fit_one_model(
    context,
    base,
    fixed,
    parameters: Parameters,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    seed_state: dict,
    seed_offset: int,
    args: argparse.Namespace,
    signature: str,
    checkpoint: Path,
    score_domain,
    fit_candidate,
) -> None:
    """Fit and save one outer model."""
    print(
        f"START M1 repeat={parameters.repeat:02d} "
        f"outer={parameters.outer_fold:02d} seed={seed_offset}",
        flush=True,
    )
    context.standardizer = copy.deepcopy(seed_state["standardizer"])
    backbone_history, backbone_summary = fixed.train_fixed_weight(
        base,
        context,
        seed_state["backbone"],
        seed_state["scorer"],
        outer_train,
        parameters.graphban,
        parameters.boltz,
        args.backbone_epochs,
        args.steps_per_epoch,
        seed_state["seed"],
    )
    context.backbone_effective_weights = {
        context.models.architecture.auxiliary_feature: parameters.graphban,
        context.models.architecture.sparse_features[0]: parameters.boltz,
    }
    context.domain_features = None
    base.refresh_frozen_features(context)
    head_seed = seed_state["seed"] + 5000
    fitted = fit_candidate(
        context,
        outer_train,
        learnable_coefficient=True,
        residual_anchor_weight=parameters.anchor,
        seed=head_seed,
        epochs=args.head_epochs,
    )
    values = score_domain(
        context,
        fitted,
        row_chunk_size=args.row_chunk_size,
    )
    predictions, outer_metrics = base.evaluate_strategy(
        context,
        fitted,
        outer_test,
        parameters.identifier,
    )
    predictions.insert(0, "repeat", parameters.repeat)
    predictions.insert(1, "outer_fold", parameters.outer_fold)
    predictions.insert(2, "seed_offset", seed_offset)
    predictions.insert(3, "candidate_id", parameters.identifier)
    for position, (name, value) in enumerate(
        parameters.as_dict().items(),
        start=4,
    ):
        predictions.insert(position, name, value)
    domain_rows = tuple(context.relations.domain.entity_a)
    domain_columns = tuple(context.relations.domain.entity_b)
    predictions = annotate_full_domain_ranks(
        predictions,
        values,
        domain_rows,
        domain_columns,
        context.models.architecture.primary_direction,
        args.global_top_fraction,
        args.target_top_k,
    )
    metadata = {
        "run_signature": signature,
        "repeat": parameters.repeat,
        "outer_fold": parameters.outer_fold,
        "seed_offset": seed_offset,
        "candidate_id": parameters.identifier,
        **parameters.as_dict(),
        "joint_seed": seed_state["seed"],
        "head_seed": head_seed,
        "n_outer_train": len(outer_train),
        "n_outer_test": len(outer_test),
        "domain_shape": list(values.shape),
        "backbone_epochs_completed": len(backbone_history),
        "backbone_selected_epoch": backbone_summary["selected_epoch"],
        "head_epochs_completed": len(fitted.history),
        "primary_coefficient": fitted.primary_coefficient,
        "outer_label_metrics": outer_metrics,
        "completed_utc": utc_now(),
    }
    save_checkpoint(checkpoint, metadata, values, predictions)
    print(
        f"PASS  M1 repeat={parameters.repeat:02d} "
        f"outer={parameters.outer_fold:02d} seed={seed_offset}",
        flush=True,
    )


def strict_repeat_metrics(
    predictions: pd.DataFrame,
    evaluate_predictions,
    ranking_temperature: float,
    global_top_fraction: float,
    target_top_k: int,
) -> pd.DataFrame:
    """Compute strict OOF metrics."""
    rows = []
    top_column = f"in_global_top{global_top_fraction:g}"
    for (repeat, seed_offset), part in predictions.groupby(
        ["repeat", "seed_offset"],
        sort=True,
    ):
        metrics = evaluate_predictions(part, ranking_temperature)
        global_skill = (
            metrics["ranking_pooled_auprc"]
            - metrics["positive_prevalence"]
        ) / (1.0 - metrics["positive_prevalence"])
        within_skill = (
            1.0
            - metrics["pairwise_logistic_loss"] / np.log(2.0)
        )
        metrics.update(
            {
                "global_skill": global_skill,
                "within_skill": within_skill,
                "selection_score_j": (
                    0.5 * global_skill + 0.5 * within_skill
                ),
            }
        )
        positives = part["label"] == 1
        positive_in_top = int(part.loc[positives, top_column].sum())
        recall = positive_in_top / int(positives.sum())
        positive_in_top_k = int(
            part.loc[
                positives,
                f"in_global_top{target_top_k}",
            ].sum()
        )
        success = (
            part.loc[positives]
            .groupby("entity_a", sort=False)["within_cid_rank"]
            .min()
            .le(target_top_k)
        )
        rows.append(
            {
                "repeat": int(repeat),
                "seed_offset": int(seed_offset),
                **metrics,
                f"global_recall_top{global_top_fraction:g}": recall,
                f"global_ef_top{global_top_fraction:g}": (
                    recall / global_top_fraction
                ),
                "positive_in_global_top": positive_in_top,
                f"global_recall_top{target_top_k}": (
                    positive_in_top_k / int(positives.sum())
                ),
                f"positive_in_global_top{target_top_k}": positive_in_top_k,
                f"success_top{target_top_k}": float(success.mean()),
                f"n_cid_success_top{target_top_k}": int(success.sum()),
                "n_cid_with_positive": int(len(success)),
            }
        )
    return pd.DataFrame(rows)


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize OOF metrics."""
    numeric = metrics.select_dtypes(include=[np.number]).drop(
        columns=["repeat", "seed_offset"]
    )
    row = {"evaluation_source": "strict_nested_outer_oof"}
    for column in numeric:
        row[column] = float(numeric[column].mean())
        row[f"{column}_sd"] = float(numeric[column].std(ddof=1))
    return pd.DataFrame([row])


def write_progress(
    path: Path,
    status: str,
    signature: str,
    expected_models: int,
    completed_models: int,
    selected_folds: int,
    expected_folds: int,
    payload: dict,
) -> None:
    """Write M1 progress."""
    path.write_text(
        json.dumps(
            {
                "updated_utc": utc_now(),
                "status": status,
                "run_signature": signature,
                "selected_outer_folds": selected_folds,
                "expected_outer_folds": expected_folds,
                "completed_model_fits": completed_models,
                "expected_model_fits": expected_models,
                "signature_payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    """Generate fold-specific M1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--selected-parameters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--seed-offsets", default="0")
    parser.add_argument("--domain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone-epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=120)
    parser.add_argument("--row-chunk-size", type=int, default=16)
    parser.add_argument("--global-top-fraction", type=float, default=0.05)
    parser.add_argument("--target-top-k", type=int, default=100)
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    base_path = script_dir / BASE_DEPENDENCY
    fixed_path = script_dir / FIXED_DEPENDENCY
    config_path = Path(args.config).resolve()
    core_dir = Path(args.core_dir).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    selected_path = Path(args.selected_parameters).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_offsets = parse_int_values(args.seed_offsets)
    n_repeats, n_outer_folds = load_protocol(config_path)
    selected = read_selected(selected_path)
    signature, signature_payload = run_signature(
        script_path,
        config_path,
        splits_dir,
        base_path,
        fixed_path,
        args,
        seed_offsets,
    )
    manifest_path = output_dir / "run_manifest.json"

    sys.path.insert(0, str(core_dir))
    base = load_module(base_path, "cpi_mec_m1_base")
    fixed = load_module(fixed_path, "cpi_mec_m1_fixed")
    from evaluation import evaluate_predictions, outer_rows, score_domain
    from selection import audit_splits, load_materialized_splits
    from training import fit_candidate, prepare_context
    from transforms import row_z

    experiment_config = base.build_experiment_config(
        config_path,
        output_dir,
    )
    context = prepare_context(
        experiment_config,
        args.domain,
        args.device,
        backbone_epochs=0,
        steps_per_epoch=args.steps_per_epoch,
    )
    splits = load_materialized_splits(splits_dir)
    audit_splits(splits, context)

    requested = selected.copy()
    if args.repeat is not None:
        requested = requested.loc[requested["repeat"] == args.repeat]
    if args.outer_fold is not None:
        requested = requested.loc[
            requested["outer_fold"] == args.outer_fold
        ]
    for row in requested.itertuples(index=False):
        parameters = parameters_from_row(row)
        outer_train, outer_test = outer_rows(
            splits,
            parameters.repeat,
            parameters.outer_fold,
        )
        joint_seed = (
            context.settings.seed
            + parameters.repeat * 100000
            + parameters.outer_fold * 10000
            + 2700
        )
        seed_states = fixed.build_seed_states(
            context,
            base,
            seed_offsets,
            joint_seed,
        )
        for seed_offset in seed_offsets:
            checkpoint = checkpoint_dir / checkpoint_name(
                parameters,
                seed_offset,
            )
            if checkpoint.is_file():
                load_checkpoint(checkpoint, signature)
                print(f"SKIP  complete {checkpoint.name}", flush=True)
                continue
            fit_one_model(
                context,
                base,
                fixed,
                parameters,
                outer_train,
                outer_test,
                seed_states[seed_offset],
                seed_offset,
                args,
                signature,
                checkpoint,
                score_domain,
                fit_candidate,
            )

    expected_folds = n_repeats * n_outer_folds
    selected_keys = set(
        zip(selected["repeat"].astype(int), selected["outer_fold"].astype(int))
    )
    complete_selection = len(selected_keys) == expected_folds
    expected_paths = []
    if complete_selection:
        for row in selected.itertuples(index=False):
            parameters = parameters_from_row(row)
            expected_paths.extend(
                checkpoint_dir / checkpoint_name(parameters, seed_offset)
                for seed_offset in seed_offsets
            )
    completed_count = sum(path.is_file() for path in expected_paths)
    expected_count = expected_folds * len(seed_offsets)
    if not complete_selection or completed_count != expected_count:
        write_progress(
            manifest_path,
            "partial",
            signature,
            expected_count,
            completed_count,
            len(selected_keys),
            expected_folds,
            signature_payload,
        )
        print(
            f"PARTIAL  selected folds={len(selected_keys)}/{expected_folds}, "
            f"model fits={completed_count}/{expected_count}",
            flush=True,
        )
        return

    matrix_sum = None
    prediction_tables = []
    model_rows = []
    for path in expected_paths:
        checkpoint = load_checkpoint(path, signature)
        values = checkpoint["values"].astype(np.float64)
        matrix_sum = values if matrix_sum is None else matrix_sum + values
        prediction_tables.append(checkpoint["predictions"])
        model_rows.append(checkpoint["metadata"])
    pre_z = (matrix_sum / expected_count).astype(np.float32)
    row_z_values, _, _ = row_z(pre_z, "lower_is_better")
    predictions = pd.concat(prediction_tables, ignore_index=True)
    repeat_metrics = strict_repeat_metrics(
        predictions,
        evaluate_predictions,
        context.settings.ranking_temperature,
        args.global_top_fraction,
        args.target_top_k,
    )
    summary = metric_summary(repeat_metrics)
    ensemble_predictions = (
        predictions.groupby(
            ["relation_row", "entity_a", "entity_b", "label"],
            as_index=False,
            sort=False,
        )
        .agg(
            probability=("probability", "mean"),
            ranking_score=("ranking_score", "mean"),
            matrix_score=("matrix_score", "mean"),
            within_cid_rank_mean=("within_cid_rank", "mean"),
            global_top_frequency=(
                f"in_global_top{args.global_top_fraction:g}",
                "mean",
            ),
            n_oof_predictions=("ranking_score", "size"),
        )
    )

    domain_rows = tuple(context.relations.domain.entity_a)
    domain_columns = tuple(context.relations.domain.entity_b)
    primary_role = context.models.architecture.primary_input
    contract = context.config["inputs"][primary_role]["contract"]
    index_name = contract.get(
        "entity_a_column",
        contract.get("row_entity_column", "entity_a"),
    )
    matrix_frame(
        pre_z,
        domain_rows,
        domain_columns,
        index_name,
    ).to_csv(
        output_dir / "M1_nested_CV_ensemble_preZ_matrix.csv",
        float_format=CSV_FLOAT_FORMAT,
    )
    matrix_frame(
        row_z_values,
        domain_rows,
        domain_columns,
        index_name,
    ).to_csv(
        output_dir / "M1_nested_CV_ensemble_rowZ_matrix.csv",
        float_format=CSV_FLOAT_FORMAT,
    )
    predictions.to_csv(
        output_dir / "outer_label_oof_predictions.csv",
        index=False,
    )
    ensemble_predictions.to_csv(
        output_dir / "outer_label_oof_ensemble_predictions.csv",
        index=False,
    )
    repeat_metrics.to_csv(
        output_dir / "outer_label_oof_metrics_by_repeat.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "outer_label_oof_metric_summary.csv",
        index=False,
    )
    pd.DataFrame(model_rows).drop(
        columns=["outer_label_metrics"]
    ).to_csv(
        output_dir / "model_fit_summary.csv",
        index=False,
    )
    selected.to_csv(
        output_dir / "selected_parameters_by_outer_fold.csv",
        index=False,
    )

    relation_counts = predictions.groupby("relation_row").size().to_numpy()
    expected_relation_count = n_repeats * len(seed_offsets)
    acceptance = pd.DataFrame(
        [
            ("selected_outer_folds", len(selected_keys) == expected_folds, len(selected_keys)),
            ("model_fit_count", len(expected_paths) == expected_count, len(expected_paths)),
            ("pre_z_shape", pre_z.shape == (len(domain_rows), len(domain_columns)), str(pre_z.shape)),
            ("pre_z_finite", bool(np.isfinite(pre_z).all()), int(np.isfinite(pre_z).sum())),
            ("row_z_finite", bool(np.isfinite(row_z_values).all()), int(np.isfinite(row_z_values).sum())),
            (
                "row_z_mean",
                bool(np.max(np.abs(row_z_values.mean(axis=1))) <= 5e-6),
                float(np.max(np.abs(row_z_values.mean(axis=1)))),
            ),
            (
                "row_z_population_sd",
                bool(
                    np.max(
                        np.abs(row_z_values.std(axis=1, ddof=0) - 1.0)
                    )
                    <= 5e-6
                ),
                float(
                    np.max(
                        np.abs(row_z_values.std(axis=1, ddof=0) - 1.0)
                    )
                ),
            ),
            (
                "strict_oof_count_per_label",
                bool(np.all(relation_counts == expected_relation_count)),
                f"{relation_counts.min()}..{relation_counts.max()}",
            ),
        ],
        columns=["check", "passed", "value"],
    )
    acceptance.to_csv(output_dir / "matrix_acceptance.csv", index=False)
    if not acceptance["passed"].all():
        raise RuntimeError("One or more M1 acceptance checks failed")

    write_progress(
        manifest_path,
        "completed",
        signature,
        expected_count,
        expected_count,
        expected_folds,
        expected_folds,
        signature_payload,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "module": script_path.name,
            "module_version": SCRIPT_VERSION,
            "module_release": SCRIPT_RELEASE,
            "matrix_definition": (
                "Mean complete-domain prediction across every nested-CV "
                "outer model and configured seed."
            ),
            "formal_performance_source": (
                "Strict outer-test OOF label predictions only."
            ),
            "domain_shape": [len(domain_rows), len(domain_columns)],
            "n_outer_models": expected_count,
            "n_strict_oof_predictions": len(predictions),
            "selected_parameters_source": str(selected_path),
            "selected_parameters_sha256": sha256(selected_path),
            "outputs": {
                "pre_z": "M1_nested_CV_ensemble_preZ_matrix.csv",
                "row_z": "M1_nested_CV_ensemble_rowZ_matrix.csv",
                "strict_oof_predictions": "outer_label_oof_predictions.csv",
                "strict_oof_metric_summary": (
                    "outer_label_oof_metric_summary.csv"
                ),
            },
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS  M1 matrix: {len(domain_rows)} x {len(domain_columns)}",
        flush=True,
    )
    print(f"PASS  strict OOF predictions: {len(predictions)}", flush=True)
    print(f"PASS  output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
